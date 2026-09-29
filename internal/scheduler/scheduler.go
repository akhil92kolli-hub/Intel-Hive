package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/pipeline"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
)

type ShardAssignment struct {
	ModelVersion string
	ShardID      string
	Layer        model.LayerRange
	WorkerID     string
}

type AssignmentPlan struct {
	JobID     string
	ModelID   string
	Shards    []ShardAssignment
	WorkerIDs []string
}

type GenerationResult struct {
	JobID           string
	TokenIDs        []uint32
	Text            []byte
	ExecutionTimeMs int64
	TokensPerSec    float64
}

type Scheduler struct {
	registry *Registry
	workers  map[string]worker.Runtime
	mu       sync.Mutex
}

func New(registry *Registry) *Scheduler {
	if registry == nil {
		registry = NewRegistry()
	}
	return &Scheduler{registry: registry, workers: make(map[string]worker.Runtime)}
}

func (s *Scheduler) Registry() *Registry { return s.registry }

// RegisterWorker records capabilities and optionally attaches an executable runtime.
func (s *Scheduler) RegisterWorker(record WorkerRecord, runtime worker.Runtime) error {
	if err := s.registry.Register(record); err != nil {
		return err
	}
	if runtime != nil {
		s.mu.Lock()
		s.workers[record.ID] = runtime
		s.mu.Unlock()
	}
	return nil
}

func (s *Scheduler) UnregisterWorker(id string) error {
	s.mu.Lock()
	delete(s.workers, id)
	s.mu.Unlock()
	return s.registry.Unregister(id)
}

// BuildPlan selects the highest-scoring eligible worker for each shard.
// Selection is deterministic: score descending, then worker ID ascending.
func (s *Scheduler) BuildPlan(jobID, modelID string, shards []model.ModelShard) (AssignmentPlan, error) {
	if jobID == "" || modelID == "" {
		return AssignmentPlan{}, fmt.Errorf("job ID and model ID are required")
	}
	if len(shards) == 0 {
		return AssignmentPlan{}, fmt.Errorf("no model shards supplied")
	}

	plan := AssignmentPlan{JobID: jobID, ModelID: modelID, Shards: make([]ShardAssignment, 0, len(shards))}
	used := make(map[string]bool)
	for _, shard := range shards {
		if shard.Version != shards[0].Version {
			return AssignmentPlan{}, fmt.Errorf("all shards must use the same model version")
		}
		candidates := s.registry.List()
		eligible := make([]WorkerRecord, 0)
		for _, worker := range candidates {
			if worker.IsEligible() && worker.CanServeModelShard(shard) && !used[worker.ID] {
				eligible = append(eligible, worker)
			}
		}
		if len(eligible) == 0 {
			return AssignmentPlan{}, fmt.Errorf("no eligible worker for shard %s (%d-%d)", shard.ID, shard.Layers.Start, shard.Layers.End)
		}
		sort.Slice(eligible, func(i, j int) bool {
			left, right := eligible[i], eligible[j]
			if left.Score() == right.Score() {
				return left.ID < right.ID
			}
			return left.Score() > right.Score()
		})
		selected := eligible[0]
		used[selected.ID] = true
		plan.Shards = append(plan.Shards, ShardAssignment{ModelVersion: shard.Version, ShardID: shard.ID, Layer: shard.Layers, WorkerID: selected.ID})
		plan.WorkerIDs = append(plan.WorkerIDs, selected.ID)
	}
	return plan, nil
}

// Execute runs a plan and, if a worker fails, marks it offline, rebuilds the
// complete assignment, and retries the request without requiring a client retry.
func (s *Scheduler) Execute(ctx context.Context, plan AssignmentPlan, payload []byte) (pipeline.PipelineResult, error) {
	if len(plan.Shards) == 0 {
		return pipeline.PipelineResult{}, fmt.Errorf("cannot execute an empty assignment plan")
	}

	currentPlan := plan
	for attempts := 0; attempts <= s.registry.Count(); attempts++ {
		result, err := s.executePlan(ctx, currentPlan, payload)
		if err == nil {
			return result, nil
		}

		var executionError *pipeline.ShardExecutionError
		if !errors.As(err, &executionError) {
			return result, err
		}
		if !errors.Is(executionError, worker.ErrUnavailable) {
			return result, err
		}
		if offlineErr := s.MarkOffline(executionError.WorkerID); offlineErr != nil {
			return result, fmt.Errorf("%w; failed to mark worker %s offline: %v", err, executionError.WorkerID, offlineErr)
		}
		if attempts == s.registry.Count() {
			return result, fmt.Errorf("%w; no replacement attempt remains", err)
		}

		shards := make([]model.ModelShard, len(plan.Shards))
		for i, assignment := range plan.Shards {
			shards[i] = model.ModelShard{
				ID: assignment.ShardID, Version: assignment.ModelVersion, ModelID: plan.ModelID, Layers: assignment.Layer,
			}
		}
		currentPlan, err = s.BuildPlan(plan.JobID, plan.ModelID, shards)
		if err != nil {
			return result, fmt.Errorf("%w; could not rebuild pipeline: %v", executionError, err)
		}
	}
	return pipeline.PipelineResult{}, fmt.Errorf("pipeline retry limit exceeded")
}

// Generate runs one prefill pass and then one decode pass per emitted token.
// Each pass visits every assigned shard; the final shard samples the next
// token, which is fed into the first shard on the following decode pass.
func (s *Scheduler) Generate(
	ctx context.Context,
	plan AssignmentPlan,
	prompt string,
	maxTokens int,
) (GenerationResult, error) {
	if len(plan.Shards) == 0 {
		return GenerationResult{}, fmt.Errorf("cannot generate with an empty assignment plan")
	}
	if strings.TrimSpace(prompt) == "" {
		return GenerationResult{}, fmt.Errorf("prompt is required")
	}
	if maxTokens <= 0 || maxTokens > 256 {
		return GenerationResult{}, fmt.Errorf("max_tokens must be between 1 and 256")
	}

	currentPlan := plan
	retries := s.registry.Count()
	for attempt := 0; attempt <= retries; attempt++ {
		sequenceID := fmt.Sprintf("%s:%d", plan.JobID, attempt)
		generated, err := s.generateOnce(
			ctx, currentPlan, prompt, maxTokens, sequenceID,
		)
		cleanupCtx, cancelCleanup := context.WithTimeout(context.Background(), 2*time.Second)
		cleanupErr := s.endInferenceSequence(cleanupCtx, currentPlan, sequenceID, err == nil)
		cancelCleanup()
		if cleanupErr != nil {
			if err != nil {
				return GenerationResult{}, fmt.Errorf("%w; failed to end inference sequence: %v", err, cleanupErr)
			}
			return GenerationResult{}, fmt.Errorf("failed to end inference sequence: %w", cleanupErr)
		}
		if err == nil {
			return generated, nil
		}

		var executionError *pipeline.ShardExecutionError
		if !errors.As(err, &executionError) || !errors.Is(executionError, worker.ErrUnavailable) {
			return GenerationResult{}, err
		}
		if offlineErr := s.MarkOffline(executionError.WorkerID); offlineErr != nil {
			return GenerationResult{}, fmt.Errorf("%w; failed to mark worker %s offline: %v", err, executionError.WorkerID, offlineErr)
		}
		if attempt == retries {
			return GenerationResult{}, fmt.Errorf("%w; no replacement attempt remains", err)
		}

		shards := make([]model.ModelShard, len(plan.Shards))
		for i, assignment := range plan.Shards {
			shards[i] = model.ModelShard{
				ID: assignment.ShardID, Version: assignment.ModelVersion, ModelID: plan.ModelID, Layers: assignment.Layer,
			}
		}
		currentPlan, err = s.BuildPlan(plan.JobID, plan.ModelID, shards)
		if err != nil {
			return GenerationResult{}, fmt.Errorf("%w; could not rebuild pipeline: %v", executionError, err)
		}
	}
	return GenerationResult{}, fmt.Errorf("generation retry limit exceeded")
}

func (s *Scheduler) endInferenceSequence(
	ctx context.Context,
	plan AssignmentPlan,
	sequenceID string,
	completed bool,
) error {
	s.mu.Lock()
	workers := make(map[string]worker.Runtime, len(s.workers))
	for id, runtime := range s.workers {
		workers[id] = runtime
	}
	s.mu.Unlock()

	var firstErr error
	for _, workerID := range plan.WorkerIDs {
		lifecycle, ok := workers[workerID].(worker.SequenceLifecycle)
		if !ok {
			continue
		}
		if err := lifecycle.EndSequence(ctx, plan.JobID, sequenceID, completed); err != nil {
			if errors.Is(err, worker.ErrUnavailable) {
				continue
			}
			if firstErr == nil {
				firstErr = fmt.Errorf("worker %s: %w", workerID, err)
			}
		}
	}
	return firstErr
}

func (s *Scheduler) generateOnce(
	ctx context.Context,
	plan AssignmentPlan,
	prompt string,
	maxTokens int,
	sequenceID string,
) (GenerationResult, error) {
	started := time.Now()
	generated := GenerationResult{JobID: plan.JobID, TokenIDs: make([]uint32, 0, maxTokens)}
	var nextToken []uint32
	for position := uint32(0); len(generated.TokenIDs) < maxTokens; position++ {
		phase := "DECODE"
		stepPrompt := ""
		stepTokenIDs := nextToken
		if position == 0 {
			phase = "PREFILL"
			stepPrompt = prompt
			stepTokenIDs = nil
		}

		stepResult, err := s.executeInferenceStep(
			ctx, plan, sequenceID, phase, position, stepPrompt, stepTokenIDs,
		)
		if err != nil {
			return GenerationResult{}, err
		}
		if stepResult.EndOfSequence {
			break
		}
		token := *stepResult.SampledTokenID
		generated.TokenIDs = append(generated.TokenIDs, token)
		generated.Text = append(generated.Text, stepResult.GeneratedText...)
		nextToken = []uint32{token}
	}
	generated.ExecutionTimeMs = time.Since(started).Milliseconds()
	if generated.ExecutionTimeMs > 0 {
		generated.TokensPerSec = float64(len(generated.TokenIDs)) * 1000 / float64(generated.ExecutionTimeMs)
	}
	return generated, nil
}

func (s *Scheduler) executeInferenceStep(
	ctx context.Context,
	plan AssignmentPlan,
	sequenceID, phase string,
	position uint32,
	prompt string,
	inputTokenIDs []uint32,
) (worker.JobResult, error) {
	s.mu.Lock()
	workers := make(map[string]worker.Runtime, len(s.workers))
	for id, runtime := range s.workers {
		workers[id] = runtime
	}
	s.mu.Unlock()

	executor := pipeline.NewPipelineExecutor(workers)
	assignments := make([]pipeline.ShardAssignment, len(plan.Shards))
	for i, assignment := range plan.Shards {
		assignments[i] = pipeline.ShardAssignment{
			ModelVersion: assignment.ModelVersion, ShardID: assignment.ShardID, Layer: assignment.Layer, WorkerID: assignment.WorkerID,
		}
	}
	return executor.ExecuteInferenceStep(
		ctx, plan.JobID, plan.ModelID, sequenceID, phase, position,
		prompt, inputTokenIDs, assignments,
	)
}

func (s *Scheduler) executePlan(
	ctx context.Context,
	plan AssignmentPlan,
	payload []byte,
) (pipeline.PipelineResult, error) {
	s.mu.Lock()
	workers := make(map[string]worker.Runtime, len(s.workers))
	for id, runtime := range s.workers {
		workers[id] = runtime
	}
	s.mu.Unlock()

	executor := pipeline.NewPipelineExecutor(workers)
	shards := make([]pipeline.ShardAssignment, len(plan.Shards))
	for i, assignment := range plan.Shards {
		shards[i] = pipeline.ShardAssignment{
			ModelVersion: assignment.ModelVersion, ShardID: assignment.ShardID, Layer: assignment.Layer, WorkerID: assignment.WorkerID,
		}
	}
	return executor.Execute(ctx, pipeline.DistributedJob{
		ID: plan.JobID, ModelID: plan.ModelID, RequestID: plan.JobID,
		Shards: shards, InitialPayload: payload,
	})
}

func (s *Scheduler) MarkBusy(plan AssignmentPlan) error {
	for _, id := range plan.WorkerIDs {
		if err := s.registry.UpdateState(id, WorkerStateBusy); err != nil {
			return err
		}
	}
	return nil
}

func (s *Scheduler) MarkReady(plan AssignmentPlan) error {
	for _, id := range plan.WorkerIDs {
		if err := s.registry.UpdateState(id, WorkerStateReady); err != nil {
			return err
		}
	}
	return nil
}

// MarkOffline removes a worker from future plans without deleting its record.
func (s *Scheduler) MarkOffline(id string) error {
	return s.registry.UpdateState(id, WorkerStateOffline)
}

func (s *Scheduler) ExpireWorkers(timeout time.Duration) []string {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	cutoff := time.Now().UTC().Add(-timeout)
	offline := make([]string, 0)
	for _, worker := range s.registry.List() {
		if worker.LastSeen.Before(cutoff) && worker.State != WorkerStateOffline {
			if err := s.MarkOffline(worker.ID); err == nil {
				offline = append(offline, worker.ID)
			}
		}
	}
	return offline
}
