package pipeline

import (
	"context"
	"fmt"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"sort"
	"strings"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/transport"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
)

// DistributedJob represents a complete inference request split across workers.
type DistributedJob struct {
	ID             string
	ModelID        string
	RequestID      string
	Shards         []ShardAssignment
	InitialPayload []byte
}

// ShardAssignment maps a layer range to a specific worker.
type ShardAssignment struct {
	ModelVersion string
	ShardID      string
	Layer        model.LayerRange
	WorkerID     string
}

// ShardMetric tracks execution metrics for a single shard.
type ShardMetric struct {
	ShardID         string
	WorkerID        string
	LayerRange      model.LayerRange
	ExecutionTimeMs int64
	SerializationMs int64
	ActivationSizeB int64
	TokensPerSec    float64
}

// PipelineResult contains the distributed execution result.
type PipelineResult struct {
	JobID           string
	FinalOutput     []byte
	ExecutionTimeMs int64
	TokensPerSec    float64
	ShardMetrics    map[string]ShardMetric
}

type ShardExecutionError struct {
	ShardID  string
	WorkerID string
	Err      error
}

func (e *ShardExecutionError) Error() string {
	return fmt.Sprintf("shard %s execution on worker %s failed: %v", e.ShardID, e.WorkerID, e.Err)
}

func (e *ShardExecutionError) Unwrap() error { return e.Err }

// PipelineExecutor coordinates distributed inference across worker runtimes.
type PipelineExecutor struct {
	Workers   map[string]worker.Runtime
	Transport *transport.InMemoryTransport
}

func NewPipelineExecutor[T worker.Runtime](runtimes map[string]T) *PipelineExecutor {
	workers := make(map[string]worker.Runtime, len(runtimes))
	for id, runtime := range runtimes {
		workers[id] = runtime
	}
	return &PipelineExecutor{
		Workers:   workers,
		Transport: transport.NewInMemoryTransport(),
	}
}

// Execute runs a complete distributed inference pipeline.
func (e *PipelineExecutor) Execute(ctx context.Context, job DistributedJob) (PipelineResult, error) {
	startTime := time.Now()
	result := PipelineResult{
		JobID:        job.ID,
		ShardMetrics: make(map[string]ShardMetric),
	}

	// Sort shards by layer range for sequential execution.
	sort.Slice(job.Shards, func(i, j int) bool {
		return job.Shards[i].Layer.Start < job.Shards[j].Layer.Start
	})

	// Validate shard assignment.
	if err := e.validateAssignment(job); err != nil {
		return result, err
	}

	// Execute shards in sequence.
	currentPayload := job.InitialPayload
	for i, assignment := range job.Shards {
		runtime, ok := e.Workers[assignment.WorkerID]
		if !ok {
			return result, &ShardExecutionError{
				ShardID:  assignment.ShardID,
				WorkerID: assignment.WorkerID,
				Err:      fmt.Errorf("%w: runtime for %s not found", worker.ErrUnavailable, assignment.WorkerID),
			}
		}

		shardStartTime := time.Now()
		jobExec := worker.Job{
			ID:       job.ID,
			ModelID:  job.ModelID,
			ShardID:  assignment.ShardID,
			Layers:   assignment.Layer,
			Payload:  currentPayload,
			Sequence: uint64(i),
		}

		jobResult, err := runtime.Execute(ctx, jobExec)
		if err != nil {
			return result, &ShardExecutionError{
				ShardID:  assignment.ShardID,
				WorkerID: assignment.WorkerID,
				Err:      err,
			}
		}

		shardDuration := time.Since(shardStartTime)
		result.ShardMetrics[assignment.ShardID] = ShardMetric{
			ShardID:         assignment.ShardID,
			WorkerID:        assignment.WorkerID,
			LayerRange:      assignment.Layer,
			ExecutionTimeMs: shardDuration.Milliseconds(),
			TokensPerSec:    jobResult.TokensPerSec,
			ActivationSizeB: int64(len(jobResult.Output)),
		}

		// For sequential execution, the output becomes the next input.
		currentPayload = jobResult.Output
	}

	result.FinalOutput = currentPayload
	result.ExecutionTimeMs = time.Since(startTime).Milliseconds()
	result.TokensPerSec = 10.0

	return result, nil
}

// ExecuteInferenceStep sends one prefill or decode pass across the assigned
// shards. The first shard receives prompt text or token IDs; subsequent shards
// receive the preceding shard's activation.
func (e *PipelineExecutor) ExecuteInferenceStep(
	ctx context.Context,
	jobID, modelID, sequenceID, phase string,
	position uint32,
	prompt string,
	inputTokenIDs []uint32,
	assignments []ShardAssignment,
) (worker.JobResult, error) {
	if strings.TrimSpace(jobID) == "" || strings.TrimSpace(modelID) == "" || strings.TrimSpace(sequenceID) == "" {
		return worker.JobResult{}, fmt.Errorf("job, model, and sequence IDs are required")
	}
	if phase != "PREFILL" && phase != "DECODE" {
		return worker.JobResult{}, fmt.Errorf("unsupported inference phase %q", phase)
	}
	if phase == "PREFILL" && position != 0 {
		return worker.JobResult{}, fmt.Errorf("prefill position must be zero")
	}
	if phase == "DECODE" && position == 0 {
		return worker.JobResult{}, fmt.Errorf("decode position must be greater than zero")
	}
	if (prompt == "") == (len(inputTokenIDs) == 0) {
		return worker.JobResult{}, fmt.Errorf("exactly one of prompt or input token IDs is required for the first shard")
	}
	if phase == "DECODE" && prompt != "" {
		return worker.JobResult{}, fmt.Errorf("decode steps cannot contain prompt text")
	}
	if phase == "DECODE" && len(inputTokenIDs) != 1 {
		return worker.JobResult{}, fmt.Errorf("decode steps require exactly one input token")
	}
	if len(assignments) == 0 {
		return worker.JobResult{}, fmt.Errorf("no shards assigned")
	}

	ordered := append([]ShardAssignment(nil), assignments...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Layer.Start < ordered[j].Layer.Start
	})
	distributedJob := DistributedJob{ID: jobID, ModelID: modelID, Shards: ordered}
	if err := e.validateAssignment(distributedJob); err != nil {
		return worker.JobResult{}, err
	}

	var previous *activation.Envelope
	for index, assignment := range ordered {
		runtime := e.Workers[assignment.WorkerID]
		input := worker.Job{
			ID: jobID, RequestID: jobID, ModelVersion: assignment.ModelVersion, WorkerID: assignment.WorkerID, ModelID: modelID, ShardID: assignment.ShardID,
			Layers: assignment.Layer, Phase: phase, SequenceID: sequenceID,
			Position: position, FinalShard: index == len(ordered)-1,
			Sequence: uint64(position),
		}
		if index > 0 {
			input.PreviousWorker = ordered[index-1].WorkerID
		}
		if index+1 < len(ordered) {
			input.NextWorker = ordered[index+1].WorkerID
		}
		if input.ModelVersion == "" {
			return worker.JobResult{}, fmt.Errorf("model version is required")
		}
		if index == 0 {
			input.Prompt = prompt
			input.InputTokenIDs = append([]uint32(nil), inputTokenIDs...)
		} else {
			input.Activation = previous
		}

		result, err := runtime.Execute(ctx, input)
		if err != nil {
			return worker.JobResult{}, &ShardExecutionError{
				ShardID: assignment.ShardID, WorkerID: assignment.WorkerID, Err: err,
			}
		}
		if result.JobID != "" && result.JobID != jobID {
			return worker.JobResult{}, fmt.Errorf("worker %s returned a result for job %s, expected %s", assignment.WorkerID, result.JobID, jobID)
		}
		if index < len(ordered)-1 {
			if result.Activation == nil || result.SampledTokenID != nil || result.EndOfSequence {
				return worker.JobResult{}, fmt.Errorf("non-final shard %s must return only a non-empty activation", assignment.ShardID)
			}
			if err := result.Activation.ValidateBoundary(jobID, jobID, sequenceID, modelID, input.ModelVersion, assignment.WorkerID, input.NextWorker, assignment.Layer.End, position); err != nil {
				return worker.JobResult{}, err
			}
			previous = result.Activation
			continue
		}
		if (result.SampledTokenID != nil) == result.EndOfSequence || result.Activation != nil {
			return worker.JobResult{}, fmt.Errorf("final shard %s must return exactly one sampled token or EOS", assignment.ShardID)
		}
		return result, nil
	}
	return worker.JobResult{}, fmt.Errorf("inference step produced no final shard result")
}

func (e *PipelineExecutor) validateAssignment(job DistributedJob) error {
	if len(job.Shards) == 0 {
		return fmt.Errorf("no shards assigned")
	}

	sort.Slice(job.Shards, func(i, j int) bool {
		return job.Shards[i].Layer.Start < job.Shards[j].Layer.Start
	})

	// Check for contiguous coverage.
	expected := 0
	for _, shard := range job.Shards {
		if shard.Layer.Start != expected {
			return fmt.Errorf("layer gap detected at %d", expected)
		}
		expected = shard.Layer.End + 1
	}

	// Verify all assigned workers exist.
	for _, shard := range job.Shards {
		if _, ok := e.Workers[shard.WorkerID]; !ok {
			return fmt.Errorf("worker %s not available", shard.WorkerID)
		}
	}

	return nil
}

// SimulateFullPipeline demonstrates a complete 3-worker distributed inference.
func SimulateFullPipeline(ctx context.Context) (PipelineResult, error) {
	workers := mock.NewMockWorkers()
	runtimes := make(map[string]worker.Runtime, len(workers))
	for _, w := range workers {
		if err := w.Start(ctx); err != nil {
			return PipelineResult{}, err
		}
		if err := w.Register(ctx); err != nil {
			return PipelineResult{}, err
		}
	}
	for id, w := range workers {
		runtimes[id] = w
	}

	executor := NewPipelineExecutor(runtimes)

	job := DistributedJob{
		ID:             "job-123",
		ModelID:        "qwen2.5-3b",
		RequestID:      "request-456",
		InitialPayload: []byte("What is artificial intelligence?"),
		Shards: []ShardAssignment{
			{
				ShardID:  "shard-0-9",
				Layer:    model.LayerRange{Start: 0, End: 9},
				WorkerID: "worker-a",
			},
			{
				ShardID:  "shard-10-19",
				Layer:    model.LayerRange{Start: 10, End: 19},
				WorkerID: "worker-b",
			},
			{
				ShardID:  "shard-20-29",
				Layer:    model.LayerRange{Start: 20, End: 29},
				WorkerID: "worker-c",
			},
		},
	}

	return executor.Execute(ctx, job)
}
