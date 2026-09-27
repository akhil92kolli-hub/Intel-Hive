package scheduler

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/pipeline"
)

type ShardAssignment struct {
	ShardID  string
	Layer    model.LayerRange
	WorkerID string
}

type AssignmentPlan struct {
	JobID     string
	ModelID   string
	Shards    []ShardAssignment
	WorkerIDs []string
}

type Scheduler struct {
	registry *Registry
	workers  map[string]*mock.MockWorker
	mu       sync.Mutex
}

func New(registry *Registry) *Scheduler {
	if registry == nil {
		registry = NewRegistry()
	}
	return &Scheduler{registry: registry, workers: make(map[string]*mock.MockWorker)}
}

func (s *Scheduler) Registry() *Registry { return s.registry }

// RegisterWorker records capabilities and optionally attaches a mock runtime.
func (s *Scheduler) RegisterWorker(record WorkerRecord, runtime *mock.MockWorker) error {
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
		candidates := s.registry.List()
		eligible := make([]WorkerRecord, 0)
		for _, worker := range candidates {
			if worker.IsEligible() && worker.CanServeLayer(shard.Layers) && !used[worker.ID] {
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
		plan.Shards = append(plan.Shards, ShardAssignment{ShardID: shard.ID, Layer: shard.Layers, WorkerID: selected.ID})
		plan.WorkerIDs = append(plan.WorkerIDs, selected.ID)
	}
	return plan, nil
}

// Execute runs a plan through the attached mock runtimes. Real workers will use
// the same plan but execute through the transport protocol instead.
func (s *Scheduler) Execute(ctx context.Context, plan AssignmentPlan, payload []byte) (pipeline.PipelineResult, error) {
	s.mu.Lock()
	workers := make(map[string]*mock.MockWorker, len(s.workers))
	for id, worker := range s.workers {
		workers[id] = worker
	}
	s.mu.Unlock()

	executor := pipeline.NewPipelineExecutor(workers)
	shards := make([]pipeline.ShardAssignment, len(plan.Shards))
	for i, assignment := range plan.Shards {
		shards[i] = pipeline.ShardAssignment{ShardID: assignment.ShardID, Layer: assignment.Layer, WorkerID: assignment.WorkerID}
	}
	return executor.Execute(ctx, pipeline.DistributedJob{ID: plan.JobID, ModelID: plan.ModelID, RequestID: plan.JobID, Shards: shards, InitialPayload: payload})
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
