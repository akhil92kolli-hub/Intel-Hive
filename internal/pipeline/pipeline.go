package pipeline

import (
    "context"
    "fmt"
    "sort"
    "time"

    "github.com/akhil92kolli-hub/Intel-Hive/internal/model"
    "github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
    "github.com/akhil92kolli-hub/Intel-Hive/internal/transport"
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
    ShardID  string
    Layer    model.LayerRange
    WorkerID string
}

// ShardMetric tracks execution metrics for a single shard.
type ShardMetric struct {
    ShardID           string
    WorkerID          string
    LayerRange        model.LayerRange
    ExecutionTimeMs   int64
    SerializationMs   int64
    ActivationSizeB   int64
    TokensPerSec      float64
}

// PipelineResult contains the distributed execution result.
type PipelineResult struct {
    JobID           string
    FinalOutput     []byte
    ExecutionTimeMs int64
    TokensPerSec    float64
    ShardMetrics    map[string]ShardMetric
}

// PipelineExecutor coordinates distributed inference across mock workers.
type PipelineExecutor struct {
    Workers   map[string]*mock.MockWorker
    Transport *transport.InMemoryTransport
}

func NewPipelineExecutor(workers map[string]*mock.MockWorker) *PipelineExecutor {
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
        worker, ok := e.Workers[assignment.WorkerID]
        if !ok {
            return result, fmt.Errorf("worker %s not found", assignment.WorkerID)
        }

        shardStartTime := time.Now()
        jobExec := mock.Job{
            ID:       job.ID,
            ModelID:  job.ModelID,
            Layers:   assignment.Layer,
            Payload:  currentPayload,
            Sequence: uint64(i),
        }

        jobResult, err := worker.Execute(ctx, jobExec)
        if err != nil {
            return result, fmt.Errorf("shard %s execution failed: %v", assignment.ShardID, err)
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
    for _, w := range workers {
        if err := w.Start(ctx); err != nil {
            return PipelineResult{}, err
        }
        if err := w.Register(ctx); err != nil {
            return PipelineResult{}, err
        }
    }

    executor := NewPipelineExecutor(workers)

    job := DistributedJob{
        ID:        "job-123",
        ModelID:   "qwen2.5-3b",
        RequestID: "request-456",
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
