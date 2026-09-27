package pipeline

import (
    "context"
    "testing"

    "github.com/akhil92kolli-hub/Intel-Hive/internal/model"
    "github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
)

func TestPipelineFullExecution(t *testing.T) {
    ctx := context.Background()
    workers := mock.NewMockWorkers()
    for _, w := range workers {
        if err := w.Start(ctx); err != nil {
            t.Fatalf("start failed: %v", err)
        }
    }

    executor := NewPipelineExecutor(workers)
    job := DistributedJob{
        ID:        "job-full",
        ModelID:   "qwen2.5-3b",
        RequestID: "request-full",
        InitialPayload: []byte("test prompt"),
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

    result, err := executor.Execute(ctx, job)
    if err != nil {
        t.Fatalf("execute failed: %v", err)
    }

    if result.JobID != job.ID {
        t.Fatalf("job id mismatch")
    }
    if len(result.ShardMetrics) != 3 {
        t.Fatalf("expected 3 shard metrics, got %d", len(result.ShardMetrics))
    }
    if len(result.FinalOutput) == 0 {
        t.Fatal("expected non-empty final output")
    }
}

func TestPipelineLayerGapDetection(t *testing.T) {
    ctx := context.Background()
    workers := mock.NewMockWorkers()
    executor := NewPipelineExecutor(workers)

    job := DistributedJob{
        ID:        "job-gap",
        ModelID:   "qwen2.5-3b",
        RequestID: "request-gap",
        InitialPayload: []byte("test"),
        Shards: []ShardAssignment{
            {
                ShardID:  "shard-0-9",
                Layer:    model.LayerRange{Start: 0, End: 9},
                WorkerID: "worker-a",
            },
            {
                ShardID:  "shard-15-25",
                Layer:    model.LayerRange{Start: 15, End: 25},
                WorkerID: "worker-b",
            },
        },
    }

    _, err := executor.Execute(ctx, job)
    if err == nil {
        t.Fatal("expected error for layer gap")
    }
}

func TestPipelineWorkerNotFound(t *testing.T) {
    ctx := context.Background()
    workers := mock.NewMockWorkers()
    executor := NewPipelineExecutor(workers)

    job := DistributedJob{
        ID:        "job-missing",
        ModelID:   "qwen2.5-3b",
        RequestID: "request-missing",
        InitialPayload: []byte("test"),
        Shards: []ShardAssignment{
            {
                ShardID:  "shard-0-9",
                Layer:    model.LayerRange{Start: 0, End: 9},
                WorkerID: "worker-nonexistent",
            },
        },
    }

    _, err := executor.Execute(ctx, job)
    if err == nil {
        t.Fatal("expected error for missing worker")
    }
}
