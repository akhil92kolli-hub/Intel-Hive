# T009: Mock Distributed Inference Pipeline

## Objective

Build the first end-to-end distributed inference flow using mock workers.

The pipeline should prove:

```text
Scheduler
    ↓ (assign shards)
  Worker A (layers 0-9) → Worker B (layers 10-19) → Worker C (layers 20-29)
    ↓
Activation Transport
    ↓
Final Output
```

## Pipeline Components

### 1. Job Distribution

The scheduler creates a distributed job that assigns each shard to a capable worker.

```go
type DistributedJob struct {
    ID          string
    ModelID     string
    RequestID   string
    Shards      []ShardAssignment
    InitialPayload []byte
}

type ShardAssignment struct {
    ShardID   string
    Layer     model.LayerRange
    WorkerID  string
}
```

### 2. Pipeline Executor

Execute shards in sequence, with each worker's output becoming the next worker's input.

```go
type PipelineExecutor struct {
    Workers   map[string]worker.Runtime
    Transport *transport.InMemoryTransport
}

func (e *PipelineExecutor) Execute(ctx context.Context, job DistributedJob) ([]byte, error)
```

### 3. Shard Sequencing

Order shards by layer range and execute them in sequence:
- Worker A processes layers 0-9
- Worker B receives activation from A, processes layers 10-19
- Worker C receives activation from B, processes layers 20-29
- Final activation is the model output

### 4. Activation Routing

Each worker's output activation is:
1. Serialized
2. Validated (checksum)
3. Routed to the next worker via transport
4. Deserialized by the receiving worker

## Example Execution

```text
Initial payload: "What is AI?"

Worker-A (layers 0-9):
  - receives: "What is AI?"
  - emits: activation_0_9 (to Worker-B)

Worker-B (layers 10-19):
  - receives: activation_0_9 (from Worker-A)
  - emits: activation_10_19 (to Worker-C)

Worker-C (layers 20-29):
  - receives: activation_10_19 (from Worker-B)
  - emits: activation_20_29 (final output)

Scheduler collects: activation_20_29 as result
```

## Error Handling

The pipeline should handle:
- worker offline (retry or fail-fast)
- layer mismatch (reject invalid assignments)
- checksum validation failure (reject corrupted activation)
- timeout (basic retry logic)

## Mock Metrics

Capture:
- shard execution time
- activation serialization/deserialization time
- transport latency
- total end-to-end time
- tokens per second (mock value)

## Tests

1. Basic 3-worker pipeline
2. Single-worker fallback (full model on one worker)
3. Worker offline during pipeline
4. Checksum validation failure
5. Layer mismatch detection

## Output

The pipeline returns:
```go
type PipelineResult struct {
    FinalOutput     []byte
    ExecutionTimeMs int64
    TokensPerSec    float64
    ShardMetrics    map[string]ShardMetric
}
```

## Why this is critical

This is the first proof that distributed execution works without hardware.

Once this passes, the scheduler and worker assignment logic are ready to be built on top.
