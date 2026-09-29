# T010/T011: Scheduler API and Dynamic Worker Assignment

The scheduler owns worker selection and execution ordering. It stores
platform-independent capabilities and attaches workers through the
`internal/worker.Runtime` interface; mock workers are one implementation, not
the scheduler's required runtime type.

## API

```go
registry := scheduler.NewRegistry()
s := scheduler.New(registry)

s.RegisterWorker(record, runtime) // runtime implements worker.Runtime
plan, err := s.BuildPlan(jobID, modelID, modelShards)
result, err := s.Execute(ctx, plan, payload)
```

## Assignment policy

For every logical model shard, the scheduler filters workers by:

1. `READY` state
2. layer-range coverage
3. one worker per shard in a plan

It then scores eligible workers using the highest advertised benchmark and applies a load penalty. Ties are deterministic by worker ID.

`UNBENCHMARKED`, `BUSY`, and `OFFLINE` workers are retained in the registry but
are not selected for standard inference jobs.

## Execution ordering

Plans are ordered by the layer range supplied by the model. The pipeline executor runs them sequentially because activation `n` is the input to shard `n+1`. This is pipeline execution, not independent data parallelism.

## Failure and health

`MarkOffline` removes a worker from future plans while retaining its record.
`ExpireWorkers` transitions workers whose `LastSeen` exceeds the configured
timeout to `OFFLINE`. During execution, a shard failure identifies its assigned
worker only when the runtime classifies the failure as `worker.ErrUnavailable`;
the scheduler then marks that worker offline, rebuilds the complete plan, and
retries the original payload while retaining the same job ID. Other execution
errors are returned without misclassifying a healthy worker as offline. Retries
are bounded by the number of registered workers. If no eligible replacement
exists, execution returns an error instead of a success-shaped result.

The retry restarts the pipeline from its initial payload. Workers should treat
shard execution as request-scoped inference without externally visible side
effects.

## Validation

The scheduler tests cover:

- deterministic three-shard assignment
- rejection of unbenchmarked workers
- end-to-end execution through mock workers
- worker failure, full-plan reconstruction, and replacement execution
- explicit failure when no replacement worker is eligible

The LAN gateway can currently register Android workers and record heartbeats,
but it is not yet connected to the root scheduler's registry and does not
dispatch jobs to them. The failure retry here is an in-process M2 foundation;
remote heartbeat-loss handling and network-backed worker execution remain.
The next integration step is a remote `worker.Runtime` adapter backed by the
worker WebSocket session, followed by shard execution on Android.
