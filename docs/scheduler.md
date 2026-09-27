# T010/T011: Scheduler API and Dynamic Worker Assignment

The scheduler now owns worker selection and execution ordering. Workers only advertise capabilities and execute the shard they receive.

## API

```go
registry := scheduler.NewRegistry()
s := scheduler.New(registry)

s.RegisterWorker(record, runtime)
plan, err := s.BuildPlan(jobID, modelID, modelShards)
result, err := s.Execute(ctx, plan, payload)
```

## Assignment policy

For every logical model shard, the scheduler filters workers by:

1. `READY` state
2. layer-range coverage
3. one worker per shard in a plan

It then scores eligible workers using the highest advertised benchmark and applies a load penalty. Ties are deterministic by worker ID.

`UNBENCHMARKED`, `BUSY`, and `OFFLINE` workers are retained in the registry but are not selected for standard inference jobs.

## Execution ordering

Plans are ordered by the layer range supplied by the model. The pipeline executor runs them sequentially because activation `n` is the input to shard `n+1`. This is pipeline execution, not independent data parallelism.

## Failure and health

`MarkOffline` removes a worker from future plans while retaining its record. `ExpireWorkers` transitions workers whose `LastSeen` exceeds the configured timeout to `OFFLINE`.

## Validation

The scheduler tests cover:

- deterministic three-shard assignment
- rejection of unbenchmarked workers
- end-to-end execution through mock workers

The next integration step is to connect registration/heartbeat messages to this registry and add reassignment after a worker failure.
