# T008: Mock Worker

## Objective

Create a mock worker runtime so the scheduler can be exercised end-to-end without Android hardware.

The mock worker should present the same interface as a real worker but simulate:
- registration
- heartbeat
- capability advertisement
- shard execution
- activation generation
- activation receipt

## Why this matters

This is the key step that lets distributed inference be developed and tested before a physical phone arrives.

## Generic Worker Runtime Interface

```go
type WorkerRuntime interface {
    Start(ctx context.Context) error
    Stop(ctx context.Context) error

    Register(ctx context.Context) error
    Heartbeat(ctx context.Context) error
    Capabilities(ctx context.Context) WorkerCapabilities
    Execute(ctx context.Context, job Job) (JobResult, error)
}
```

## Mock Worker Implementation

```go
type MockWorker struct {
    ID         string
    Platform   Platform
    State      WorkerState
    Transport  transport.Transport
    Capabilities WorkerCapabilities
    Layers     []LayerRange
}
```

Example:

```text
MockWorker-A -> layers 0–9
MockWorker-B -> layers 10–19
MockWorker-C -> layers 20–29
```

## Registration Behavior

A mock worker should:
- register with the scheduler
- advertise supported model types and layer ranges
- report `UNBENCHMARKED` if no benchmark was run
- optionally allow a simulated benchmark score

## Execution Behavior

Simulated execution:

1. receive job with shard and model id
2. verify that shard range is within the mock worker's assigned layers
3. generate activation payload
4. send activation to the next worker or scheduler using transport
5. return an execution result with simulated metrics

## Scheduler Interaction

Mock workers do not need to know if they are Android or iOS. They expose the same runtime contract and can be scheduled identically.

## Example Simulation

```text
Scheduler assigns:
  worker-a -> layers 0-9
  worker-b -> layers 10-19
  worker-c -> layers 20-29

worker-a executes layer 0-9 and emits activation
worker-b executes layer 10-19 and emits activation
worker-c executes layer 20-29 and emits final output
```

## In-Memory Worker Graph

This can be modeled as:

```go
type MockWorkerGraph struct {
    Workers map[string]*MockWorker
}
```

This enables a local scheduler test environment without networking.

## Deliverables

- `MockWorker` implementation
- registration and heartbeat simulation
- capability advertisement
- layer-range assignment simulation
- activation generation and forwarding
- local transport-based execution scenario

## Next Step

Once mock workers are ready, the scheduler can be tested on a mock network, and only later replaced by Android workers.
