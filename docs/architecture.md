# Architecture

## Phase-0 topology

```text
Test client --HTTP--> Gateway/Scheduler --WebSocket/Protobuf--> Android workers
                                                   ├─ shard 0: layers 0-9
                                                   ├─ shard 1: layers 10-19
                                                   └─ shard 2: layers 20-29
```

## Core abstraction

A worker advertises one or more `ModelShard` capabilities. The scheduler selects a healthy worker for each required shard and constructs a pipeline:

```text
mobile-3b-s0 → worker-a
mobile-3b-s1 → worker-b
mobile-3b-s2 → worker-c
```

Worker identity is therefore replaceable. If a worker fails before execution completes, Phase 0 may restart the request with another worker holding the same shard. KV-cache migration is explicitly out of scope.

## Inference session

Autoregressive generation runs the pipeline once per token. Each worker owns request-scoped state:

- job and model/shard identifiers
- sequence position
- shard-local KV cache
- previous and next pipeline peers

The KV cache must never be global to the process.

## Health policy

Workers send heartbeats every five seconds. The registry marks a worker suspect after one missed heartbeat, unavailable after 15 seconds, and removes it from the active pool after 30 seconds. These values are configuration, not protocol guarantees.
