# Intel-Hive

Distributed AI inference across Android devices.

## Phase 0

Phase 0 proves the smallest useful path:

```text
HTTP request → Go scheduler → Android workers → generated response
```

The scheduler operates on `ModelShard` objects rather than device identities. The initial target is a three-shard, LAN-only pipeline for a quantized GGUF model.

## Repository layout

- `android-worker/` — Kotlin Android worker and llama.cpp integration
- `server/` — Go gateway, scheduler, registry, and job runtime
- `proto/` — Protocol Buffer contracts
- `models/manifests/` — model and shard metadata
- `benchmark/` — repeatable transport and inference measurements
- `deployment/` — local Docker Compose services
- `docs/` — architecture and protocol notes

## First milestone

T001 establishes the repository and protocol contracts. The next implementation step is the single-device Android benchmark, followed by activation transport between two workers.
