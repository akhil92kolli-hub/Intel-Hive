# Intel-Hive

Distributed AI inference across Android and iOS devices.

## Roadmap

The project is driven by four acceptance gates: M1 proves three-phone LAN
inference, M2 proves transparent worker replacement, M3 proves automatic
heterogeneous pipelines across 10–20 devices, and M4 introduces remote Internet
workers with measurable economics. Phase 0 covers M1/M2; Phase 1 covers M3;
the production/network phase begins with M4.

See [docs/roadmap.md](docs/roadmap.md) for task sequencing, acceptance tests,
and mobile release targets. The scheduler operates on `ModelShard` objects
rather than device identities.

## Repository layout

- `android-worker/` — Kotlin Android worker and llama.cpp integration
- `ios-worker/` — Swift iOS worker, scheduler transport, capabilities, and inference backend interface
- `server/` — Go gateway, scheduler, registry, and job runtime
- `proto/` — Protocol Buffer contracts shared by workers and server
- `models/manifests/` — model and shard metadata
- `benchmark/` — repeatable transport and inference measurements
- `deployment/` — local Docker Compose services
- `docs/` — architecture and protocol notes

## iOS worker

The iOS worker mirrors the Android registration and benchmark flow using Swift concurrency, URLSession WebSockets, CryptoKit, and Metal capability discovery. It reports `platform: "ios"` and `inference_backend: "metal"` when a Metal device is available. The llama.cpp/ggml Metal bridge is injected through the `InferenceBackend` protocol rather than vendored in this repository.

```bash
cd ios-worker
swift test
```

## Immediate engineering gate

The next target is **M1**: three ordinary Android phones jointly execute a
quantized 3B model through assigned layer shards over a local network and return
a generated response. The immediate implementation sequence starts with T010
(Scheduler Core) and T011 (Dynamic Worker Assignment), as detailed in the
[roadmap](docs/roadmap.md).
