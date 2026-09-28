# Intel-Hive

Distributed AI inference across Android and iOS devices.

## Phase 0

Phase 0 proves the smallest useful path:

```text
HTTP request → Go scheduler → Android/iOS workers → generated response
```

The scheduler operates on `ModelShard` objects rather than device identities. The initial target is a three-shard, LAN-only pipeline for a quantized GGUF model.

## Repository layout

- `website/` — React/Vite IntelHive landing page for Cloudflare Pages or Vercel
- `android-worker/` — Kotlin Android worker and llama.cpp integration
- `ios-worker/` — Swift iOS worker, scheduler transport, capabilities, and inference backend interface
- `server/` — Go gateway, scheduler, registry, and job runtime
- `proto/` — Protocol Buffer contracts shared by workers and server
- `models/manifests/` — model and shard metadata
- `benchmark/` — repeatable transport and inference measurements
- `deployment/` — local Docker Compose services
- `docs/` — architecture and protocol notes

## Website deployment

The static landing page lives in `website/`. In Cloudflare Pages or Vercel, set the project root directory to `website`, build command to `npm run build`, and output directory to `dist`. See [`website/README.md`](website/README.md) for local development and provider-specific setup.

## iOS worker

The iOS worker mirrors the Android registration and benchmark flow using Swift concurrency, URLSession WebSockets, CryptoKit, and Metal capability discovery. It reports `platform: "ios"` and `inference_backend: "metal"` when a Metal device is available. The llama.cpp/ggml Metal bridge is injected through the `InferenceBackend` protocol rather than vendored in this repository.

```bash
cd ios-worker
swift test
```

## First milestone

T001 establishes the repository and protocol contracts. The next implementation step is the single-device benchmark on Android and iOS, followed by activation transport between workers.
