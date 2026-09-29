# Intel-Hive Android Worker

## Overview

The Android worker is a Kotlin application with a scheduler WebSocket runtime,
an IntelHive-owned shard inference contract, and a separate full-model
llama.cpp benchmark adapter. The native layer-range backend is not implemented
yet.

## Architecture

```
AndroidWorker
├── MainActivity
│   ├── lifecycle management
│   └── UI state
│
├── WorkerService
│   ├── scheduler connection (WebSocket)
│   ├── heartbeat loop
│   └── job lifecycle
│
├── InferenceEngine contract (`inference/ShardInference.kt`)
│   ├── model loading
│   ├── layer execution
│   ├── KV cache management
│   └── metrics collection
├── LlamaCppBenchmarkEngine
│   └── full-model local benchmark only; not a shard executor
│
├── ModelManager
│   ├── shard discovery
│   ├── hash verification
│   └── local storage
│
├── TransportLayer
│   ├── activation serialization
│   ├── peer communication
│   └── checksum validation
│
└── Benchmark
    ├── latency measurement
    ├── throughput calculation
    └── result export (JSON)
```

## Phase-0 Scope

- **Single device inference:** measure tokens/second for Qwen2.5-3B-Instruct Q4_K_M
- **Local model loading:** GGUF from application assets or filesystem
- **Activation contract:** describe layer-boundary tensors with metadata and a
  verified payload checksum
- **Benchmark harness:** JSON output with timing and throughput
- **Worker registration:** connect to the LAN scheduler using its WebSocket protocol
- **Worker heartbeat:** report liveness, battery, and charging state while connected
- **Assignment protocol:** accept correlated prefill/decode assignments, validate
  inputs, report completion/failure, and release per-sequence state

The runtime reports the device as `UNBENCHMARKED` until inference benchmarking
has completed. It currently does not execute remotely assigned model shards.
Run the Go gateway from `server/` and enter its WebSocket URL in the app. The
debug build permits `ws://` for LAN testing; use only a trusted local network.
The worker service now consumes the scheduler's assignment/result protocol,
but the native layer-shard executor is not implemented yet. Until that backend
is available, assignments are acknowledged and then explicitly failed rather
than returning mock inference results.

The typed Android contract carries model/version and inclusive layer-range
metadata, prefill/decode sequence identity, activation tensor metadata, and
payload checksums. WebSocket assignments now carry the complete activation
envelope, including request identity, worker route, pass position, dtype,
shape, and checksum. The Android worker validates this envelope before handing
it to an executor. The executor remains unavailable until the native backend
is integrated, so this protocol work does not meet M1.

## Build

```bash
cd android-worker
./gradlew assembleDebug
```

## Model Setup

Place the quantized GGUF model in `android-worker/app/src/main/assets/models/`:

```
qwen2.5-3b-instruct-q4_k_m.gguf (~2.2 GB)
```

## Benchmark Output

The app exports JSON to the device's app cache directory:

```json
{
  "timestamp": "2026-09-27T10:30:00Z",
  "model": "qwen2.5-3b-instruct",
  "quantization": "Q4_K_M",
  "device": {
    "model": "Pixel 6",
    "android_version": "14",
    "ram_mb": 8192,
    "gpu": "Adreno 650"
  },
  "benchmark": {
    "prefill_tokens": 128,
    "generated_tokens": 100,
    "total_time_ms": 12345,
    "tokens_per_second": 8.1,
    "prefill_speed_tokens_per_second": 100.5,
    "generation_speed_tokens_per_second": 8.1
  },
  "memory": {
    "peak_rss_mb": 3500,
    "peak_gpu_mb": 1200
  },
  "thermal": {
    "initial_temp_c": 32.5,
    "peak_temp_c": 44.2,
    "final_temp_c": 38.1
  },
  "activations": {
    "layer_output_size_bytes": 1572864,
    "dtype": "float32",
    "shape": [1, 128, 3072]
  }
}
```
