# Intel-Hive Android Worker

## Overview

The Android worker is a Kotlin application that integrates `llama.cpp` for inference and maintains a WebSocket connection to the scheduler.

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
├── InferenceEngine
│   ├── model loading
│   ├── layer execution
│   ├── KV cache management
│   └── metrics collection
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
- **Activation export:** serialize a single layer activation as Protobuf
- **Benchmark harness:** JSON output with timing and throughput
- **No networking** (until T003)

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
