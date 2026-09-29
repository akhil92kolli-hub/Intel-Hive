# Intel-Hive Android Worker

## Overview

The Android worker is a Kotlin application with a scheduler WebSocket runtime,
an IntelHive-owned shard inference contract, and a JNI adapter for IntelHive's
Qwen2 layer-range executor. The shard executor uses the same pinned llama.cpp
source revision as the host feasibility tests. The full-model llama.cpp
benchmark adapter remains separate and is not used for shard execution.

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
├── NativeLayerRangeShardExecutor
│   └── transport-neutral contract → JNI → IntelHive Qwen2 layer-range engine
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
has completed. With the pinned model installed, the JNI runtime can execute
Qwen2.5-3B layer shards and marshal activation/token results to the worker
contract. Run the Go gateway from `server/` and enter its WebSocket URL in the
app. The debug build permits `ws://` for LAN testing; use only a trusted local
network.

The typed Android contract carries model/version and inclusive layer-range
metadata, prefill/decode sequence identity, activation tensor metadata, and
payload checksums. WebSocket assignments now carry the complete activation
envelope, including request identity, worker route, pass position, dtype,
shape, and checksum. The Android worker validates this envelope before handing
it to the native executor. The current Android backend supports only the
pinned Qwen2.5-3B Q4_K_M model, F32 row-major activations, and CPU execution.
It does not yet establish networked multi-phone M1, GPU acceleration, generic
GGUF/architecture support, or KV-state migration. The instrumented three-shard
test runs the ranges sequentially on one Android device and tests prefill plus
one decode step.

## Build

The checked-in Gradle wrapper pins Gradle `8.7`. Install JDK 17 and set
`ANDROID_SDK_ROOT` to an Android SDK containing Platform 34, Build Tools
34.0.0, Platform-Tools, Android NDK `26.3.11579264`, and CMake `3.22.1`:

```bash
sdkmanager \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "ndk;26.3.11579264" \
  "cmake;3.22.1"
```

The app builds only `arm64-v8a`. Provide the llama.cpp checkout or extracted
source at the revision recorded in `native/layer-range/upstream.lock`. By
default Gradle expects it at `.devtools/llama-source` relative to the
repository root; override that path with
`-Pintelhive.llamaSourceDir=/absolute/path/to/llama.cpp`.

```bash
cd android-worker
./gradlew \
  -Pintelhive.llamaSourceDir="$(cd .. && pwd)/.devtools/llama-source" \
  :app:assembleDebug
```

The JNI load-probe test checks that the packaged library can load and initialize
llama.cpp. To run it on a connected arm64 device:

```bash
./gradlew \
  -Pintelhive.llamaSourceDir="$(cd .. && pwd)/.devtools/llama-source" \
  :app:connectedDebugAndroidTest
```

The instrumentation suite also contains a real-weight test. It is skipped when
the expected GGUF is not installed on the device.

## Model Setup

For the real-weight instrumentation test, push the verified Qwen2.5-3B-Instruct
Q4_K_M GGUF into the app-specific external files directory:

```
adb shell mkdir -p /sdcard/Android/data/com.intelhive.worker/files/models
adb push qwen2.5-3b-instruct-q4_k_m.gguf \
  /sdcard/Android/data/com.intelhive.worker/files/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

The instrumented test uses that external-file copy directly, avoiding a second
multi-gigabyte copy. The worker service itself loads the model from its private
cache at `cache/models/qwen2.5-3b-instruct-q4_k_m.gguf`.

The executor hashes the complete model file before loading it. The expected
SHA-256 is
`626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`; the GGUF
must report 36 layers and hidden size 2048. Use only a model artifact with this
digest. The large model file is intentionally not checked into the repository.

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
