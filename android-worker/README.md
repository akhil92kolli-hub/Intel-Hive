# Intel-Hive Android Worker

## Overview

The Android worker is a Kotlin application with a scheduler WebSocket runtime,
an IntelHive-owned shard inference contract, and a JNI adapter for IntelHive's
Qwen2 layer-range executor. The shard executor uses the same pinned llama.cpp
source revision as the host feasibility tests. The device benchmark runs the
production executor through all three logical shards with real prefill, decode,
and KV positions.

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
├── NativeShardBenchmarkEngine
│   └── production executor benchmark across all three local logical shards
│
├── RequiredModelManager
│   ├── manifest loading
│   ├── download and progress tracking
│   ├── size and SHA-256 verification
│   └── app-specific model storage
│
├── TransportLayer
│   ├── activation serialization
│   ├── peer communication
│   └── checksum validation
│
└── Benchmark
    ├── latency measurement
    ├── throughput calculation
    ├── local result export (JSON)
    └── append-only Supabase upload
```

## Phase-0 Scope

- **Single device inference:** measure tokens/second for Qwen2.5-3B-Instruct Q4_K_M
- **Model setup:** download the configured Qwen GGUF and verify its size and digest
- **Activation contract:** describe layer-boundary tensors with metadata and a
  verified payload checksum
- **Benchmark harness:** JSON output with timing and throughput
- **Worker registration:** connect to the LAN scheduler using its WebSocket protocol
- **Worker heartbeat:** report liveness, battery, and charging state while connected
- **Assignment protocol:** accept correlated prefill/decode assignments, validate
  inputs, report completion/failure, and release per-sequence state

The runtime reports the device as `UNBENCHMARKED` until inference benchmarking
has completed. On first launch the app shows the required Qwen2.5-3B model and
offers an Android-managed download (Wi-Fi-only by default). The worker cannot
connect until the model file is downloaded, its declared byte size and SHA-256
are verified, and the packaged JNI engine successfully loads it. Run the Go
gateway from `server/` and enter its WebSocket URL in the app. The debug build
permits `ws://` for LAN testing; use only a trusted local network.

The typed Android contract carries model/version and inclusive layer-range
metadata, prefill/decode sequence identity, activation tensor metadata, and
payload checksums. WebSocket assignments now carry the complete activation
envelope, including request identity, worker route, pass position, dtype,
shape, and checksum. The Android worker validates this envelope before handing it to the native
executor. The current Android backend supports only the configured Qwen2.5-3B
Q4_K_M model, F32 row-major activations, and multithreaded CPU execution. The
transport adapter accepts prompt text only for the first prefill shard,
tokenizes it with the loaded GGUF vocabulary, and constructs an explicit
token-count/KV-offset request before entering the native backend.
Benchmark output uses `execution_mode=single_device_all_shards`; this does not
establish activation transport between independent workers. It does not yet
establish networked multi-phone M1, GPU acceleration, generic
GGUF/architecture support, or KV-state migration. The instrumented three-shard
test runs the ranges sequentially on one Android device and tests prefill plus
one decode step.

## Runtime and Build

The APK contains `libintelhive_jni.so`, which links IntelHive's JNI bridge,
layer-range executor, and statically linked llama.cpp/ggml CPU libraries.
The multi-gigabyte GGUF remains a separate download. Install JDK 17 and set
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
gradle -p android-worker \
  -Pintelhive.llamaSourceDir="$PWD/.devtools/llama-source" \
  :app:assembleDebug
```

The app includes a bundled testing catalog for the pinned GGUF. To make a build
read its model manifest from an HTTPS configuration endpoint instead, pass
`-Pintelhive.modelManifestUrl=https://host/path/qwen2.5-3b-instruct.json`.
The endpoint must return the same manifest shape as
`models/manifests/qwen2.5-3b-instruct.json`; its model ID, architecture,
dimensions, digest, and size are validated against the supported native
executor before the download URL is used.

The default build reads that manifest from the public Supabase
`model-artifacts` bucket. Supabase currently hosts the small manifest while its
GGUF URL points to the verified upstream artifact. The GGUF is 2,104,932,768
bytes and cannot be stored in the project until its current 50 MB Storage limit
is raised. After that limit is raised and the object is uploaded and verified,
use this public object path in the manifest:

```text
model-artifacts/qwen2.5-3b-instruct/1.0.0/qwen2.5-3b-instruct-q4_k_m.gguf
```

The JNI load-probe test checks that the packaged library can load and initialize
llama.cpp. To run it on a connected arm64 device:

```bash
gradle -p android-worker \
  -Pintelhive.llamaSourceDir="$PWD/.devtools/llama-source" \
  :app:connectedDebugAndroidTest
```

Install the debug APK with:

```bash
adb install -r android-worker/app/build/outputs/apk/debug/app-debug.apk
```

The instrumentation suite also contains a real-weight test. It is skipped when
the expected GGUF is not installed on the device.

## Model Setup

The executor hashes the complete model file before loading it. The expected
SHA-256 is
`626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`; the GGUF
must report 36 layers and hidden size 2048. Use only a model artifact with this
digest. The large model file is intentionally not checked into the repository.
The verified artifact lives in app-specific external storage under
`files/Documents/models/`; the Android DownloadManager writes to a temporary
`.part` file, and the app renames it into place only after both size and digest
checks pass. A failed or mismatched artifact is never made available to the
worker executor.

## Benchmark Output

The app exports JSON to the device's app cache directory and then submits a
normalized row to Supabase's `public.benchmark_results` table. The local file
remains available if the network upload fails. The APK uses only the public
publishable key; personal access tokens and service-role keys must never be
placed in Gradle properties or application resources.

Row Level Security permits Android and iOS clients to insert results for the
pinned model artifact. Client-side reads, updates, and deletes are denied. Build
configuration can override the endpoint and publishable key when needed:

```bash
gradle -p android-worker \
  -Pintelhive.supabaseUrl=https://project-ref.supabase.co \
  -Pintelhive.supabasePublishableKey=sb_publishable_example \
  :app:assembleDebug
```

An exported and uploaded result includes:

```json
{
  "status": "COMPLETED",
  "execution_mode": "single_device_all_shards",
  "model_id": "qwen2.5-3b-instruct",
  "prefill_tokens": 32,
  "generated_tokens": 8,
  "tokens_per_second": 0.07,
  "layer_ranges": [
    {"shardId": "s0", "layerStart": 0, "layerEnd": 9},
    {"shardId": "s1", "layerStart": 10, "layerEnd": 19},
    {"shardId": "s2", "layerStart": 20, "layerEnd": 35}
  ]
}
```

The CPU/Vulkan boundary and the work required for a real Vulkan backend are
recorded in `docs/android-native-backends.md`.
