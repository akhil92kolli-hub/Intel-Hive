# Distributed inference protocol + pluggable mobile inference engines

The current Phase-0 implementation provides:

- a JSON WebSocket protocol for worker registration and heartbeats
- a Go gateway with an in-memory worker registry
- a scheduler bridge that dispatches shard assignments to connected WebSocket workers
- an Android foreground service that registers a device and reports liveness/power state
- a Go scheduler and pipeline executor with mock and remote-worker runtimes

The server-side remote execution protocol is wired. The Android worker now
executes real Qwen2.5-3B GGUF layer ranges through JNI, including prefill,
decode, and worker-local KV positions. iOS assignment execution remains
incomplete. Android benchmark results use the production shard executor and
persist an explicit execution mode.

## Current worker flow

```text
Android WorkerService -> WebSocket /worker -> Go worker registry
```

Workers remain `UNBENCHMARKED` until benchmark results and model/shard
capabilities are provided. Ready workers with registered model shards are
available to the Go scheduler. `POST /inference` builds a deterministic
pipeline from registered shard ranges, runs a prefill pass followed by
token-by-token decode passes, and returns generated text, token IDs,
`execution_mode`, and the participating `worker_ids`.
Assignments include independent pass-ordinal and KV-token-offset metadata for
worker-local KV state. A pass ordinal is never used as an inference position.
For prompt prefill, the first Android worker tokenizes UTF-8 text with the
verified GGUF vocabulary and reports the resolved token count. The scheduler
uses that count as the decode KV offset and propagates it to later shards.
Phase 0 recognizes the pinned 36-layer Qwen2.5-3B model artifact. The Android
runtime executes these assignments; independent-device activation transport
still requires the two-phone and three-phone validation runs.

The activation value contract is defined in Go as `activation.Envelope` and
mirrored at the Android transport boundary. It carries the request/sequence,
model ID/version/artifact digest, route and output layer, `pass_ordinal`,
`kv_token_offset`, and `token_count`. M1 tensors are only F16/F32,
row-major-contiguous, little-endian values with an exact byte length and a
`sha256:<hex>` payload checksum. Scheduler job assignments and completions
carry this envelope in their `activation` field; the legacy raw byte-only
field has been removed from the WebSocket contract.

The protocol version is `phase-0-v3`. A v3 assignment carries the pinned model
artifact and adjacent worker IDs, allowing a receiver to reject stale,
misrouted, or mismatched-model tensors before they reach the native backend.

## Remaining M1 integration

1. Run the scheduler with two independently connected Android workers and prove
   that canonical activations cross the network between shard owners.
2. Repeat with three workers assigned `0-9`, `10-19`, and `20-35`.
3. Record distributed results with
   `execution_mode=distributed_pipeline`; the current Android device benchmark
   is `single_device_all_shards` and does not prove network transport.
4. Implement assignment execution in the iOS worker runtime separately.

The Android backend is currently multithreaded CPU. Vulkan requires backend
buffer allocation and scheduling in the IntelHive layer-range executor; merely
enabling llama.cpp's Vulkan build option does not move its custom graphs or
mmap-backed tensors to the GPU. See `docs/android-native-backends.md`.

## Relevant implementation files

- `server/cmd/scheduler/main.go`
- `server/internal/protocol/worker.go`
- `server/internal/transport/worker_socket.go`
- `server/internal/registry/worker.go`
- `internal/scheduler/scheduler.go`
- `internal/pipeline/pipeline.go`
- `internal/activation/contract.go`
- `android-worker/app/src/main/kotlin/com/intellihive/worker/service/WorkerService.kt`
- `android-worker/app/src/main/kotlin/com/intellihive/worker/inference/ShardInference.kt`
- `android-worker/app/src/main/kotlin/com/intellihive/worker/inference/NativeShardBenchmarkEngine.kt`

## Model target

For M1 experiments, the planned target remains Qwen2.5-3B-Instruct in GGUF
Q4_K_M format, with the backend selected after device capability testing.
