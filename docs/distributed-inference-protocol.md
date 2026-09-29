# Distributed inference protocol + pluggable mobile inference engines

The current Phase-0 implementation provides:

- a JSON WebSocket protocol for worker registration and heartbeats
- a Go gateway with an in-memory worker registry
- a scheduler bridge that dispatches shard assignments to connected WebSocket workers
- an Android foreground service that registers a device and reports liveness/power state
- a Go scheduler and pipeline executor with mock and remote-worker runtimes

The server-side remote execution protocol is wired. The Android worker consumes
assignments but has no native layer-range executor; iOS assignment execution
also remains incomplete. The Android full-model benchmark class has JNI
declarations, but a working native llama.cpp bridge and real shard execution
are not present in this repository.

## Current worker flow

```text
Android WorkerService -> WebSocket /worker -> Go worker registry
```

Workers remain `UNBENCHMARKED` until benchmark results and model/shard
capabilities are provided. Ready workers with registered model shards are
available to the Go scheduler. `POST /inference` builds a deterministic
pipeline from registered shard ranges, runs a prefill pass followed by
token-by-token decode passes, and returns generated text and token IDs.
Assignments include sequence and position metadata for worker-local KV state.
Phase 0 currently recognizes the Qwen2.5-3B model identifiers and their known
layer counts. The loop is wired through the server protocol and scheduler, but
mobile runtimes do not yet execute real model inference assignments.

The activation value contract is defined in Go as `activation.Envelope` and
mirrored by the Android `Activation` data class. Both use string sequence IDs,
request and model/version identity, source/destination workers, pass position,
inclusive output-layer identity, tensor dtype/shape, Base64-encoded payload
bytes in JSON, and a `sha256:<hex>` checksum. Scheduler job assignments and
completions carry this envelope in their `activation` field; the legacy raw
byte-only field has been removed from the v2 WebSocket contract. The server
checks envelope identity, route, layer boundary, payload shape, and checksum
before forwarding or accepting a non-final activation.

The protocol version is `phase-0-v2`. A v2 assignment also carries the model
revision and adjacent worker IDs, allowing a receiver to reject stale or
misrouted tensors before they reach the native backend.

## Remaining M1 integration

1. Implement assignment execution in the iOS worker runtime.
2. Integrate the T002C feasibility backend with Android's built native
   runtime and GGUF/Qwen model loader.
3. Wire the typed Android `InferenceEngine`, `ModelSpec`, `ShardSpec`, and
   activation/state contract to a built native layer-range backend.
4. Verify the native execution path on two and then three physical Android phones.

The Android service now parses and validates `job_assignment`, acknowledges
valid assignments, reports correlated completion/failure messages, and handles
`sequence_end`. Its current executor explicitly rejects work because no native
layer-range backend is present; this protocol integration alone does not make
the Android device inference-ready.

The Android inference package now defines an IntelHive-owned `InferenceEngine`
contract and typed model/shard/activation structures. Activations identify the
job, request, sequence, model version, source/destination workers, pass
position, output layer, tensor dtype/shape, and SHA-256 checksum. The
`LlamaCppBenchmarkEngine` remains a single-device full-model benchmark and
does not implement `InferenceEngine`. The native T002C prototype is kept under
`native/layer-range`; it uses ggml to prove that a residual stream can be
executed across independently stateful inclusive layer ranges, including
prefill, decode positions, KV reset, and malformed input handling. It uses
deterministic test weights and is not an Android Qwen executor.

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
- `android-worker/app/src/main/kotlin/com/intellihive/worker/inference/InferenceEngine.kt`

## Model target

For M1 experiments, the planned target remains Qwen2.5-3B-Instruct in GGUF
Q4_K_M format, with the backend selected after device capability testing.
