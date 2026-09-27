# Distributed inference protocol + pluggable mobile inference engines

This Phase-0 implementation provides:

- a protocol definition for worker registration, heartbeat, jobs, and activation packets
- a Go scheduler/runtime with worker registry, model registry, and pipeline assignment logic
- a pluggable mobile inference engine abstraction with a llama.cpp engine entry point
- a native Android bridge for llama.cpp (Vulkan-ready)

## Runtime flow

```text
Request -> Go Scheduler -> Worker registry -> candidate shard selection -> job assignment -> activation transport
```

## Native llama.cpp integration boundary

The Android project includes a C++ JNI bridge that is designed to connect to llama.cpp's native C API. The implementation is intentionally explicit about the boundary so the project can be built against a real llama.cpp Android artifact or a local CMake integration.

## Files to review first

- `server/cmd/scheduler/main.go`
- `server/internal/protocol/messages.go`
- `server/internal/engines/engine.go`
- `android-worker/app/src/main/cpp/llama_bridge.cpp`
- `android-worker/app/src/main/java/com/intellihive/worker/inference/NativeInferenceEngine.java`

## Model recommendation

Primary model for MVP:

- Qwen2.5-3B-Instruct
- GGUF format
- Q4_K_M quantization
- llama.cpp runtime
- Vulkan backend
- 2K–4K context for Phase-0 experiments

## Next implementation task

After protocol + engine boundaries are in place, the next step is T003:

- real worker registration and heartbeat protocol over WebSocket
- job assignments and responses
- Android benchmark collection from a real device
- first cross-device activation payload experiment (no model migration yet)
