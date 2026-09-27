# Intel-Hive iOS Worker

The iOS worker mirrors the Android Phase-0 worker while using native iOS APIs:

- Swift concurrency and `URLSessionWebSocketTask` for scheduler transport
- CryptoKit for a stable worker identity and device public key
- Metal capability discovery (with CPU fallback)
- A backend-neutral inference interface ready for llama.cpp/ggml Metal bindings
- A benchmark JSON schema compatible with the Android benchmark output

## Requirements

- iOS 16 or newer
- arm64 device (simulator registration is supported, but inference is not)
- Xcode 15 or newer
- A minimum of 3 GB free storage for the Qwen2.5-3B-Instruct Q4_K_M model

## Add to an iOS app

Add this directory as a local Swift package in Xcode, then create a worker service:

```swift
import IntelHiveWorker

let runtime = WorkerRuntime(
    configuration: .init(schedulerURL: URL(string: "ws://192.168.1.10:8080/ws")!)
)
await runtime.start()
```

The runtime registers as `platform: "ios"`, reports `backend: "metal"` when Metal is available, and remains `UNBENCHMARKED` until an inference backend is installed and a benchmark is run.

## Build and test

```bash
cd ios-worker
swift test
```

## Inference backend

`InferenceEngine` intentionally contains no vendored model binary or third-party native code. Connect the llama.cpp iOS/Metal bridge by implementing `InferenceBackend` and pass it to `BenchmarkService`. This keeps model licensing, binary size, and upstream llama.cpp updates separate from the worker protocol.

## App lifecycle

iOS may suspend background apps. A production app should run the worker while foregrounded or use an appropriate permitted background mode; the runtime does not attempt to bypass iOS lifecycle restrictions.
