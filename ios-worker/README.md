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
    configuration: .init(schedulerURL: URL(string: "ws://192.168.1.10:8080/worker")!)
)
Task {
    do {
        try await runtime.start()
        // Keep the application in the foreground while serving as a worker.
    } catch {
        print("Worker failed to connect: \(error.localizedDescription)")
    }
}
```

The runtime uses the Phase-0 JSON envelope protocol shared with the Go gateway:

- send `register` and wait for an accepted `register_ack`
- report iOS version, architecture, total/available RAM, CPU/Metal capability,
  network type, battery/charging state, and loaded-shard list
- send a heartbeat after registration and wait for its acknowledgment before
  sending the next one
- stop and expose the failure reason if registration or a heartbeat fails

It reports `backend: "metal"` when Metal is available and remains
`UNBENCHMARKED`; job dispatch and shard execution are not implemented yet.
The `device_id` currently uses the stable worker ID, and a generated P-256
public key is included as identity metadata. The gateway does not authenticate
or verify that key yet.

For local LAN testing, connect to `/worker` on the scheduler. iOS App Transport
Security may block cleartext `ws://` connections in the containing app. Prefer
`wss://` with a trusted certificate, or add a narrowly scoped ATS exception to
the **development app's** `Info.plist` for the test host; do not ship a broad
arbitrary-load exception.

## Build and test

```bash
cd ios-worker
swift test
```

## Inference backend

`InferenceEngine` intentionally contains no vendored model binary or third-party native code. Connect the llama.cpp iOS/Metal bridge by implementing `InferenceBackend` and pass it to `BenchmarkService`. This keeps model licensing, binary size, and upstream llama.cpp updates separate from the worker protocol.

## App lifecycle

iOS may suspend background apps. A production app should run the worker while foregrounded or use an appropriate permitted background mode; the runtime does not attempt to bypass iOS lifecycle restrictions.
