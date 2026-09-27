# T007: Worker-to-Worker Activation Transport

## Objective

Create the transport layer that moves an activation from one worker to another without requiring physical phones or a hardware-specific runtime.

This is intentionally generic:

```text
Worker A
   │
   ├── produces activation
   │
   ▼
Transport
   │
   ▼
Worker B
```

This also supports:
- Android → Android
- Android → iOS
- iOS → Android
- mock → mock

The identity of the worker is separate from the transport format.

## Activation Object

The activation packet contains:

```go
type ActivationPacket struct {
    JobID             string
    RequestID         string
    SequenceID        uint64
    SourceWorker      string
    DestinationWorker string
    Layer             int
    DType             string
    Shape             []int64
    Payload           []byte
    Checksum          string
}
```

The transport layer is responsible for:
- serialization
- validation
- checksum verification
- payload routing
- retry / resend semantics (Phase-0 basic)

## Transport Interface

```go
type Transport interface {
    Send(packet ActivationPacket) error
    Receive() (ActivationPacket, error)
    Validate(packet ActivationPacket) error
}
```

For Phase-0, we can implement a local in-memory transport or a mock TCP/websocket abstraction.

## In-Memory Transport

The simplest implementation is an in-memory broker for local tests:

```go
type InMemoryTransport struct {
    queue chan ActivationPacket
}
```

- Worker A pushes activation into the queue
- Worker B reads from the queue
- transport validates checksum before delivering

## Worker-to-Worker Path

```text
Worker A -> Transport -> Worker B
```

Transport flow:

1. A creates activation using the logical layer range data
2. A serializes it to bytes
3. A sends packet to transport broker
4. transport verifies checksum and routes to destination worker
5. B receives packet, validates body, passes to next layer or final output

## Example Payload

```json
{
  "job_id": "job_123",
  "request_id": "req_456",
  "sequence_id": 17,
  "source_worker": "worker-a",
  "destination_worker": "worker-b",
  "layer": 12,
  "dtype": "f16",
  "shape": [1, 17, 2048],
  "payload": "base64-encoded-binary",
  "checksum": "sha256:..."
}
```

## Mistakes to avoid

- Don't couple transport to Android-specific APIs
- Don't couple transport to a specific worker runtime
- Don't invent a custom format that only one runtime understands
- Keep checksum validation mandatory

## Tests

Transport tests should prove:
- send → receive works with valid data
- checksum mismatch is rejected
- destination mismatch is preserved
- payload round-trips without mutation

## Next stage

Once this works, the next milestone is a mock worker implementation that can simulate a real shard and participate in the routing flow.
