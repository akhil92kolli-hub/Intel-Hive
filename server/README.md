# IntelHive worker gateway

The Phase-0 gateway accepts Android/iOS worker registration, heartbeats, and
job-result messages over a LAN WebSocket. `POST /inference` assigns model
shards to ready workers and runs prefill followed by token-by-token decode
across the shard pipeline. Workers remain
`UNBENCHMARKED` until a completed benchmark and loaded shards are advertised.
The Android and iOS apps do not yet execute job assignments, so this bridge
currently requires a protocol-compatible worker client or test worker.

## Run locally

```bash
cd server
go run ./cmd/scheduler
```

The gateway listens on `:8080` by default. Set `LISTEN_ADDR` to override it.
`GET /healthz` returns a basic health response, and workers connect to
`/worker`. `POST /inference` accepts:

```json
{"model_id":"qwen2.5-3b-instruct","prompt":"Hello","max_tokens":32}
```

The server currently supports the Phase-0 Qwen model IDs, both with 36 layers.
It requires the complete shard partition to be advertised by connected workers. A
successful response contains the job ID, generated output text when supplied
by the final worker, generated token IDs, and basic timing metrics.
`max_tokens` defaults to 32 and must be between 1 and 256. Inference requests
are bounded to 1 MiB.

Example Android emulator endpoint:

```text
ws://10.0.2.2:8080/worker
```

For a physical phone, use the development machine's LAN IP address, such as
`ws://192.168.1.10:8080/worker`. The debug Android build permits cleartext
WebSocket connections for this LAN-only development flow. Do not expose this
unauthenticated endpoint to the public Internet; remote-worker authentication
and TLS belong to M4.

## Worker WebSocket messages

Messages use a JSON envelope:

```json
{"type":"register","payload":{"protocol_version":"phase-0-v2","worker_id":"..."}}
{"type":"register_ack","payload":{"accepted":true,"worker_id":"...","heartbeat_interval_seconds":15,"state":"UNBENCHMARKED"}}
{"type":"heartbeat","payload":{"worker_id":"...","state":"UNBENCHMARKED","timestamp":"..."}}
{"type":"heartbeat_ack","payload":{"worker_id":"...","ack":true}}
```

Registration includes device memory, compute/backend availability, network and
power information, benchmark status, and any already-loaded model shards. A
worker must register before sending heartbeats. Disconnecting marks the worker
offline in the in-memory registry and scheduler.

The server sends a `job_assignment` for every shard in the prefill pass and
each decode pass. The first prefill shard receives the prompt text; following
shards receive an activation envelope. On decode, the first shard receives the
last sampled token ID and following shards receive activation envelopes. The final shard
must return either a sampled token ID plus an optional text fragment, or EOS.
Assignments carry a stable sequence ID and increasing token position so
workers can retain shard-local KV state. After EOS, reaching the token limit,
or abandoning a failed attempt, the server sends `sequence_end` so workers can
release that state. Workers reply with
`job_accepted`, followed by exactly one `job_complete` or `job_failed`
message. Completion and failure payloads must include the assignment ID, job
ID, and worker ID. Activation envelopes carry request/sequence/model identity,
pass position, source and destination worker, inclusive output layer, tensor
dtype/shape, Base64 payload, and SHA-256 checksum. Results are routed back to
the waiting inference request.
The current Android/iOS clients only implement registration/heartbeats and
cannot yet execute these assignments; the token loop is currently validated
with protocol-compatible workers, not real model execution.
