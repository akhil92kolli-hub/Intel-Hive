# Worker Protocol

Workers maintain a persistent WebSocket connection to the server at `/worker`.
Phase 0 carries JSON envelopes (`type` and `payload`) for registration,
heartbeats, job assignments, and job results; the `.proto` files define the
broader message shapes but are not yet generated or used by this socket
implementation.
Registration includes device capabilities and any loaded model shards. The
server marks unbenchmarked workers as `UNBENCHMARKED` and records heartbeats in
its in-memory registry. Disconnecting marks the worker `OFFLINE`.

The iOS Swift package and Android worker use the same JSON WebSocket envelope.
The iOS client waits for registration and heartbeat acknowledgments, uses the
server-negotiated heartbeat interval, and reports capability/power information.
The Go gateway dispatches scheduler assignments to WebSocket workers and
routes results back to the waiting `POST /inference` request. `/inference`
performs a prefill pass across every assigned shard followed by autoregressive
decode passes until EOS or `max_tokens` (default 32, maximum 256). The Android
and iOS clients currently implement registration and heartbeats only; native
job execution remains unimplemented.

This LAN development endpoint has no worker authentication and must not be
exposed to the public Internet. TLS, authentication, and remote-worker
discovery are M4 concerns.

Lifecycle:

1. `WorkerRegister`
2. `WorkerRegisterAck`
3. periodic `Heartbeat`
4. `ModelReady` for each loaded shard
5. `JobAssignment` and `JobAccepted` for each shard and token position
6. activation packets between pipeline stages
7. `JobComplete` or `JobFailed`

The JSON socket message types for the current execution path are
`job_assignment`, `job_accepted`, `job_complete`, and `job_failed`. Assignment
and result payloads are correlated by `assignment_id`; terminal results must
also carry the matching `job_id` and `worker_id`.

Activation envelopes carry explicit request, model/version, worker route,
position, layer, shape, dtype, and a SHA-256 checksum. Serialization must
remain simple until transport measurements justify compression or quantization.

## Autoregressive inference step contract

`InferenceStep` and its WebSocket `job_assignment` fields define the
token-loop contract. Each shard pass has a stable `sequence_id` for worker-local
KV-cache state, a phase (`PREFILL` or `DECODE`), and an increasing pass
`position` (prefill is zero). Workers track the actual model token offset after
tokenizing the prompt. Exactly one input form is
present: token IDs at the first shard or a serialized activation envelope at
later shards. During prefill, the prompt token IDs are processed at position zero.
During decode, one sampled token is processed at each increasing position.

Every non-final shard returns an activation for the next shard. The final shard
returns either one sampled token ID or an end-of-sequence signal and may return
the corresponding generated text fragment. The scheduler starts another decode
pass with that token until EOS or the requested token limit. The scheduler sends `sequence_end` to each assigned worker after completion or
when abandoning a failed attempt. Workers must discard sequence-local KV state
when receiving that message. Result job, sequence, and position values must
match the corresponding step.
The server dispatches this loop now; mobile clients must implement these
assignment/result fields before they can execute inference.
