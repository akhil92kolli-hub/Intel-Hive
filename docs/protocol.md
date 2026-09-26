# Worker Protocol

Workers maintain a persistent WebSocket connection to the server. The payload inside each WebSocket message is a Protocol Buffer envelope. Phase 0 uses LAN/Wi-Fi and does not define Internet federation.

Lifecycle:

1. `WorkerRegister`
2. `WorkerRegisterAck`
3. periodic `Heartbeat`
4. `ModelReady` for each loaded shard
5. `JobAssignment` and `JobAccepted`
6. activation packets between pipeline stages
7. `JobComplete` or `JobFailed`

Activation packets carry explicit shape, dtype, sequence, and a SHA-256 checksum. Serialization must remain simple until transport measurements justify compression or quantization.
