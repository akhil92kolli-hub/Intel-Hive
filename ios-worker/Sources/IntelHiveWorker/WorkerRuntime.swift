import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public actor SchedulerClient {
    private let configuration: WorkerConfiguration
    private let identity: WorkerIdentity
    private let capabilities: DeviceCapabilities
    private let session: URLSession
    private var socket: URLSessionWebSocketTask?
    private var heartbeatInterval: Duration = .seconds(15)

    public init(configuration: WorkerConfiguration, identity: WorkerIdentity = WorkerIdentity()) {
        self.configuration = configuration
        self.identity = identity
        self.capabilities = .current()
        self.session = URLSession(configuration: .default)
    }

    @discardableResult
    public func connect() async throws -> WorkerRegisterAcknowledgement {
        guard configuration.schedulerURL.scheme == "ws" || configuration.schedulerURL.scheme == "wss" else {
            throw WorkerRuntimeError.invalidSchedulerURL
        }
        guard socket == nil else {
            throw WorkerRuntimeError.alreadyConnected
        }

        let task = session.webSocketTask(with: configuration.schedulerURL)
        task.resume()
        socket = task
        do {
            try await sendRegister()
            let acknowledgement: WorkerServerEnvelope<WorkerRegisterAcknowledgement> =
                try await receive(type: "register_ack")
            guard acknowledgement.payload.accepted else {
                throw WorkerRuntimeError.registrationRejected(
                    acknowledgement.payload.reason ?? "scheduler rejected worker registration"
                )
            }
            guard acknowledgement.payload.workerID == identity.workerID else {
                throw WorkerRuntimeError.registrationIdentityMismatch
            }
            heartbeatInterval = .seconds(
                Int64(acknowledgement.payload.heartbeatIntervalSeconds.clamped(to: 5...300))
            )
            return acknowledgement.payload
        } catch {
            close()
            throw error
        }
    }

    private func sendRegister() async throws {
        guard socket != nil else { throw URLError(.notConnectedToInternet) }
        let payload = WorkerRegisterPayload(
            protocolVersion: configuration.protocolVersion,
            workerID: identity.workerID,
            platform: "ios",
            osVersion: capabilities.osVersion,
            deviceID: identity.workerID,
            devicePublicKey: identity.publicKey,
            memory: MemoryInfo(
                totalMB: capabilities.ramMB,
                availableMB: capabilities.availableRAMMB
            ),
            compute: ComputeInfo(
                cpu: ComputeUnitInfo(available: true),
                gpu: GPUInfo(
                    available: capabilities.backend == "metal",
                    vendor: capabilities.gpuVendor,
                    name: capabilities.gpuName
                ),
                npu: ComputeUnitInfo(available: false)
            ),
            inference: InferenceInfo(
                runtime: "llama.cpp",
                backend: capabilities.backend,
                benchmarkStatus: "NOT_RUN",
                performance: nil
            ),
            network: NetworkInfo(type: await DeviceCapabilities.currentNetworkType()),
            power: DeviceCapabilities.powerStatus(),
            loadedShards: []
        )
        try await send(WorkerEnvelope(type: "register", payload: payload))
    }

    public func sendHeartbeat(
        state: WorkerState,
        tokensPerSecond: Double? = nil
    ) async throws {
        guard socket != nil else { throw URLError(.notConnectedToInternet) }
        let power = DeviceCapabilities.powerStatus()
        let heartbeat = HeartbeatPayload(
            workerID: identity.workerID,
            state: state,
            batteryPercent: power.batteryPercent,
            charging: power.charging,
            temperatureC: nil,
            utilization: nil,
            tokensPerSecond: tokensPerSecond,
            timestamp: Date()
        )
        try await send(WorkerEnvelope(type: "heartbeat", payload: heartbeat))
        let acknowledgement: WorkerServerEnvelope<HeartbeatAcknowledgement> =
            try await receive(type: "heartbeat_ack")
        guard acknowledgement.payload.ack, acknowledgement.payload.workerID == identity.workerID else {
            throw WorkerRuntimeError.invalidHeartbeatAcknowledgement
        }
    }

    public func negotiatedHeartbeatInterval() -> Duration { heartbeatInterval }

    public func close() {
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
    }

    private func send<Payload: Encodable & Sendable>(
        _ envelope: WorkerEnvelope<Payload>
    ) async throws {
        guard let socket else { throw URLError(.notConnectedToInternet) }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(envelope)
        guard let text = String(data: data, encoding: .utf8) else {
            throw WorkerRuntimeError.invalidEncodedMessage
        }
        try await socket.send(.string(text))
    }

    private func receive<Payload: Decodable>(
        type expectedType: String
    ) async throws -> WorkerServerEnvelope<Payload> {
        guard let socket else { throw URLError(.notConnectedToInternet) }
        let message = try await socket.receive()
        let data: Data
        switch message {
        case .string(let text):
            data = Data(text.utf8)
        case .data(let bytes):
            data = bytes
        @unknown default:
            throw WorkerRuntimeError.invalidServerMessage
        }

        return try WorkerMessageDecoder.decode(
            Payload.self,
            from: data,
            expectedType: expectedType
        )
    }
}

public actor WorkerRuntime {
    private let client: SchedulerClient
    private var state: WorkerState = .unknown
    private var heartbeatTask: Task<Void, Never>?
    private var lastFailure: String?

    public init(configuration: WorkerConfiguration) {
        self.client = SchedulerClient(configuration: configuration)
    }

    public func start() async throws {
        guard state == .unknown || state == .offline else {
            throw WorkerRuntimeError.alreadyStarted
        }
        state = .registering
        lastFailure = nil
        do {
            _ = try await client.connect()
            state = .unbenchmarked
            let interval = await client.negotiatedHeartbeatInterval()
            heartbeatTask = Task { [weak self] in
                guard let self else { return }
                await self.heartbeatLoop(interval: interval)
            }
        } catch {
            state = .offline
            lastFailure = error.localizedDescription
            await client.close()
            throw error
        }
    }

    public func stop() async {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        await client.close()
        state = .offline
    }

    public func currentState() -> WorkerState { state }
    public func failureReason() -> String? { lastFailure }

    private func heartbeatLoop(interval: Duration) async {
        do {
            while !Task.isCancelled {
                try await client.sendHeartbeat(state: state)
                try await Task.sleep(for: interval)
            }
        } catch is CancellationError {
            return
        } catch {
            state = .offline
            lastFailure = error.localizedDescription
            await client.close()
        }
    }
}

public enum WorkerRuntimeError: LocalizedError, Equatable {
    case invalidSchedulerURL
    case alreadyConnected
    case alreadyStarted
    case registrationRejected(String)
    case registrationIdentityMismatch
    case invalidHeartbeatAcknowledgement
    case invalidEncodedMessage
    case invalidServerMessage
    case unexpectedServerMessage(String)
    case serverRejectedMessage(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSchedulerURL:
            return "Scheduler URL must use ws:// or wss://."
        case .alreadyConnected:
            return "Scheduler client is already connected."
        case .alreadyStarted:
            return "Worker runtime is already starting or running."
        case .registrationRejected(let reason):
            return "Worker registration rejected: \(reason)"
        case .registrationIdentityMismatch:
            return "Scheduler acknowledged a different worker identity."
        case .invalidHeartbeatAcknowledgement:
            return "Scheduler returned an invalid heartbeat acknowledgment."
        case .invalidEncodedMessage:
            return "Worker message could not be encoded as UTF-8."
        case .invalidServerMessage:
            return "Scheduler returned an unsupported WebSocket message."
        case .unexpectedServerMessage(let type):
            return "Unexpected scheduler message type: \(type)."
        case .serverRejectedMessage(let reason):
            return "Scheduler rejected the worker message: \(reason)"
        }
    }
}

private extension UInt32 {
    func clamped(to range: ClosedRange<UInt32>) -> UInt32 {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
