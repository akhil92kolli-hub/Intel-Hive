import Foundation

public actor SchedulerClient {
    private let configuration: WorkerConfiguration
    private let identity: WorkerIdentity
    private let capabilities: DeviceCapabilities
    private var socket: URLSessionWebSocketTask?

    public init(configuration: WorkerConfiguration, identity: WorkerIdentity = WorkerIdentity()) {
        self.configuration = configuration; self.identity = identity; self.capabilities = .current()
    }

    public func connect() async throws {
        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: configuration.schedulerURL)
        task.resume(); socket = task
        try await sendRegister()
    }

    public func sendRegister(benchmarkScore: Double = 0) async throws {
        guard let socket else { throw URLError(.notConnected) }
        let payload = WorkerRegisterPayload(protocolVersion: configuration.protocolVersion, workerID: identity.workerID,
            devicePublicKey: identity.publicKey, platform: "ios", osVersion: capabilities.osVersion,
            architecture: capabilities.architecture, ramMB: capabilities.ramMB, gpuVendor: capabilities.gpuVendor,
            gpuName: capabilities.gpuName, inferenceBackend: capabilities.backend, charging: false,
            batteryPercent: 0, networkType: "unknown", benchmarkScore: benchmarkScore)
        let data = try JSONEncoder().encode(payload)
        try await socket.send(.data(data))
    }

    public func sendHeartbeat(state: WorkerState, tokensPerSecond: Double = 0) async throws {
        guard let socket else { throw URLError(.notConnected) }
        let heartbeat = HeartbeatPayload(workerID: identity.workerID, state: state,
            batteryPercent: 0, charging: false, temperatureC: 0, utilization: 0,
            tokensPerSecond: tokensPerSecond)
        try await socket.send(.data(try JSONEncoder().encode(heartbeat)))
    }

    public func close() { socket?.cancel(with: .goingAway, reason: nil); socket = nil }
}

public actor WorkerRuntime {
    private let configuration: WorkerConfiguration
    private let client: SchedulerClient
    private var state: WorkerState = .unknown

    public init(configuration: WorkerConfiguration) {
        self.configuration = configuration; self.client = SchedulerClient(configuration: configuration)
    }

    public func start() async {
        state = .registering
        do { try await client.connect(); state = .unbenchmarked; try await heartbeatLoop() }
        catch { state = .offline }
    }

    public func stop() { client.close(); state = .offline }
    public func currentState() -> WorkerState { state }

    private func heartbeatLoop() async throws {
        while state != .offline {
            try await client.sendHeartbeat(state: state)
            try await Task.sleep(for: configuration.heartbeatInterval)
        }
    }
}
