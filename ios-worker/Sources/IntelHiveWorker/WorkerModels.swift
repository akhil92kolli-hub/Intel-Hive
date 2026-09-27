import Foundation

public enum WorkerState: String, Codable, Sendable {
    case unknown = "UNKNOWN"
    case registering = "REGISTERING"
    case registered = "REGISTERED"
    case unbenchmarked = "UNBENCHMARKED"
    case ready = "READY"
    case busy = "BUSY"
    case throttled = "THROTTLED"
    case offline = "OFFLINE"
}

public struct WorkerConfiguration: Sendable {
    public let schedulerURL: URL
    public let protocolVersion: String
    public let heartbeatInterval: Duration

    public init(schedulerURL: URL, protocolVersion: String = "0.1", heartbeatInterval: Duration = .seconds(15)) {
        self.schedulerURL = schedulerURL
        self.protocolVersion = protocolVersion
        self.heartbeatInterval = heartbeatInterval
    }
}

public struct WorkerRegisterPayload: Codable, Sendable {
    public let protocolVersion: String
    public let workerID: String
    public let devicePublicKey: String
    public let platform: String
    public let osVersion: String
    public let architecture: String
    public let ramMB: UInt64
    public let gpuVendor: String
    public let gpuName: String
    public let inferenceBackend: String
    public let charging: Bool
    public let batteryPercent: UInt32
    public let networkType: String
    public let benchmarkScore: Double

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version", workerID = "worker_id"
        case devicePublicKey = "device_public_key", platform, osVersion = "os_version"
        case architecture, ramMB = "ram_mb", gpuVendor = "gpu_vendor", gpuName = "gpu_name"
        case inferenceBackend = "inference_backend", charging, batteryPercent = "battery_percent"
        case networkType = "network_type", benchmarkScore = "benchmark_score"
    }
}

public struct HeartbeatPayload: Codable, Sendable {
    public let workerID: String
    public let state: WorkerState
    public let batteryPercent: UInt32
    public let charging: Bool
    public let temperatureC: Float
    public let utilization: Float
    public let tokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case workerID = "worker_id", state
        case batteryPercent = "battery_percent", charging
        case temperatureC = "temperature_c", utilization
        case tokensPerSecond = "tokens_per_second"
    }
}
