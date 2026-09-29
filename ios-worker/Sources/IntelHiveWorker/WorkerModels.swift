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

    public init(
        schedulerURL: URL,
        protocolVersion: String = "phase-0-v2",
        heartbeatInterval: Duration = .seconds(15)
    ) {
        self.schedulerURL = schedulerURL
        self.protocolVersion = protocolVersion
        self.heartbeatInterval = heartbeatInterval
    }
}

public struct LoadedShard: Codable, Sendable, Equatable {
    public let modelID: String
    public let shardID: String
    public let layerStart: Int
    public let layerEnd: Int

    public init(modelID: String, shardID: String, layerStart: Int, layerEnd: Int) {
        self.modelID = modelID
        self.shardID = shardID
        self.layerStart = layerStart
        self.layerEnd = layerEnd
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case shardID = "shard_id"
        case layerStart = "layer_start"
        case layerEnd = "layer_end"
    }
}

public struct WorkerRegisterPayload: Codable, Sendable {
    public let protocolVersion: String
    public let workerID: String
    public let platform: String
    public let osVersion: String
    public let deviceID: String
    public let devicePublicKey: String
    public let memory: MemoryInfo
    public let compute: ComputeInfo
    public let inference: InferenceInfo
    public let network: NetworkInfo
    public let power: PowerInfo
    public let loadedShards: [LoadedShard]

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case workerID = "worker_id"
        case platform
        case osVersion = "os_version"
        case deviceID = "device_id"
        case devicePublicKey = "device_public_key"
        case memory
        case compute
        case inference
        case network
        case power
        case loadedShards = "loaded_shards"
    }
}

public struct MemoryInfo: Codable, Sendable {
    public let totalMB: UInt64
    public let availableMB: UInt64

    enum CodingKeys: String, CodingKey {
        case totalMB = "total_mb"
        case availableMB = "available_mb"
    }
}

public struct ComputeInfo: Codable, Sendable {
    public let cpu: ComputeUnitInfo
    public let gpu: GPUInfo
    public let npu: ComputeUnitInfo
}

public struct ComputeUnitInfo: Codable, Sendable {
    public let available: Bool
}

public struct GPUInfo: Codable, Sendable {
    public let available: Bool
    public let vendor: String
    public let name: String
}

public struct InferenceInfo: Codable, Sendable {
    public let runtime: String
    public let backend: String
    public let benchmarkStatus: String
    public let performance: PerformanceInfo?

    enum CodingKeys: String, CodingKey {
        case runtime
        case backend
        case benchmarkStatus = "benchmark_status"
        case performance
    }
}

public struct PerformanceInfo: Codable, Sendable {
    public let tokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case tokensPerSecond = "tokens_per_second"
    }
}

public struct NetworkInfo: Codable, Sendable {
    public let type: String
}

public struct PowerInfo: Codable, Sendable {
    public let batteryPercent: UInt32
    public let charging: Bool

    enum CodingKeys: String, CodingKey {
        case batteryPercent = "battery_percent"
        case charging
    }
}

public struct HeartbeatPayload: Codable, Sendable {
    public let workerID: String
    public let state: WorkerState
    public let batteryPercent: UInt32
    public let charging: Bool
    public let temperatureC: Float?
    public let utilization: Float?
    public let tokensPerSecond: Double?
    public let timestamp: Date

    enum CodingKeys: String, CodingKey {
        case workerID = "worker_id"
        case state
        case batteryPercent = "battery_percent"
        case charging
        case temperatureC = "temperature_c"
        case utilization
        case tokensPerSecond = "tokens_per_second"
        case timestamp
    }
}

struct WorkerEnvelope<Payload: Encodable & Sendable>: Encodable, Sendable {
    let type: String
    let payload: Payload
}

struct WorkerServerEnvelope<Payload: Decodable>: Decodable {
    let type: String
    let payload: Payload
}

struct WorkerServerMessageHeader: Decodable {
    let type: String
}

public struct WorkerRegisterAcknowledgement: Decodable, Sendable {
    public let accepted: Bool
    public let workerID: String
    public let heartbeatIntervalSeconds: UInt32
    public let state: String
    public let reason: String?

    enum CodingKeys: String, CodingKey {
        case accepted
        case workerID = "worker_id"
        case heartbeatIntervalSeconds = "heartbeat_interval_seconds"
        case state
        case reason
    }
}

public struct HeartbeatAcknowledgement: Decodable, Sendable {
    public let workerID: String
    public let ack: Bool

    enum CodingKeys: String, CodingKey {
        case workerID = "worker_id"
        case ack
    }
}

struct WorkerErrorPayload: Decodable {
    let message: String
}

enum WorkerMessageDecoder {
    static func decode<Payload: Decodable>(
        _ payloadType: Payload.Type,
        from data: Data,
        expectedType: String,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> WorkerServerEnvelope<Payload> {
        let header = try decoder.decode(WorkerServerMessageHeader.self, from: data)
        if header.type == "error" {
            let error = try decoder.decode(WorkerServerEnvelope<WorkerErrorPayload>.self, from: data)
            throw WorkerRuntimeError.serverRejectedMessage(error.payload.message)
        }
        guard header.type == expectedType else {
            throw WorkerRuntimeError.unexpectedServerMessage(header.type)
        }
        return try decoder.decode(WorkerServerEnvelope<Payload>.self, from: data)
    }
}
