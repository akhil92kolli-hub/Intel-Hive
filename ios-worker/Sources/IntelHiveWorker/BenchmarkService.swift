import Foundation

public actor BenchmarkService {
    private let engine: InferenceEngine
    private let capabilities: DeviceCapabilities
    private let workerID: String
    private let appVersion: String
    private let uploader: SupabaseBenchmarkUploader?
    private let resultDirectory: URL
    private let fileManager: FileManager
    private var savedResultURL: URL?
    private var uploadFailure: String?

    public init(
        engine: InferenceEngine,
        workerID: String = WorkerIdentity().workerID,
        appVersion: String? = nil,
        supabase: SupabaseConfiguration? = .intelHive,
        resultDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) {
        self.engine = engine
        self.capabilities = .current()
        self.workerID = workerID
        self.appVersion = appVersion ?? Self.defaultAppVersion()
        self.uploader = supabase.map { SupabaseBenchmarkUploader(configuration: $0) }
        self.fileManager = fileManager
        self.resultDirectory = resultDirectory ?? Self.defaultResultDirectory(fileManager: fileManager)
    }

    public func run(
        modelURL: URL,
        prefillTokens: Int = 128,
        generatedTokens: Int = 100
    ) async throws -> Data {
        try await engine.loadModel(at: modelURL)
        let result = try await engine.benchmark(
            prefillTokens: prefillTokens,
            generatedTokens: generatedTokens
        )
        let output = BenchmarkOutput(result: result, capabilities: capabilities)
        let data = try JSONEncoder().encode(output)
        let runID = UUID()
        savedResultURL = try persist(data, runID: runID)

        if let uploader {
            do {
                try await uploader.upload(
                    SupabaseBenchmarkRow(
                        clientRunID: runID,
                        workerID: workerID,
                        appVersion: appVersion,
                        backend: capabilities.backend,
                        result: result,
                        capabilities: capabilities,
                        rawResult: output
                    )
                )
                uploadFailure = nil
            } catch {
                uploadFailure = error.localizedDescription
            }
        }
        return data
    }

    public func lastSavedResultURL() -> URL? { savedResultURL }
    public func lastUploadFailure() -> String? { uploadFailure }

    public static func defaultResultDirectory(fileManager: FileManager = .default) -> URL {
        let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        return base.appendingPathComponent("IntelHive/benchmarks", isDirectory: true)
    }

    private func persist(_ data: Data, runID: UUID) throws -> URL {
        try fileManager.createDirectory(
            at: resultDirectory,
            withIntermediateDirectories: true
        )
        let destination = resultDirectory
            .appendingPathComponent("benchmark-\(runID.uuidString.lowercased()).json")
        try data.write(to: destination, options: .atomic)
        return destination
    }

    private static func defaultAppVersion() -> String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            ?? "0.1.0"
    }
}

struct BenchmarkOutput: Codable, Sendable {
    let timestamp: String
    let model: String
    let quantization: String
    let device: BenchmarkDeviceOutput
    let benchmark: BenchmarkMetricsOutput

    init(result: BenchmarkResult, capabilities: DeviceCapabilities) {
        timestamp = result.timestamp.ISO8601Format()
        model = result.model
        quantization = result.quantization
        device = BenchmarkDeviceOutput(capabilities: capabilities)
        benchmark = BenchmarkMetricsOutput(result: result)
    }
}

struct BenchmarkDeviceOutput: Codable, Sendable {
    let model: String
    let osVersion: String
    let ramMB: UInt64
    let gpu: String
    let backend: String

    enum CodingKeys: String, CodingKey {
        case model
        case osVersion = "os_version"
        case ramMB = "ram_mb"
        case gpu
        case backend
    }

    init(capabilities: DeviceCapabilities) {
        model = capabilities.gpuName
        osVersion = capabilities.osVersion
        ramMB = capabilities.ramMB
        gpu = capabilities.gpuName
        backend = capabilities.backend
    }
}

struct BenchmarkMetricsOutput: Codable, Sendable {
    let prefillTokens: Int
    let generatedTokens: Int
    let totalTimeMS: Int64
    let tokensPerSecond: Double
    let prefillSpeedTokensPerSecond: Double
    let generationSpeedTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case prefillTokens = "prefill_tokens"
        case generatedTokens = "generated_tokens"
        case totalTimeMS = "total_time_ms"
        case tokensPerSecond = "tokens_per_second"
        case prefillSpeedTokensPerSecond = "prefill_speed_tokens_per_second"
        case generationSpeedTokensPerSecond = "generation_speed_tokens_per_second"
    }

    init(result: BenchmarkResult) {
        prefillTokens = result.prefillTokens
        generatedTokens = result.generatedTokens
        totalTimeMS = result.totalTimeMS
        tokensPerSecond = result.tokensPerSecond
        prefillSpeedTokensPerSecond = result.prefillSpeedTokensPerSecond
        generationSpeedTokensPerSecond = result.generationSpeedTokensPerSecond
    }
}

struct SupabaseBenchmarkRow: Encodable, Sendable {
    let clientRunID: UUID
    let platform = "ios"
    let workerID: String
    let appVersion: String
    let modelID = RequiredModelManifest.pinnedModelID
    let modelVersion = RequiredModelManifest.pinnedVersion
    let modelArtifactDigest = RequiredModelManifest.pinnedDigest
    let deviceModel: String
    let osVersion: String
    let backend: String
    let prefillTokens: Int
    let generatedTokens: Int
    let totalTimeMS: Int64
    let tokensPerSecond: Double
    let prefillTokensPerSecond: Double
    let generationTokensPerSecond: Double
    let rawResult: BenchmarkOutput

    enum CodingKeys: String, CodingKey {
        case clientRunID = "client_run_id"
        case platform
        case workerID = "worker_id"
        case appVersion = "app_version"
        case modelID = "model_id"
        case modelVersion = "model_version"
        case modelArtifactDigest = "model_artifact_digest"
        case deviceModel = "device_model"
        case osVersion = "os_version"
        case backend
        case prefillTokens = "prefill_tokens"
        case generatedTokens = "generated_tokens"
        case totalTimeMS = "total_time_ms"
        case tokensPerSecond = "tokens_per_second"
        case prefillTokensPerSecond = "prefill_tokens_per_second"
        case generationTokensPerSecond = "generation_tokens_per_second"
        case rawResult = "raw_result"
    }

    init(
        clientRunID: UUID,
        workerID: String,
        appVersion: String,
        backend: String,
        result: BenchmarkResult,
        capabilities: DeviceCapabilities,
        rawResult: BenchmarkOutput
    ) {
        self.clientRunID = clientRunID
        self.workerID = workerID
        self.appVersion = appVersion
        self.deviceModel = capabilities.gpuName
        self.osVersion = capabilities.osVersion
        self.backend = backend
        self.prefillTokens = result.prefillTokens
        self.generatedTokens = result.generatedTokens
        self.totalTimeMS = result.totalTimeMS
        self.tokensPerSecond = result.tokensPerSecond
        self.prefillTokensPerSecond = result.prefillSpeedTokensPerSecond
        self.generationTokensPerSecond = result.generationSpeedTokensPerSecond
        self.rawResult = rawResult
    }
}
