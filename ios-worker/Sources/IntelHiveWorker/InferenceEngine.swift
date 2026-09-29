import Foundation

public protocol InferenceBackend: Sendable {
    var name: String { get }
    func loadModel(at url: URL) async throws
    func benchmark(prefillTokens: Int, generatedTokens: Int) async throws -> BenchmarkResult
}

public struct BenchmarkResult: Codable, Sendable {
    public let timestamp: Date
    public let model: String
    public let quantization: String
    public let prefillTokens: Int
    public let generatedTokens: Int
    public let totalTimeMS: Int64
    public let tokensPerSecond: Double
    public let prefillSpeedTokensPerSecond: Double
    public let generationSpeedTokensPerSecond: Double

    public init(model: String, quantization: String, prefillTokens: Int, generatedTokens: Int,
                totalTimeMS: Int64, tokensPerSecond: Double,
                prefillSpeedTokensPerSecond: Double, generationSpeedTokensPerSecond: Double) {
        self.timestamp = Date(); self.model = model; self.quantization = quantization
        self.prefillTokens = prefillTokens; self.generatedTokens = generatedTokens
        self.totalTimeMS = totalTimeMS; self.tokensPerSecond = tokensPerSecond
        self.prefillSpeedTokensPerSecond = prefillSpeedTokensPerSecond
        self.generationSpeedTokensPerSecond = generationSpeedTokensPerSecond
    }
}

public actor InferenceEngine {
    private let backend: InferenceBackend
    public init(backend: InferenceBackend) { self.backend = backend }
    public func loadModel(at url: URL) async throws { try await backend.loadModel(at: url) }
    public func benchmark(prefillTokens: Int = 128, generatedTokens: Int = 100) async throws -> BenchmarkResult {
        try await backend.benchmark(prefillTokens: prefillTokens, generatedTokens: generatedTokens)
    }
}
