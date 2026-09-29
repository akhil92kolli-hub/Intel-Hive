import Foundation

public actor BenchmarkService {
    private let engine: InferenceEngine
    private let capabilities = DeviceCapabilities.current()

    public init(engine: InferenceEngine) { self.engine = engine }

    public func run(modelURL: URL, prefillTokens: Int = 128, generatedTokens: Int = 100) async throws -> Data {
        try await engine.loadModel(at: modelURL)
        let result = try await engine.benchmark(prefillTokens: prefillTokens, generatedTokens: generatedTokens)
        let output: [String: AnyEncodable] = [
            "timestamp": AnyEncodable(result.timestamp.ISO8601Format()),
            "model": AnyEncodable(result.model), "quantization": AnyEncodable(result.quantization),
            "device": AnyEncodable(["os_version": capabilities.osVersion, "ram_mb": capabilities.ramMB,
                                     "gpu": capabilities.gpuName, "backend": capabilities.backend]),
            "benchmark": AnyEncodable(["prefill_tokens": result.prefillTokens, "generated_tokens": result.generatedTokens,
                                        "total_time_ms": result.totalTimeMS, "tokens_per_second": result.tokensPerSecond,
                                        "prefill_speed_tokens_per_second": result.prefillSpeedTokensPerSecond,
                                        "generation_speed_tokens_per_second": result.generationSpeedTokensPerSecond])
        ]
        return try JSONEncoder().encode(output)
    }
}

public struct AnyEncodable: Encodable {
    private let encodeValue: (Encoder) throws -> Void
    public init<T: Encodable>(_ value: T) { encodeValue = value.encode }
    public func encode(to encoder: Encoder) throws { try encodeValue(encoder) }
}
