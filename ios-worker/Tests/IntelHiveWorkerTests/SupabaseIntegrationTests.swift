import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import IntelHiveWorker

final class SupabaseIntegrationTests: XCTestCase {
    override func tearDown() {
        RequestCaptureURLProtocol.handler = nil
        super.tearDown()
    }

    func testSupabaseManifestSourceLoadsCanonicalManifest() async throws {
        let manifestData = Data(
            """
            {
              "model_id":"qwen2.5-3b-instruct",
              "version":"1.0.0",
              "artifact_digest":"sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
              "architecture":"qwen2",
              "layers":36,
              "hidden_size":2048,
              "shards":[{
                "download_url":"https://example.test/model.gguf",
                "sha256":"sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
                "size_bytes":2104932768
              }],
              "total_size_bytes":2104932768
            }
            """.utf8
        )
        RequestCaptureURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.absoluteString, "https://example.test/manifest.json")
            return (
                HTTPURLResponse(
                    url: try XCTUnwrap(request.url),
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Length": String(manifestData.count)]
                )!,
                manifestData
            )
        }
        let session = URLSession(configuration: .capturingRequests)
        let manager = RequiredModelManager(
            storageDirectory: FileManager.default.temporaryDirectory,
            session: session
        )
        let configuration = SupabaseConfiguration(
            projectURL: URL(string: "https://example.test")!,
            publishableKey: "sb_publishable_test",
            modelManifestURL: URL(string: "https://example.test/manifest.json")!
        )

        let manifest = try await manager.loadManifest(from: .supabase(configuration))

        XCTAssertEqual(manifest.modelID, RequiredModelManifest.pinnedModelID)
        XCTAssertEqual(manifest.artifactDigest, RequiredModelManifest.pinnedDigest)
    }

    func testBenchmarkUploadUsesPublishableKeyAndCanonicalRow() async throws {
        let configuration = SupabaseConfiguration(
            projectURL: URL(string: "https://example.test")!,
            publishableKey: "sb_publishable_test",
            modelManifestURL: URL(string: "https://example.test/manifest.json")!
        )
        RequestCaptureURLProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/rest/v1/benchmark_results")
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "apikey"), "sb_publishable_test")
            XCTAssertEqual(
                request.value(forHTTPHeaderField: "Authorization"),
                "Bearer sb_publishable_test"
            )
            XCTAssertEqual(request.value(forHTTPHeaderField: "Prefer"), "return=minimal")
            let body = try XCTUnwrap(request.httpBody)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            XCTAssertEqual(object["platform"] as? String, "ios")
            XCTAssertEqual(object["worker_id"] as? String, "ios-test-worker")
            XCTAssertEqual(object["model_id"] as? String, RequiredModelManifest.pinnedModelID)
            XCTAssertEqual(
                object["model_artifact_digest"] as? String,
                RequiredModelManifest.pinnedDigest
            )
            XCTAssertNotNil(object["raw_result"] as? [String: Any])
            return (
                HTTPURLResponse(
                    url: try XCTUnwrap(request.url),
                    statusCode: 201,
                    httpVersion: nil,
                    headerFields: nil
                )!,
                Data()
            )
        }
        let capabilities = DeviceCapabilities.current()
        let result = BenchmarkResult(
            model: RequiredModelManifest.pinnedModelID,
            quantization: "Q4_K_M",
            prefillTokens: 37,
            generatedTokens: 2,
            totalTimeMS: 1_000,
            tokensPerSecond: 2,
            prefillSpeedTokensPerSecond: 37,
            generationSpeedTokensPerSecond: 2
        )
        let row = SupabaseBenchmarkRow(
            clientRunID: UUID(),
            workerID: "ios-test-worker",
            appVersion: "0.1.0",
            backend: capabilities.backend,
            result: result,
            capabilities: capabilities,
            rawResult: BenchmarkOutput(result: result, capabilities: capabilities)
        )
        let uploader = SupabaseBenchmarkUploader(
            configuration: configuration,
            session: URLSession(configuration: .capturingRequests)
        )

        try await uploader.upload(row)
    }

    func testBenchmarkRemainsLocalWhenUploadFails() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let invalidSupabase = SupabaseConfiguration(
            projectURL: URL(string: "http://example.test")!,
            publishableKey: "invalid",
            modelManifestURL: URL(string: "http://example.test/manifest.json")!
        )
        let engine = InferenceEngine(backend: StubInferenceBackend())
        let service = BenchmarkService(
            engine: engine,
            workerID: "ios-test-worker",
            appVersion: "0.1.0",
            supabase: invalidSupabase,
            resultDirectory: directory
        )

        let data = try await service.run(modelURL: URL(fileURLWithPath: "/tmp/model.gguf"))
        let savedResultURL = await service.lastSavedResultURL()
        let savedURL = try XCTUnwrap(savedResultURL)
        let uploadFailure = await service.lastUploadFailure()

        XCTAssertFalse(data.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: savedURL.path))
        XCTAssertNotNil(uploadFailure)
    }
}

private struct StubInferenceBackend: InferenceBackend {
    let name = "stub"

    func loadModel(at url: URL) async throws {}

    func benchmark(prefillTokens: Int, generatedTokens: Int) async throws -> BenchmarkResult {
        BenchmarkResult(
            model: RequiredModelManifest.pinnedModelID,
            quantization: "Q4_K_M",
            prefillTokens: prefillTokens,
            generatedTokens: generatedTokens,
            totalTimeMS: 1_000,
            tokensPerSecond: 10,
            prefillSpeedTokensPerSecond: 20,
            generationSpeedTokensPerSecond: 10
        )
    }
}

private final class RequestCaptureURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            let handler = try XCTUnwrap(Self.handler)
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private extension URLSessionConfiguration {
    static var capturingRequests: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RequestCaptureURLProtocol.self]
        return configuration
    }
}
