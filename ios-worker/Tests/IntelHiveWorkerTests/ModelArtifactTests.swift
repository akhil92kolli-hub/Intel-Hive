import XCTest
@testable import IntelHiveWorker

final class ModelArtifactTests: XCTestCase {
    func testBundledManifestMatchesPinnedNativeArtifact() throws {
        let manifest = try RequiredModelManifest.bundled()

        XCTAssertEqual(manifest.modelID, RequiredModelManifest.pinnedModelID)
        XCTAssertEqual(manifest.version, RequiredModelManifest.pinnedVersion)
        XCTAssertEqual(manifest.artifactDigest, RequiredModelManifest.pinnedDigest)
        XCTAssertEqual(manifest.sizeBytes, RequiredModelManifest.pinnedSizeBytes)
        XCTAssertEqual(manifest.downloadURL.scheme, "https")
    }

    func testManifestRejectsInsecureDownloadURL() throws {
        let json = """
        {
          "model_id":"qwen2.5-3b-instruct",
          "version":"1.0.0",
          "artifact_digest":"sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
          "architecture":"qwen2",
          "layers":36,
          "hidden_size":2048,
          "shards":[{
            "download_url":"http://example.test/model.gguf",
            "sha256":"sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
            "size_bytes":2104932768
          }],
          "total_size_bytes":2104932768
        }
        """

        XCTAssertThrowsError(try RequiredModelManifest.decode(Data(json.utf8)))
    }
}
