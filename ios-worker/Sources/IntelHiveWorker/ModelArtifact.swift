import CryptoKit
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct RequiredModelManifest: Sendable, Equatable {
    public let modelID: String
    public let version: String
    public let architecture: String
    public let layerCount: Int
    public let hiddenSize: Int
    public let artifactDigest: String
    public let sizeBytes: Int64
    public let downloadURL: URL

    public static let fileName = "qwen2.5-3b-instruct-q4_k_m.gguf"
    public static let pinnedModelID = "qwen2.5-3b-instruct"
    public static let pinnedVersion = "1.0.0"
    public static let pinnedDigest =
        "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
    public static let pinnedSizeBytes: Int64 = 2_104_932_768

    public static func bundled() throws -> RequiredModelManifest {
        guard let url = Bundle.module.url(
            forResource: "qwen2.5-3b-instruct", withExtension: "json"
        ) else {
            throw ModelArtifactError.bundledManifestMissing
        }
        return try decode(Data(contentsOf: url))
    }

    public static func decode(_ data: Data) throws -> RequiredModelManifest {
        let decoded = try JSONDecoder().decode(ManifestDocument.self, from: data)
        guard let shard = decoded.shards.first, let url = URL(string: shard.downloadURL) else {
            throw ModelArtifactError.invalidManifest("a download shard with a valid URL is required")
        }
        let manifest = RequiredModelManifest(
            modelID: decoded.modelID,
            version: decoded.version,
            architecture: decoded.architecture,
            layerCount: decoded.layerCount,
            hiddenSize: decoded.hiddenSize,
            artifactDigest: decoded.artifactDigest,
            sizeBytes: decoded.totalSizeBytes,
            downloadURL: url
        )
        try manifest.validate(shard: shard)
        return manifest
    }

    public func validate() throws {
        guard modelID == Self.pinnedModelID else {
            throw ModelArtifactError.invalidManifest("unsupported model_id")
        }
        guard version == Self.pinnedVersion else {
            throw ModelArtifactError.invalidManifest("unsupported model version")
        }
        guard architecture == "qwen2", layerCount == 36, hiddenSize == 2048 else {
            throw ModelArtifactError.invalidManifest("model dimensions do not match the native executor")
        }
        guard artifactDigest == Self.pinnedDigest, sizeBytes == Self.pinnedSizeBytes else {
            throw ModelArtifactError.invalidManifest("artifact digest or size does not match the pinned model")
        }
        guard downloadURL.scheme?.lowercased() == "https", downloadURL.host != nil else {
            throw ModelArtifactError.invalidManifest("model download URL must use HTTPS")
        }
    }

    private func validate(shard: DownloadShard) throws {
        try validate()
        guard shard.sha256 == artifactDigest, shard.sizeBytes == sizeBytes else {
            throw ModelArtifactError.invalidManifest("download shard digest and size must match the artifact")
        }
    }
}

public enum ModelManifestSource: Sendable {
    case bundled
    case supabase(SupabaseConfiguration = .intelHive)
    case remote(URL)
}

public enum ModelArtifactError: LocalizedError, Sendable {
    case bundledManifestMissing
    case invalidManifest(String)
    case invalidManifestResponse
    case manifestTooLarge
    case unverifiedArtifact
    case sizeMismatch(expected: Int64, actual: Int64)
    case digestMismatch
    case storageUnavailable

    public var errorDescription: String? {
        switch self {
        case .bundledManifestMissing:
            return "The bundled model manifest is missing."
        case .invalidManifest(let reason):
            return "Invalid model manifest: \(reason)."
        case .invalidManifestResponse:
            return "The model manifest endpoint returned an invalid response."
        case .manifestTooLarge:
            return "The model manifest exceeds the size limit."
        case .unverifiedArtifact:
            return "The requested model artifact is not verified."
        case .sizeMismatch(let expected, let actual):
            return "Model size mismatch: expected \(expected) bytes, got \(actual)."
        case .digestMismatch:
            return "Model SHA-256 does not match the manifest."
        case .storageUnavailable:
            return "App model storage is unavailable."
        }
    }
}

/// Owns only verified model artifacts. An iOS app host can expose download
/// progress and cancellation through its URLSession delegate, then call
/// `installVerifiedDownload` when its temporary download completes.
public actor RequiredModelManager {
    private let storageDirectory: URL
    private let session: URLSession
    private let fileManager: FileManager

    public init(
        storageDirectory: URL? = nil,
        session: URLSession = .shared,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager
        self.session = session
        self.storageDirectory = storageDirectory ?? Self.defaultStorageDirectory(fileManager: fileManager)
    }

    public func loadManifest(
        from source: ModelManifestSource = .supabase()
    ) async throws -> RequiredModelManifest {
        switch source {
        case .bundled:
            return try RequiredModelManifest.bundled()
        case .supabase(let configuration):
            return try await loadRemoteManifest(from: configuration.modelManifestURL)
        case .remote(let url):
            return try await loadRemoteManifest(from: url)
        }
    }

    public func installedFile(for manifest: RequiredModelManifest) -> URL {
        storageDirectory.appendingPathComponent(RequiredModelManifest.fileName)
    }

    /// Rechecks the full digest before returning a model URL to a native backend.
    public func verifiedInstalledFile(for manifest: RequiredModelManifest) throws -> URL? {
        try manifest.validate()
        let file = installedFile(for: manifest)
        guard fileManager.fileExists(atPath: file.path) else { return nil }
        try verify(file: file, against: manifest)
        return file
    }

    /// Copies a completed URLSession temporary download to a private `.part`
    /// file, verifies exact bytes and SHA-256, then promotes only that verified
    /// file to the native-model location.
    @discardableResult
    public func installVerifiedDownload(
        temporaryFile: URL,
        manifest: RequiredModelManifest
    ) throws -> URL {
        try manifest.validate()
        try ensureStorageDirectory()
        let partial = storageDirectory.appendingPathComponent(RequiredModelManifest.fileName + ".part")
        if fileManager.fileExists(atPath: partial.path) {
            try fileManager.removeItem(at: partial)
        }
        try fileManager.copyItem(at: temporaryFile, to: partial)
        do {
            try verify(file: partial, against: manifest)
            let destination = installedFile(for: manifest)
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: partial, to: destination)
            return destination
        } catch {
            try? fileManager.removeItem(at: partial)
            throw error
        }
    }

    public func removeInstalledModel() throws {
        let destination = storageDirectory.appendingPathComponent(RequiredModelManifest.fileName)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
    }

    public static func defaultStorageDirectory(fileManager: FileManager = .default) -> URL {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        return base.appendingPathComponent("IntelHive/models", isDirectory: true)
    }

    private func ensureStorageDirectory() throws {
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: storageDirectory.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else { throw ModelArtifactError.storageUnavailable }
            return
        }
        try fileManager.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
    }

    private func loadRemoteManifest(from url: URL) async throws -> RequiredModelManifest {
        guard url.scheme?.lowercased() == "https", url.host != nil else {
            throw ModelArtifactError.invalidManifest("configured manifest URL must use HTTPS")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse,
              (200...299).contains(response.statusCode) else {
            throw ModelArtifactError.invalidManifestResponse
        }
        guard response.expectedContentLength <= Self.maximumManifestBytes,
              data.count <= Self.maximumManifestBytes else {
            throw ModelArtifactError.manifestTooLarge
        }
        return try RequiredModelManifest.decode(data)
    }

    private func verify(file: URL, against manifest: RequiredModelManifest) throws {
        let attributes = try fileManager.attributesOfItem(atPath: file.path)
        let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? -1
        guard byteCount == manifest.sizeBytes else {
            throw ModelArtifactError.sizeMismatch(expected: manifest.sizeBytes, actual: byteCount)
        }
        let digest = try sha256(file: file)
        guard "sha256:\(digest)" == manifest.artifactDigest else {
            throw ModelArtifactError.digestMismatch
        }
    }

    private func sha256(file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1_048_576) ?? Data()
            if chunk.isEmpty { break }
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static let maximumManifestBytes = 256 * 1024
}

private struct ManifestDocument: Decodable {
    let modelID: String
    let version: String
    let architecture: String
    let layerCount: Int
    let hiddenSize: Int
    let artifactDigest: String
    let totalSizeBytes: Int64
    let shards: [DownloadShard]

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case version
        case architecture
        case layerCount = "layers"
        case hiddenSize = "hidden_size"
        case artifactDigest = "artifact_digest"
        case totalSizeBytes = "total_size_bytes"
        case shards
    }
}

private struct DownloadShard: Decodable {
    let downloadURL: String
    let sha256: String
    let sizeBytes: Int64

    enum CodingKeys: String, CodingKey {
        case downloadURL = "download_url"
        case sha256
        case sizeBytes = "size_bytes"
    }
}
