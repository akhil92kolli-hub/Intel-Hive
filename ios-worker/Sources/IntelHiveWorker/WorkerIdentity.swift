import Foundation
import CryptoKit

public struct WorkerIdentity: Sendable {
    public let workerID: String
    public let publicKey: String
    private let privateKey: P256.Signing.PrivateKey

    public init(store: UserDefaults = .standard) {
        let idKey = "intel-hive.worker-id"
        let keyKey = "intel-hive.private-key"
        workerID = store.string(forKey: idKey) ?? {
            let value = "ios-\(UUID().uuidString.lowercased())"
            store.set(value, forKey: idKey)
            return value
        }()

        if let encoded = store.string(forKey: keyKey),
           let data = Data(base64Encoded: encoded),
           let key = try? P256.Signing.PrivateKey(rawRepresentation: data) {
            privateKey = key
        } else {
            let key = P256.Signing.PrivateKey()
            privateKey = key
            store.set(key.rawRepresentation.base64EncodedString(), forKey: keyKey)
        }
        publicKey = privateKey.publicKey.rawRepresentation.base64EncodedString()
    }

    public func sign(_ data: Data) throws -> Data {
        try privateKey.signature(for: data).rawRepresentation
    }
}
