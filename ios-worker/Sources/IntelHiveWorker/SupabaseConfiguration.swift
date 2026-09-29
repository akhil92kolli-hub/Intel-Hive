import Foundation

public struct SupabaseConfiguration: Sendable, Equatable {
    public let projectURL: URL
    public let publishableKey: String
    public let modelManifestURL: URL

    public init(projectURL: URL, publishableKey: String, modelManifestURL: URL) {
        self.projectURL = projectURL
        self.publishableKey = publishableKey
        self.modelManifestURL = modelManifestURL
    }

    public static let intelHive = SupabaseConfiguration(
        projectURL: URL(string: "https://uozyxansakogtpqxcpdp.supabase.co")!,
        publishableKey: "sb_publishable_tmCxtqxpIaNKrimWVidtxA_BnULyxkK",
        modelManifestURL: URL(
            string: "https://uozyxansakogtpqxcpdp.supabase.co/storage/v1/object/public/" +
                "model-artifacts/manifests/qwen2.5-3b-instruct/1.0.0/manifest.json"
        )!
    )

    func validate() throws {
        guard projectURL.scheme?.lowercased() == "https", projectURL.host != nil else {
            throw SupabaseUploadError.invalidConfiguration
        }
        guard publishableKey.hasPrefix("sb_publishable_") else {
            throw SupabaseUploadError.invalidConfiguration
        }
        guard modelManifestURL.scheme?.lowercased() == "https", modelManifestURL.host != nil else {
            throw SupabaseUploadError.invalidConfiguration
        }
    }
}
