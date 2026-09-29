import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public enum SupabaseUploadError: LocalizedError, Sendable, Equatable {
    case invalidConfiguration
    case invalidResponse
    case rejected(statusCode: Int, detail: String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            return "Supabase benchmark upload is not configured correctly."
        case .invalidResponse:
            return "Supabase returned an invalid benchmark upload response."
        case .rejected(let statusCode, let detail):
            let suffix = detail.isEmpty ? "" : ": \(detail)"
            return "Supabase benchmark upload failed with HTTP \(statusCode)\(suffix)"
        }
    }
}

public final class SupabaseBenchmarkUploader: @unchecked Sendable {
    private let configuration: SupabaseConfiguration
    private let session: URLSession
    private let encoder = JSONEncoder()

    public init(
        configuration: SupabaseConfiguration = .intelHive,
        session: URLSession = .shared
    ) {
        self.configuration = configuration
        self.session = session
    }

    func upload(_ row: SupabaseBenchmarkRow) async throws {
        try configuration.validate()
        let endpoint = configuration.projectURL
            .appendingPathComponent("rest/v1/benchmark_results")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue(configuration.publishableKey, forHTTPHeaderField: "apikey")
        request.setValue(
            "Bearer \(configuration.publishableKey)",
            forHTTPHeaderField: "Authorization"
        )
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("return=minimal", forHTTPHeaderField: "Prefer")
        request.httpBody = try encoder.encode(row)

        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw SupabaseUploadError.invalidResponse
        }
        guard (200...299).contains(response.statusCode) else {
            let detail = String(data: data.prefix(512), encoding: .utf8) ?? ""
            throw SupabaseUploadError.rejected(
                statusCode: response.statusCode,
                detail: detail
            )
        }
    }
}
