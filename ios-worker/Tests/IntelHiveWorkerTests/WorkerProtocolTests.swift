import Foundation
import XCTest
@testable import IntelHiveWorker

final class WorkerProtocolTests: XCTestCase {
    func testRegistrationEnvelopeMatchesPhaseZeroServerContract() throws {
        let payload = WorkerRegisterPayload(
            protocolVersion: "phase-0-v2",
            workerID: "ios-test-worker",
            platform: "ios",
            osVersion: "18.0",
            deviceID: "ios-test-worker",
            devicePublicKey: "public-key",
            memory: MemoryInfo(totalMB: 8192, availableMB: 4096),
            compute: ComputeInfo(
                cpu: ComputeUnitInfo(available: true),
                gpu: GPUInfo(available: true, vendor: "Apple", name: "Apple GPU"),
                npu: ComputeUnitInfo(available: false)
            ),
            inference: InferenceInfo(
                runtime: "llama.cpp",
                backend: "metal",
                benchmarkStatus: "NOT_RUN",
                performance: nil
            ),
            network: NetworkInfo(type: "wifi"),
            power: PowerInfo(batteryPercent: 80, charging: true),
            loadedShards: [LoadedShard(modelID: "qwen-3b", shardID: "shard-0-9", layerStart: 0, layerEnd: 9)]
        )
        let encoder = JSONEncoder()
        let data = try encoder.encode(WorkerEnvelope(type: "register", payload: payload))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["type"] as? String, "register")

        let body = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(body["protocol_version"] as? String, "phase-0-v2")
        XCTAssertEqual(body["worker_id"] as? String, "ios-test-worker")
        XCTAssertEqual(body["platform"] as? String, "ios")
        XCTAssertNotNil(body["device_id"])

        let memory = try XCTUnwrap(body["memory"] as? [String: Any])
        XCTAssertEqual(memory["total_mb"] as? UInt64, 8192)
        XCTAssertEqual(memory["available_mb"] as? UInt64, 4096)

        let inference = try XCTUnwrap(body["inference"] as? [String: Any])
        XCTAssertEqual(inference["runtime"] as? String, "llama.cpp")
        XCTAssertEqual(inference["benchmark_status"] as? String, "NOT_RUN")

        let shards = try XCTUnwrap(body["loaded_shards"] as? [[String: Any]])
        XCTAssertEqual(shards.first?["model_id"] as? String, "qwen-3b")
        XCTAssertEqual(shards.first?["layer_end"] as? Int, 9)
    }

    func testHeartbeatEncodesServerExpectedTimestampAndFields() throws {
        let heartbeat = HeartbeatPayload(
            workerID: "ios-test-worker",
            state: .unbenchmarked,
            batteryPercent: 65,
            charging: false,
            temperatureC: nil,
            utilization: nil,
            tokensPerSecond: nil,
            timestamp: Date(timeIntervalSince1970: 0)
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(WorkerEnvelope(type: "heartbeat", payload: heartbeat))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["type"] as? String, "heartbeat")

        let body = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(body["worker_id"] as? String, "ios-test-worker")
        XCTAssertEqual(body["state"] as? String, "UNBENCHMARKED")
        XCTAssertEqual(body["battery_percent"] as? UInt32, 65)
        XCTAssertEqual(body["timestamp"] as? String, "1970-01-01T00:00:00Z")
    }

    func testRegistrationAcknowledgementDecodesGatewayResponse() throws {
        let data = Data(
            #"{"type":"register_ack","payload":{"accepted":true,"worker_id":"ios-test-worker","heartbeat_interval_seconds":15,"state":"UNBENCHMARKED"}}"#.utf8
        )
        let response = try JSONDecoder().decode(
            WorkerServerEnvelope<WorkerRegisterAcknowledgement>.self,
            from: data
        )
        XCTAssertEqual(response.type, "register_ack")
        XCTAssertTrue(response.payload.accepted)
        XCTAssertEqual(response.payload.workerID, "ios-test-worker")
        XCTAssertEqual(response.payload.heartbeatIntervalSeconds, 15)
    }

    func testServerErrorEnvelopeSurfacesGatewayReason() {
        let data = Data(#"{"type":"error","payload":{"message":"register before sending heartbeats"}}"#.utf8)
        XCTAssertThrowsError(
            try WorkerMessageDecoder.decode(
                HeartbeatAcknowledgement.self,
                from: data,
                expectedType: "heartbeat_ack"
            )
        ) { error in
            XCTAssertEqual(
                error as? WorkerRuntimeError,
                .serverRejectedMessage("register before sending heartbeats")
            )
        }
    }

    func testUnexpectedMessageTypeIsRejectedBeforePayloadDecode() {
        let data = Data(#"{"type":"heartbeat_ack","payload":{"worker_id":"worker","ack":true}}"#.utf8)
        XCTAssertThrowsError(
            try WorkerMessageDecoder.decode(
                WorkerRegisterAcknowledgement.self,
                from: data,
                expectedType: "register_ack"
            )
        ) { error in
            XCTAssertEqual(error as? WorkerRuntimeError, .unexpectedServerMessage("heartbeat_ack"))
        }
    }
}
