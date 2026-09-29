package transport

import (
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/protocol"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/registry"
	"github.com/gorilla/websocket"
)

func TestWorkerSocketRegistersAndAcceptsHeartbeats(t *testing.T) {
	workers := registry.NewWorkerRegistry()
	server := httptest.NewServer(NewWorkerSocketHandler(workers, nil))
	defer server.Close()

	endpoint, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	endpoint.Scheme = "ws"
	endpoint.Path = "/"
	conn, _, err := websocket.DefaultDialer.Dial(endpoint.String(), nil)
	if err != nil {
		t.Fatal(err)
	}

	registration := protocol.WorkerRegisterRequest{
		ProtocolVersion: protocol.ProtocolVersion,
		WorkerID:        "android-test-worker",
		Platform:        "android",
		Memory:          protocol.MemoryInfo{TotalMB: 4096, AvailableMB: 2048},
		Inference:       protocol.InferenceInfo{Runtime: "llama.cpp", Backend: "cpu", BenchmarkStatus: "NOT_RUN"},
		Power:           protocol.PowerInfo{BatteryPercent: 75, Charging: true},
		LoadedShards: []protocol.LoadedShardInfo{{ModelVersion: "v1",
			ModelID: "qwen-3b", ShardID: "shard-0-9", LayerStart: 0, LayerEnd: 9,
		}},
	}
	writeWorkerMessage(t, conn, "register", registration)
	var registerAck struct {
		Type    string                          `json:"type"`
		Payload protocol.WorkerRegisterResponse `json:"payload"`
	}
	if err := conn.ReadJSON(&registerAck); err != nil {
		t.Fatal(err)
	}
	if registerAck.Type != "register_ack" || !registerAck.Payload.Accepted || registerAck.Payload.State != string(registry.StateUnbenchmarked) {
		t.Fatalf("unexpected registration acknowledgment: %+v", registerAck)
	}

	heartbeat := protocol.Heartbeat{
		WorkerID: "android-test-worker", State: "READY", BatteryPercent: 72,
		Charging: true, TemperatureC: 37.5,
	}
	writeWorkerMessage(t, conn, "heartbeat", heartbeat)
	var heartbeatAck struct {
		Type    string                     `json:"type"`
		Payload protocol.HeartbeatResponse `json:"payload"`
	}
	if err := conn.ReadJSON(&heartbeatAck); err != nil {
		t.Fatal(err)
	}
	if heartbeatAck.Type != "heartbeat_ack" || !heartbeatAck.Payload.Ack {
		t.Fatalf("unexpected heartbeat acknowledgment: %+v", heartbeatAck)
	}

	worker, ok := workers.GetWorker("android-test-worker")
	if !ok {
		t.Fatal("registered worker missing from registry")
	}
	if worker.State != registry.StateUnbenchmarked {
		t.Fatalf("worker should remain unbenchmarked, got %q", worker.State)
	}
	if worker.Power.BatteryPercent != 72 || worker.Power.TemperatureC != 37.5 || !worker.Health.Healthy {
		t.Fatalf("heartbeat data not recorded: power=%+v health=%+v", worker.Power, worker.Health)
	}
	if len(worker.Models) != 1 || worker.Models[0].ModelID != "qwen-3b" ||
		len(worker.Models[0].ShardIDs) != 1 || worker.Models[0].ShardIDs[0] != "shard-0-9" {
		t.Fatalf("loaded shard capability not recorded: %+v", worker.Models)
	}

	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestWorkerSocketRejectsUnsupportedProtocol(t *testing.T) {
	server := httptest.NewServer(NewWorkerSocketHandler(registry.NewWorkerRegistry(), nil))
	defer server.Close()
	endpoint, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	endpoint.Scheme = "ws"
	conn, _, err := websocket.DefaultDialer.Dial(endpoint.String(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	writeWorkerMessage(t, conn, "register", protocol.WorkerRegisterRequest{
		ProtocolVersion: "unknown", WorkerID: "worker", Platform: "android",
	})
	var reply struct {
		Type    string                          `json:"type"`
		Payload protocol.WorkerRegisterResponse `json:"payload"`
	}
	if err := conn.ReadJSON(&reply); err != nil {
		t.Fatal(err)
	}
	if reply.Type != "register_ack" || reply.Payload.Accepted || reply.Payload.Reason == "" {
		t.Fatalf("expected rejected registration, got %+v", reply)
	}
}

func writeWorkerMessage(t *testing.T, conn *websocket.Conn, messageType string, payload any) {
	t.Helper()
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.WriteJSON(workerMessage{Type: messageType, Payload: data}); err != nil {
		t.Fatal(err)
	}
}
