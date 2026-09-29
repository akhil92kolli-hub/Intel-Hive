package transport

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/scheduler"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/protocol"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/registry"
	"github.com/gorilla/websocket"
)

func TestInferenceDispatchesAcrossRegisteredWorkers(t *testing.T) {
	executionScheduler := scheduler.New(nil)
	workerSocket := NewWorkerSocketHandler(registry.NewWorkerRegistry(), nil)
	workerSocket.SetScheduler(executionScheduler)
	mux := http.NewServeMux()
	mux.Handle("/worker", workerSocket)
	mux.Handle("/inference", &InferenceHandler{Scheduler: executionScheduler, Workers: workerSocket})
	server := httptest.NewServer(mux)
	defer server.Close()

	shards := []protocol.LoadedShardInfo{
		{ModelVersion: "v1", ModelID: "qwen2.5-3b-instruct", ShardID: "s0", LayerStart: 0, LayerEnd: 11},
		{ModelVersion: "v1", ModelID: "qwen2.5-3b-instruct", ShardID: "s1", LayerStart: 12, LayerEnd: 23},
		{ModelVersion: "v1", ModelID: "qwen2.5-3b-instruct", ShardID: "s2", LayerStart: 24, LayerEnd: 35},
	}
	connections := make([]*websocket.Conn, 0, len(shards))
	assignments := make(chan protocol.JobAssignment, 9)
	for i, shard := range shards {
		conn := dialWorker(t, server.URL)
		connections = append(connections, conn)
		workerID := fmt.Sprintf("worker-%d", i)
		writeWorkerMessage(t, conn, "register", protocol.WorkerRegisterRequest{
			ProtocolVersion: protocol.ProtocolVersion,
			WorkerID:        workerID,
			Platform:        "android",
			Memory:          protocol.MemoryInfo{TotalMB: 8192, AvailableMB: 4096},
			Inference: protocol.InferenceInfo{
				Runtime: "llama.cpp", Backend: "cpu", BenchmarkStatus: "COMPLETED",
				Performance: &protocol.PerformanceInfo{TokensPerSecond: 2},
			},
			LoadedShards: []protocol.LoadedShardInfo{shard},
		})
		var ack workerMessage
		if err := conn.ReadJSON(&ack); err != nil {
			t.Fatal(err)
		}
		if ack.Type != "register_ack" {
			t.Fatalf("expected registration acknowledgment, got %s", ack.Type)
		}
		go respondToAssignments(conn, assignments)
	}
	defer func() {
		for _, conn := range connections {
			_ = conn.Close()
		}
	}()

	requestBody := bytes.NewBufferString(`{"model_id":"qwen2.5-3b-instruct","prompt":"hello","job_id":"test-job"}`)
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Post(server.URL+"/inference", "application/json", requestBody)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("expected inference success, got %d", response.StatusCode)
	}
	var result inferenceResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.JobID != "test-job" || result.ModelID != "qwen2.5-3b-instruct" ||
		result.Output != "AB" || len(result.GeneratedTokenIDs) != 2 ||
		result.GeneratedTokenIDs[0] != 101 || result.GeneratedTokenIDs[1] != 102 {
		t.Fatalf("unexpected distributed inference result: %+v", result)
	}
	close(assignments)
	prefillCount, decodeCount := 0, 0
	for assignment := range assignments {
		if assignment.SequenceID != "test-job:0" {
			t.Fatalf("unexpected sequence ID in assignment: %+v", assignment)
		}
		switch assignment.Phase {
		case protocol.InferencePhasePrefill:
			prefillCount++
			if assignment.Position != 0 {
				t.Fatalf("prefill position should be zero: %+v", assignment)
			}
			switch assignment.ShardID {
			case "s0":
				if assignment.Prompt != "hello" || assignment.Activation != nil {
					t.Fatalf("first prefill shard should receive prompt: %+v", assignment)
				}
			default:
				if assignment.Prompt != "" || assignment.Activation == nil {
					t.Fatalf("downstream prefill shard should receive activation: %+v", assignment)
				}
			}
		case protocol.InferencePhaseDecode:
			decodeCount++
			if assignment.Position == 0 || assignment.Prompt != "" {
				t.Fatalf("invalid decode step: %+v", assignment)
			}
		default:
			t.Fatalf("assignment missing inference phase: %+v", assignment)
		}
	}
	if prefillCount != 3 || decodeCount != 6 {
		t.Fatalf("expected three prefill and six decode assignments, got prefill=%d decode=%d", prefillCount, decodeCount)
	}
}

func TestRemoteWorkerClassifiesLateCompletionAsStale(t *testing.T) {
	remote := NewRemoteWorkerWithWriteLock("worker", worker.Capabilities{}, nil, nil)
	remote.pending["assignment"] = pendingRemoteJob{
		jobID: "job", result: make(chan remoteJobResult, 1),
	}
	remote.removePending("assignment")
	err := remote.Complete(protocol.JobComplete{
		AssignmentID: "assignment", JobID: "job", WorkerID: "worker", Status: "completed",
	})
	if !errors.Is(err, ErrStaleAssignment) {
		t.Fatalf("expected stale-assignment error, got %v", err)
	}
}

func TestInferenceRejectsOutOfRangeMaxTokens(t *testing.T) {
	handler := &InferenceHandler{
		Scheduler: scheduler.New(nil),
		Workers:   NewWorkerSocketHandler(registry.NewWorkerRegistry(), nil),
	}
	for _, body := range []string{
		`{"model_id":"qwen2.5-3b-instruct","prompt":"hello","max_tokens":0}`,
		`{"model_id":"qwen2.5-3b-instruct","prompt":"hello","max_tokens":257}`,
	} {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/inference", bytes.NewBufferString(body)))
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for max_tokens in %s, got %d", body, recorder.Code)
		}
	}
}

func dialWorker(t *testing.T, serverURL string) *websocket.Conn {
	t.Helper()
	endpoint, err := url.Parse(serverURL)
	if err != nil {
		t.Fatal(err)
	}
	endpoint.Scheme = "ws"
	endpoint.Path = "/worker"
	conn, _, err := websocket.DefaultDialer.Dial(endpoint.String(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return conn
}

func respondToAssignments(conn *websocket.Conn, assignments chan<- protocol.JobAssignment) {
	for {
		var message workerMessage
		if err := conn.ReadJSON(&message); err != nil {
			return
		}
		if message.Type != "job_assignment" {
			continue
		}
		var assignment protocol.JobAssignment
		if err := json.Unmarshal(message.Payload, &assignment); err != nil {
			return
		}
		assignments <- assignment
		_ = conn.WriteJSON(workerMessageForTest("job_accepted", protocol.JobAccepted{
			AssignmentID: assignment.AssignmentID, Accepted: true,
		}))
		completion := protocol.JobComplete{
			AssignmentID: assignment.AssignmentID, JobID: assignment.JobID, WorkerID: workerIDFromShard(assignment.ShardID),
			Status: "completed", SequenceID: assignment.SequenceID, Position: assignment.Position,
		}
		if assignment.FinalShard {
			if assignment.Phase == protocol.InferencePhasePrefill || assignment.Position == 1 {
				token := uint32(100 + assignment.Position + 1)
				completion.SampledTokenID = &token
				completion.GeneratedText = []byte(string(rune('A' + assignment.Position)))
			} else {
				completion.EndOfSequence = true
			}
		} else {
			a := activation.NewEnvelope(assignment.JobID, assignment.RequestID, assignment.SequenceID, assignment.ModelID, assignment.ModelVersion, completion.WorkerID, assignment.NextWorker, assignment.LayerEnd, "uint8", []int64{1}, []byte{1})
			a.Position = assignment.Position
			completion.Activation = &a
		}
		_ = conn.WriteJSON(workerMessageForTest("job_complete", completion))
	}
}

func workerIDFromShard(shardID string) string {
	switch shardID {
	case "s0":
		return "worker-0"
	case "s1":
		return "worker-1"
	default:
		return "worker-2"
	}
}

func workerMessageForTest(messageType string, payload any) workerMessage {
	data, _ := json.Marshal(payload)
	return workerMessage{Type: messageType, Payload: data}
}
