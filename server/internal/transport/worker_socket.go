package transport

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/scheduler"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/protocol"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/registry"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

const workerHeartbeatInterval = 15
const maxWorkerMessageBytes = 64 << 20

type workerMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

type workerReply struct {
	Type    string `json:"type"`
	Payload any    `json:"payload"`
}

// WorkerSocketHandler accepts worker lifecycle and job-result messages over
// a WebSocket. Workers remain UNBENCHMARKED until a benchmark is completed.
type WorkerSocketHandler struct {
	Registry  *registry.WorkerRegistry
	Logger    *log.Logger
	Scheduler *scheduler.Scheduler
	upgrader  websocket.Upgrader
	mu        sync.RWMutex
	sessions  map[string]*RemoteWorker
	models    map[string]map[string]model.ModelShard
}

func NewWorkerSocketHandler(workers *registry.WorkerRegistry, logger *log.Logger) *WorkerSocketHandler {
	if workers == nil {
		workers = registry.NewWorkerRegistry()
	}
	return &WorkerSocketHandler{
		Registry: workers,
		Logger:   logger,
		sessions: make(map[string]*RemoteWorker),
		models:   make(map[string]map[string]model.ModelShard),
		upgrader: websocket.Upgrader{ReadBufferSize: 4096, WriteBufferSize: 4096},
	}
}

func (h *WorkerSocketHandler) SetScheduler(s *scheduler.Scheduler) {
	h.Scheduler = s
}

func (h *WorkerSocketHandler) ModelShards(modelID string) ([]model.ModelShard, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	byID, ok := h.models[modelID]
	if !ok {
		return nil, false
	}
	shards := make([]model.ModelShard, 0, len(byID))
	for _, shard := range byID {
		shards = append(shards, shard)
	}
	sort.Slice(shards, func(i, j int) bool {
		return shards[i].Layers.Start < shards[j].Layers.Start
	})
	return shards, len(shards) > 0
}

func (h *WorkerSocketHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()
	conn.SetReadLimit(maxWorkerMessageBytes)
	var connWriteMu sync.Mutex

	workerID := ""
	var remote *RemoteWorker
	sessionOwned := false
	defer func() {
		if remote != nil {
			remote.Close()
		}
		if workerID != "" && sessionOwned {
			h.mu.Lock()
			isCurrent := h.sessions[workerID] == remote
			if isCurrent {
				delete(h.sessions, workerID)
			}
			h.mu.Unlock()
			if isCurrent {
				h.Registry.MarkOffline(workerID)
				if h.Scheduler != nil {
					_ = h.Scheduler.MarkOffline(workerID)
				}
			}
		}
	}()

	for {
		var message workerMessage
		if err := conn.ReadJSON(&message); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) && h.Logger != nil {
				h.Logger.Printf("worker socket read failed: %v", err)
			}
			return
		}

		switch message.Type {
		case "register":
			if workerID != "" {
				h.writeError(conn, &connWriteMu, "worker already registered on this connection")
				return
			}
			var request protocol.WorkerRegisterRequest
			if err := json.Unmarshal(message.Payload, &request); err != nil {
				h.writeError(conn, &connWriteMu, "invalid registration payload")
				return
			}
			if err := validateRegistration(request); err != nil {
				_ = h.writeJSON(conn, &connWriteMu, workerReply{Type: "register_ack", Payload: protocol.WorkerRegisterResponse{
					Accepted: false, WorkerID: request.WorkerID, Reason: err.Error(),
				}})
				return
			}
			if err := h.validateShardCatalog(request); err != nil {
				_ = h.writeJSON(conn, &connWriteMu, workerReply{Type: "register_ack", Payload: protocol.WorkerRegisterResponse{
					Accepted: false, WorkerID: request.WorkerID, Reason: err.Error(),
				}})
				return
			}
			workerID = request.WorkerID
			worker := workerFromRequest(request)
			h.Registry.RegisterOrUpdate(worker)
			capabilities := workerCapabilitiesFromRegistration(request)
			remote = NewRemoteWorkerWithWriteLock(workerID, capabilities, conn, &connWriteMu)
			if h.Scheduler != nil && len(request.LoadedShards) > 0 {
				record := scheduler.WorkerRecord{
					ID: workerID, State: schedulerState(capabilities.State),
					Platform: scheduler.WorkerPlatform(request.Platform),
					Layers:   capabilities.Layers, Benchmarks: capabilities.Benchmarks,
					MemoryMB:             capabilities.MemoryMB,
					ModelShards:          make(map[string]map[string]model.LayerRange),
					ModelArtifactDigests: make(map[string]map[string]string),
				}
				for _, shard := range request.LoadedShards {
					if record.ModelShards[shard.ModelID] == nil {
						record.ModelShards[shard.ModelID] = make(map[string]model.LayerRange)
						record.ModelArtifactDigests[shard.ModelID] = make(map[string]string)
					}
					record.ModelShards[shard.ModelID][shard.ShardID] = model.LayerRange{
						Start: shard.LayerStart, End: shard.LayerEnd,
					}
					record.ModelArtifactDigests[shard.ModelID][shard.ShardID] = shard.ModelArtifactDigest
				}
				if err := h.Scheduler.RegisterWorker(record, remote); err != nil {
					remote.Close()
					remote = nil
					h.writeError(conn, &connWriteMu, err.Error())
					return
				}
			} else if h.Scheduler != nil {
				_ = h.Scheduler.UnregisterWorker(workerID)
			}
			h.mu.Lock()
			previous := h.sessions[workerID]
			h.sessions[workerID] = remote
			if len(request.LoadedShards) > 0 {
				for _, shard := range request.LoadedShards {
					if h.models[shard.ModelID] == nil {
						h.models[shard.ModelID] = make(map[string]model.ModelShard)
					}
					h.models[shard.ModelID][shard.ShardID] = model.ModelShard{
						ID: shard.ShardID, ModelID: shard.ModelID, Version: shard.ModelVersion, ArtifactDigest: shard.ModelArtifactDigest,
						Layers: model.LayerRange{Start: shard.LayerStart, End: shard.LayerEnd},
					}
				}
			}
			h.mu.Unlock()
			sessionOwned = true
			if previous != nil && previous != remote {
				previous.Close()
			}
			if err := h.writeJSON(conn, &connWriteMu, workerReply{Type: "register_ack", Payload: protocol.WorkerRegisterResponse{
				Accepted: true, WorkerID: workerID,
				HeartbeatIntervalSeconds: workerHeartbeatInterval,
				State:                    string(worker.State),
			}}); err != nil {
				return
			}
			if remote != nil {
				remote.Activate()
			}

		case "heartbeat":
			if workerID == "" {
				h.writeError(conn, &connWriteMu, "register before sending heartbeats")
				return
			}
			var heartbeat protocol.Heartbeat
			if err := json.Unmarshal(message.Payload, &heartbeat); err != nil {
				h.writeError(conn, &connWriteMu, "invalid heartbeat payload")
				return
			}
			if heartbeat.WorkerID != workerID {
				h.writeError(conn, &connWriteMu, "heartbeat worker_id does not match registration")
				return
			}
			if heartbeat.BatteryPercent > 100 {
				h.writeError(conn, &connWriteMu, "heartbeat battery_percent must be between 0 and 100")
				return
			}
			if !h.Registry.UpdateHeartbeatWithPower(workerID, registry.PowerState{
				BatteryPercent: heartbeat.BatteryPercent,
				Charging:       heartbeat.Charging,
				TemperatureC:   heartbeat.TemperatureC,
			}) {
				h.writeError(conn, &connWriteMu, "worker registration no longer exists")
				return
			}
			if err := h.writeJSON(conn, &connWriteMu, workerReply{Type: "heartbeat_ack", Payload: protocol.HeartbeatResponse{
				WorkerID: workerID, Ack: true,
			}}); err != nil {
				return
			}

		case "job_accepted", "job_complete", "job_failed":
			if workerID == "" || remote == nil {
				h.writeError(conn, &connWriteMu, "register before sending job results")
				return
			}
			var err error
			switch message.Type {
			case "job_accepted":
				var accepted protocol.JobAccepted
				if err = json.Unmarshal(message.Payload, &accepted); err == nil {
					err = remote.Accept(accepted)
				}
			case "job_complete":
				var completed protocol.JobComplete
				if err = json.Unmarshal(message.Payload, &completed); err == nil {
					err = remote.Complete(completed)
				}
			case "job_failed":
				var failed protocol.JobFailed
				if err = json.Unmarshal(message.Payload, &failed); err == nil {
					err = remote.Fail(failed)
					if err == nil && failed.Unavailable {
						h.Registry.MarkOffline(workerID)
						if h.Scheduler != nil {
							_ = h.Scheduler.MarkOffline(workerID)
						}
						remote.Close()
					}
				}
			}
			if err != nil {
				if errors.Is(err, ErrStaleAssignment) {
					if h.Logger != nil {
						h.Logger.Printf("ignored late worker result from %s: %v", workerID, err)
					}
					continue
				}
				h.writeError(conn, &connWriteMu, err.Error())
				return
			}

		default:
			h.writeError(conn, &connWriteMu, fmt.Sprintf("unsupported worker message type %q", message.Type))
			return
		}
	}
}

func (h *WorkerSocketHandler) writeJSON(conn *websocket.Conn, mu *sync.Mutex, message workerReply) error {
	mu.Lock()
	defer mu.Unlock()
	return conn.WriteJSON(message)
}

func (h *WorkerSocketHandler) writeError(conn *websocket.Conn, mu *sync.Mutex, message string) {
	if err := h.writeJSON(conn, mu, workerReply{Type: "error", Payload: map[string]string{"message": message}}); err != nil && h.Logger != nil {
		h.Logger.Printf("failed to write worker error: %v", err)
	}
}

func schedulerState(state worker.State) scheduler.WorkerState {
	if state == worker.StateReady {
		return scheduler.WorkerStateReady
	}
	return scheduler.WorkerStateUnbenchmarked
}

func (h *WorkerSocketHandler) validateShardCatalog(request protocol.WorkerRegisterRequest) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	incoming := make(map[string]model.LayerRange)
	versions := make(map[string]string)
	digests := make(map[string]string)
	for _, shard := range request.LoadedShards {
		if strings.TrimSpace(shard.ModelVersion) == "" {
			return fmt.Errorf("loaded shard model_version is required")
		}
		if version, ok := versions[shard.ModelID]; ok && version != shard.ModelVersion {
			return fmt.Errorf("conflicting model versions")
		}
		versions[shard.ModelID] = shard.ModelVersion
		if digest, ok := digests[shard.ModelID]; ok && digest != shard.ModelArtifactDigest {
			return fmt.Errorf("conflicting model artifact digests")
		}
		digests[shard.ModelID] = shard.ModelArtifactDigest
		for _, known := range h.models[shard.ModelID] {
			if known.Version != shard.ModelVersion {
				return fmt.Errorf("model version conflicts with registered catalog")
			}
			if known.ArtifactDigest != shard.ModelArtifactDigest {
				return fmt.Errorf("model artifact digest conflicts with registered catalog")
			}
		}
		rangeForShard := model.LayerRange{Start: shard.LayerStart, End: shard.LayerEnd}
		key := shard.ModelID + "\x00" + shard.ShardID
		if previous, exists := incoming[key]; exists && previous != rangeForShard {
			return fmt.Errorf("shard %s has conflicting layer ranges in registration", shard.ShardID)
		}
		incoming[key] = rangeForShard
		if registered, exists := h.models[shard.ModelID][shard.ShardID]; exists &&
			(registered.Layers != rangeForShard || registered.Version != shard.ModelVersion ||
				registered.ArtifactDigest != shard.ModelArtifactDigest) {
			return fmt.Errorf("shard %s conflicts with its registered layer range", shard.ShardID)
		}
	}
	return nil
}

func validateRegistration(request protocol.WorkerRegisterRequest) error {
	if request.ProtocolVersion != protocol.ProtocolVersion {
		return fmt.Errorf("unsupported protocol version")
	}
	if strings.TrimSpace(request.WorkerID) == "" {
		return fmt.Errorf("worker_id is required")
	}
	if request.Platform != "android" && request.Platform != "ios" {
		return fmt.Errorf("platform must be android or ios")
	}
	if request.Power.BatteryPercent > 100 {
		return fmt.Errorf("battery_percent must be between 0 and 100")
	}
	if strings.EqualFold(request.Inference.BenchmarkStatus, "COMPLETED") {
		if request.Inference.Performance == nil || request.Inference.Performance.TokensPerSecond <= 0 {
			return fmt.Errorf("completed benchmark requires positive performance")
		}
		switch request.Inference.Performance.ExecutionMode {
		case protocol.BenchmarkExecutionModeSingleDeviceAllShards,
			protocol.BenchmarkExecutionModeDistributedPipeline:
		default:
			return fmt.Errorf("completed benchmark requires a supported execution_mode")
		}
	}
	for _, shard := range request.LoadedShards {
		if shard.ModelID == "" || shard.ShardID == "" ||
			!protocol.ValidSHA256Digest(shard.ModelArtifactDigest) ||
			shard.LayerStart < 0 || shard.LayerEnd < shard.LayerStart {
			return fmt.Errorf("loaded_shards contains an invalid shard")
		}
	}
	return nil
}

func workerFromRequest(request protocol.WorkerRegisterRequest) *registry.Worker {
	modelsByID := make(map[string][]string)
	for _, shard := range request.LoadedShards {
		modelsByID[shard.ModelID] = append(modelsByID[shard.ModelID], shard.ShardID)
	}
	models := make([]registry.ModelCapability, 0, len(modelsByID))
	for modelID, shardIDs := range modelsByID {
		models = append(models, registry.ModelCapability{ModelID: modelID, ShardIDs: shardIDs})
	}

	state := registry.StateUnbenchmarked
	benchmarkStatus := request.Inference.BenchmarkStatus
	if strings.EqualFold(benchmarkStatus, "COMPLETED") && request.Inference.Performance != nil {
		state = registry.StateReady
	}

	worker := &registry.Worker{
		ID:       request.WorkerID,
		Platform: request.Platform,
		State:    state,
		Identity: registry.WorkerIdentity{DeviceID: request.DeviceID, PublicKey: request.DevicePublicKey},
		Hardware: registry.HardwareCapabilities{RAMMB: request.Memory.TotalMB},
		Runtime: registry.RuntimeCapabilities{
			Name: request.Inference.Runtime, Backends: []string{request.Inference.Backend},
		},
		Inference: registry.InferenceCapabilities{
			Runtime: request.Inference.Runtime, Backend: request.Inference.Backend,
			BenchmarkStatus: benchmarkStatus,
		},
		Models:  models,
		Network: registry.NetworkCapabilities{Type: request.Network.Type},
		Power: registry.PowerState{
			BatteryPercent: request.Power.BatteryPercent, Charging: request.Power.Charging,
		},
		Health: registry.HealthStatus{Healthy: true, LastHeartbeatTime: time.Now().UTC()},
	}
	if request.Compute.GPU.Available {
		worker.Compute.GPU.Available = true
		worker.Compute.GPU.Name = request.Compute.GPU.Name
		worker.Compute.GPU.Vendor = request.Compute.GPU.Vendor
	}
	worker.Compute.CPU.Available = request.Compute.CPU.Available
	worker.Compute.NPU.Available = request.Compute.NPU.Available
	if request.Inference.Performance != nil {
		worker.Inference.Performance = &registry.BenchmarkInfo{
			Status: benchmarkStatus, TokensPerSecond: request.Inference.Performance.TokensPerSecond,
			ExecutionMode: request.Inference.Performance.ExecutionMode,
		}
	}
	return worker
}

// NewWorkerID creates a stable-format opaque ID for clients that need one.
func NewWorkerID() string { return uuid.NewString() }
