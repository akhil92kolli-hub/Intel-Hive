package registry

import (
	"sync"
	"time"
)

// WorkerState represents the lifecycle state of a worker.
type WorkerState string

const (
	StateUnknown       WorkerState = "UNKNOWN"
	StateRegistering   WorkerState = "REGISTERING"
	StateRegistered    WorkerState = "REGISTERED"
	StateUnbenchmarked WorkerState = "UNBENCHMARKED"
	StateReady         WorkerState = "READY"
	StateBusy          WorkerState = "BUSY"
	StateThrottled     WorkerState = "THROTTLED"
	StateOffline       WorkerState = "OFFLINE"
	StateDisabled      WorkerState = "DISABLED"
)

// WorkerIdentity represents secure identification.
type WorkerIdentity struct {
	DeviceID  string `json:"device_id"`
	PublicKey string `json:"public_key"`
}

// HardwareCapabilities describes the device hardware.
type HardwareCapabilities struct {
	Architecture string `json:"architecture"` // arm64-v8a, arm64, x86_64, etc.
	RAMMB        uint64 `json:"ram_mb"`
	CPUCores     uint32 `json:"cpu_cores,omitempty"`
}

// ComputeCapability describes available compute units.
type ComputeCapability struct {
	CPU struct {
		Available bool `json:"available"`
	} `json:"cpu"`
	GPU struct {
		Available bool   `json:"available"`
		Vendor    string `json:"vendor,omitempty"`
		Name      string `json:"name,omitempty"`
	} `json:"gpu"`
	NPU struct {
		Available bool `json:"available"`
	} `json:"npu"`
}

// RuntimeCapabilities describes inference runtime.
type RuntimeCapabilities struct {
	Name     string   `json:"name"` // llama.cpp, ort, etc.
	Version  string   `json:"version"`
	Backends []string `json:"backends"` // cpu, vulkan, metal, etc.
}

// BenchmarkInfo describes performance data.
type BenchmarkInfo struct {
	Status          string    `json:"status"` // NOT_RUN, RUNNING, COMPLETED, FAILED
	TokensPerSecond float64   `json:"tokens_per_second,omitempty"`
	LastRunTime     time.Time `json:"last_run_time,omitempty"`
}

// InferenceCapabilities describes model runtime support.
type InferenceCapabilities struct {
	Runtime         string         `json:"runtime"`
	Backend         string         `json:"backend"`
	BenchmarkStatus string         `json:"benchmark_status"`
	Performance     *BenchmarkInfo `json:"performance,omitempty"`
}

// ModelCapability describes which model shards a worker can hold.
type ModelCapability struct {
	ModelID  string   `json:"model_id"`
	ShardIDs []string `json:"shard_ids"`
}

// NetworkCapabilities describes network interface.
type NetworkCapabilities struct {
	Type string `json:"type"` // wifi, cellular, ethernet
}

// PowerState describes device power.
type PowerState struct {
	BatteryPercent uint32  `json:"battery_percent"`
	Charging       bool    `json:"charging"`
	TemperatureC   float32 `json:"temperature_c,omitempty"`
}

// HealthStatus describes worker connectivity and health.
type HealthStatus struct {
	Healthy           bool      `json:"healthy"`
	LastHeartbeatTime time.Time `json:"last_heartbeat_time"`
	MissedHeartbeats  uint32    `json:"missed_heartbeats"`
}

// Worker represents a platform-independent worker abstraction.
type Worker struct {
	ID       string      `json:"worker_id"`
	Platform string      `json:"platform"` // android, ios
	State    WorkerState `json:"state"`

	Identity  WorkerIdentity        `json:"identity"`
	Hardware  HardwareCapabilities  `json:"hardware"`
	Compute   ComputeCapability     `json:"compute"`
	Runtime   RuntimeCapabilities   `json:"runtime"`
	Inference InferenceCapabilities `json:"inference"`
	Models    []ModelCapability     `json:"models"`
	Network   NetworkCapabilities   `json:"network"`
	Power     PowerState            `json:"power"`
	Health    HealthStatus          `json:"health"`

	RegisteredAt time.Time `json:"registered_at"`
}

// WorkerRegistry manages all workers and their state.
type WorkerRegistry struct {
	mu      sync.RWMutex
	workers map[string]*Worker
}

// NewWorkerRegistry creates a new worker registry.
func NewWorkerRegistry() *WorkerRegistry {
	return &WorkerRegistry{
		workers: make(map[string]*Worker),
	}
}

// RegisterOrUpdate registers a new worker or updates an existing one.
func (r *WorkerRegistry) RegisterOrUpdate(w *Worker) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if existing, ok := r.workers[w.ID]; ok {
		// Update existing
		existing.State = w.State
		existing.Identity = w.Identity
		existing.Hardware = w.Hardware
		existing.Compute = w.Compute
		existing.Runtime = w.Runtime
		existing.Inference = w.Inference
		existing.Models = w.Models
		existing.Power = w.Power
		existing.Network = w.Network
		existing.Health = w.Health
		if w.Inference.BenchmarkStatus == "COMPLETED" {
			existing.State = StateReady
		}
	} else {
		// Register new
		w.RegisteredAt = time.Now().UTC()
		if w.State == "" {
			w.State = StateRegistered
		}
		if w.Inference.BenchmarkStatus == "" {
			w.Inference.BenchmarkStatus = "NOT_RUN"
			w.State = StateUnbenchmarked
		}
		r.workers[w.ID] = w
	}
}

// GetWorker retrieves a worker by ID.
func (r *WorkerRegistry) GetWorker(workerID string) (*Worker, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	w, ok := r.workers[workerID]
	return w, ok
}

// ListWorkers returns all registered workers.
func (r *WorkerRegistry) ListWorkers() []*Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Worker, 0, len(r.workers))
	for _, w := range r.workers {
		out = append(out, w)
	}
	return out
}

// AvailableWorkersForModel returns workers that have a specific model/shard.
func (r *WorkerRegistry) AvailableWorkersForModel(modelID, shardID string) []*Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Worker, 0)
	for _, w := range r.workers {
		if w.State != StateReady {
			continue
		}
		for _, mc := range w.Models {
			if mc.ModelID == modelID {
				for _, sid := range mc.ShardIDs {
					if sid == shardID {
						out = append(out, w)
						break
					}
				}
			}
		}
	}
	return out
}

// UpdateHeartbeat marks a worker as recently seen.
func (r *WorkerRegistry) UpdateHeartbeat(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[workerID]; ok {
		w.Health.LastHeartbeatTime = time.Now().UTC()
		w.Health.MissedHeartbeats = 0
		w.Health.Healthy = true
	}
}

// UpdateHeartbeatWithPower refreshes liveness and reported power data atomically.
func (r *WorkerRegistry) UpdateHeartbeatWithPower(workerID string, power PowerState) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	w, ok := r.workers[workerID]
	if !ok {
		return false
	}
	w.Power = power
	w.Health.LastHeartbeatTime = time.Now().UTC()
	w.Health.MissedHeartbeats = 0
	w.Health.Healthy = true
	return true
}

// MarkOffline marks a worker as offline after missed heartbeats.
func (r *WorkerRegistry) MarkOffline(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[workerID]; ok {
		w.State = StateOffline
		w.Health.Healthy = false
	}
}

// TransitionState moves a worker to a new state.
func (r *WorkerRegistry) TransitionState(workerID string, newState WorkerState) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[workerID]; ok {
		w.State = newState
	}
}
