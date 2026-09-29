package protocol

import (
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"time"
)

const ProtocolVersion = "phase-0-v3"

// WorkerRegisterRequest is sent by a worker to the server.
type WorkerRegisterRequest struct {
	ProtocolVersion string `json:"protocol_version"`
	WorkerID        string `json:"worker_id"`
	Platform        string `json:"platform"` // android, ios
	OSVersion       string `json:"os_version"`
	DeviceID        string `json:"device_id"`
	DevicePublicKey string `json:"device_public_key"`

	Memory       MemoryInfo        `json:"memory"`
	Compute      ComputeInfo       `json:"compute"`
	Inference    InferenceInfo     `json:"inference"`
	Network      NetworkInfo       `json:"network"`
	Power        PowerInfo         `json:"power"`
	LoadedShards []LoadedShardInfo `json:"loaded_shards,omitempty"`
}

// LoadedShardInfo advertises model shards that are already present and usable.
type LoadedShardInfo struct {
	ModelVersion        string `json:"model_version"`
	ModelID             string `json:"model_id"`
	ModelArtifactDigest string `json:"model_artifact_digest"`
	ShardID             string `json:"shard_id"`
	LayerStart          int    `json:"layer_start"`
	LayerEnd            int    `json:"layer_end"`
}

type MemoryInfo struct {
	TotalMB     uint64 `json:"total_mb"`
	AvailableMB uint64 `json:"available_mb"`
}

type ComputeInfo struct {
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

type InferenceInfo struct {
	Runtime         string           `json:"runtime"`          // llama.cpp, ort
	Backend         string           `json:"backend"`          // unknown, vulkan, metal, cpu
	BenchmarkStatus string           `json:"benchmark_status"` // NOT_RUN, RUNNING, COMPLETED, FAILED
	Performance     *PerformanceInfo `json:"performance,omitempty"`
}

type PerformanceInfo struct {
	TokensPerSecond float64 `json:"tokens_per_second"`
}

type NetworkInfo struct {
	Type string `json:"type"` // wifi, cellular, ethernet
}

type PowerInfo struct {
	BatteryPercent uint32 `json:"battery_percent"`
	Charging       bool   `json:"charging"`
}

// WorkerRegisterResponse is sent by the server to acknowledge registration.
type WorkerRegisterResponse struct {
	Accepted                 bool   `json:"accepted"`
	WorkerID                 string `json:"worker_id"`
	WorkerToken              string `json:"worker_token,omitempty"`
	HeartbeatIntervalSeconds uint32 `json:"heartbeat_interval_seconds"`
	State                    string `json:"state"` // REGISTERED, UNBENCHMARKED, etc.
	Reason                   string `json:"reason,omitempty"`
}

// Heartbeat is sent periodically by a worker.
type Heartbeat struct {
	WorkerID        string    `json:"worker_id"`
	State           string    `json:"state"`
	BatteryPercent  uint32    `json:"battery_percent"`
	Charging        bool      `json:"charging"`
	TemperatureC    float32   `json:"temperature_c,omitempty"`
	Utilization     float32   `json:"utilization,omitempty"`
	TokensPerSecond float64   `json:"tokens_per_second,omitempty"`
	Timestamp       time.Time `json:"timestamp"`
}

// HeartbeatResponse acknowledges a heartbeat.
type HeartbeatResponse struct {
	WorkerID string `json:"worker_id"`
	Ack      bool   `json:"ack"`
}

type JobAssignment struct {
	RequestID           string               `json:"request_id"`
	ModelVersion        string               `json:"model_version"`
	ModelArtifactDigest string               `json:"model_artifact_digest"`
	WorkerID            string               `json:"worker_id"`
	PreviousWorker      string               `json:"previous_worker,omitempty"`
	NextWorker          string               `json:"next_worker,omitempty"`
	AssignmentID        string               `json:"assignment_id"`
	JobID               string               `json:"job_id"`
	ModelID             string               `json:"model_id"`
	ShardID             string               `json:"shard_id"`
	Phase               InferencePhase       `json:"phase,omitempty"`
	SequenceID          string               `json:"sequence_id,omitempty"`
	Position            uint32               `json:"position,omitempty"`
	PassOrdinal         uint64               `json:"pass_ordinal"`
	KVTokenOffset       uint32               `json:"kv_token_offset"`
	TokenCount          uint32               `json:"token_count"`
	InputTensor         *TensorDescriptor    `json:"input_tensor,omitempty"`
	OutputTensor        *TensorDescriptor    `json:"output_tensor,omitempty"`
	Prompt              string               `json:"prompt,omitempty"`
	InputTokenIDs       []uint32             `json:"input_token_ids,omitempty"`
	Activation          *activation.Envelope `json:"activation,omitempty"`
	FinalShard          bool                 `json:"final_shard,omitempty"`
	LayerStart          int                  `json:"layer_start"`
	LayerEnd            int                  `json:"layer_end"`
	Sequence            uint64               `json:"sequence"`
	Payload             []byte               `json:"payload"`
}

// TensorDescriptor is transport metadata only. execution.BuildRequest turns
// it into the backend-independent tensor contract before native execution.
type TensorDescriptor struct {
	DType      string  `json:"dtype"`
	Shape      []int64 `json:"shape"`
	Layout     string  `json:"layout"`
	ByteOrder  string  `json:"byte_order"`
	ByteLength uint64  `json:"byte_length"`
}

type JobAccepted struct {
	AssignmentID string `json:"assignment_id"`
	Accepted     bool   `json:"accepted"`
	Reason       string `json:"reason,omitempty"`
}

type JobComplete struct {
	AssignmentID        string               `json:"assignment_id"`
	JobID               string               `json:"job_id"`
	WorkerID            string               `json:"worker_id"`
	Status              string               `json:"status"`
	Output              []byte               `json:"output"`
	SequenceID          string               `json:"sequence_id,omitempty"`
	Position            uint32               `json:"position,omitempty"`
	PassOrdinal         uint64               `json:"pass_ordinal"`
	KVTokenOffsetBefore uint32               `json:"kv_token_offset_before"`
	KVTokenOffsetAfter  uint32               `json:"kv_token_offset_after"`
	Activation          *activation.Envelope `json:"activation,omitempty"`
	SampledTokenID      *uint32              `json:"sampled_token_id,omitempty"`
	EndOfSequence       bool                 `json:"end_of_sequence,omitempty"`
	GeneratedText       []byte               `json:"generated_text,omitempty"`
	TokensPerSec        float64              `json:"tokens_per_second,omitempty"`
	LayerCount          int                  `json:"layer_count,omitempty"`
}

type JobFailed struct {
	AssignmentID string `json:"assignment_id"`
	JobID        string `json:"job_id"`
	WorkerID     string `json:"worker_id"`
	Reason       string `json:"reason"`
	Unavailable  bool   `json:"unavailable,omitempty"`
}

type SequenceEnd struct {
	JobID      string `json:"job_id"`
	SequenceID string `json:"sequence_id"`
	Completed  bool   `json:"completed"`
}

// BenchmarkStartRequest initiates a benchmark on the worker.
type BenchmarkStartRequest struct {
	JobID           string `json:"job_id"`
	PrefillTokens   int    `json:"prefill_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
}

// BenchmarkResult is sent after benchmark completes.
type BenchmarkResult struct {
	JobID           string  `json:"job_id"`
	WorkerID        string  `json:"worker_id"`
	Status          string  `json:"status"` // COMPLETED, FAILED
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`
	ErrorDetails    string  `json:"error_details,omitempty"`
}
