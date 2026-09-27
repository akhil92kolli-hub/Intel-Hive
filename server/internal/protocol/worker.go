package protocol

import "time"

const ProtocolVersion = "phase-0-v1"

// WorkerRegisterRequest is sent by a worker to the server.
type WorkerRegisterRequest struct {
    ProtocolVersion string                     `json:"protocol_version"`
    WorkerID        string                     `json:"worker_id"`
    Platform        string                     `json:"platform"`    // android, ios
    OSVersion       string                     `json:"os_version"`
    DeviceID        string                     `json:"device_id"`
    DevicePublicKey string                     `json:"device_public_key"`

    Memory   MemoryInfo   `json:"memory"`
    Compute  ComputeInfo  `json:"compute"`
    Inference InferenceInfo `json:"inference"`
    Network  NetworkInfo  `json:"network"`
    Power    PowerInfo    `json:"power"`
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
    Runtime        string `json:"runtime"`         // llama.cpp, ort
    Backend        string `json:"backend"`         // unknown, vulkan, metal, cpu
    BenchmarkStatus string `json:"benchmark_status"`  // NOT_RUN, RUNNING, COMPLETED, FAILED
    Performance    *PerformanceInfo `json:"performance,omitempty"`
}

type PerformanceInfo struct {
    TokensPerSecond float64 `json:"tokens_per_second"`
}

type NetworkInfo struct {
    Type string `json:"type"`  // wifi, cellular, ethernet
}

type PowerInfo struct {
    BatteryPercent uint32 `json:"battery_percent"`
    Charging       bool   `json:"charging"`
}

// WorkerRegisterResponse is sent by the server to acknowledge registration.
type WorkerRegisterResponse struct {
    Accepted                 bool   `json:"accepted"`
    WorkerID                 string `json:"worker_id"`
    WorkerToken              string `json:"worker_token"`
    HeartbeatIntervalSeconds uint32 `json:"heartbeat_interval_seconds"`
    State                    string `json:"state"`  // REGISTERED, UNBENCHMARKED, etc.
    Reason                   string `json:"reason,omitempty"`
}

// Heartbeat is sent periodically by a worker.
type Heartbeat struct {
    WorkerID       string    `json:"worker_id"`
    State          string    `json:"state"`
    BatteryPercent uint32    `json:"battery_percent"`
    Charging       bool      `json:"charging"`
    TemperatureC   float32   `json:"temperature_c"`
    Utilization    float32   `json:"utilization"`
    TokensPerSecond float64  `json:"tokens_per_second,omitempty"`
    Timestamp      time.Time `json:"timestamp"`
}

// HeartbeatResponse acknowledges a heartbeat.
type HeartbeatResponse struct {
    WorkerID string `json:"worker_id"`
    Ack      bool   `json:"ack"`
}

// BenchmarkStartRequest initiates a benchmark on the worker.
type BenchmarkStartRequest struct {
    JobID               string `json:"job_id"`
    PrefillTokens      int    `json:"prefill_tokens"`
    GeneratedTokens    int    `json:"generated_tokens"`
}

// BenchmarkResult is sent after benchmark completes.
type BenchmarkResult struct {
    JobID              string    `json:"job_id"`
    WorkerID           string    `json:"worker_id"`
    Status             string    `json:"status"`  // COMPLETED, FAILED
    TokensPerSecond    float64   `json:"tokens_per_second,omitempty"`
    ErrorDetails       string    `json:"error_details,omitempty"`
}
