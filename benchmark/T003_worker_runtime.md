# T003: Worker Runtime and Registration Protocol

## Objective

Implement a platform-independent Worker Runtime abstraction that supports registration without mandatory benchmarking. Android and iOS workers should be able to register, report capabilities, and transition through states independently of whether hardware benchmarking has been performed.

## Architecture

```
                  Worker Runtime
                       │
            ┌──────────┴──────────┐
            │                     │
      Android Worker         iOS Worker
            │                     │
            └──────────┬──────────┘
                       │
                  Worker API
                       │
                       ▼
                  Registration
                       │
                       ▼
                    Server
```

## Worker Lifecycle

```
INSTALL
   ↓
GENERATE DEVICE ID
   ↓
GENERATE DEVICE KEY
   ↓
START WORKER
   ↓
REGISTER
   ↓
REGISTERED
   ↓
CAPABILITY DISCOVERY
   ↓
UNBENCHMARKED
   ↓
READY*
```

## Worker States

Distinct states for production eligibility:

```
UNKNOWN          → initial state
REGISTERING      → actively registering with server
REGISTERED       → server acknowledged registration
UNBENCHMARKED    → registered but no performance data
READY            → benchmarked and eligible for inference
BUSY             → processing inference job
THROTTLED        → thermal/battery constraint
OFFLINE          → disconnected from server
DISABLED         → administratively disabled
```

## Registration Payload (Unbenchmarked)

```json
{
  "worker_id": "worker_01",
  "platform": "android",
  "os_version": "unknown",
  "architecture": "arm64-v8a",

  "memory": {
    "total_mb": 8192,
    "available_mb": 5000
  },

  "compute": {
    "cpu": {
      "available": true
    },
    "gpu": {
      "available": false,
      "vendor": null,
      "name": null
    },
    "npu": {
      "available": false
    }
  },

  "inference": {
    "runtime": "llama.cpp",
    "backend": "unknown",
    "benchmark_status": "NOT_RUN"
  },

  "network": {
    "type": "wifi"
  },

  "power": {
    "battery_percent": 87,
    "charging": true
  }
}
```

## Registration Payload (After Benchmark)

Same structure, updated fields:

```json
{
  "inference": {
    "runtime": "llama.cpp",
    "backend": "vulkan",
    "benchmark_status": "COMPLETED",
    "performance": {
      "tokens_per_second": 8.7
    }
  }
}
```

## Server-Side Worker States

```
UNKNOWN
REGISTERING
REGISTERED
UNBENCHMARKED
READY
BUSY
THROTTLED
OFFLINE
DISABLED
```

Transitions:

```
New worker
    ↓
REGISTERED
    ↓
UNBENCHMARKED
    ↓
benchmark
    ↓
READY
```

If device lacks benchmark:

```
REGISTERED → UNBENCHMARKED
```

## Capability Abstraction

Platform-independent Worker model:

```go
type Worker struct {
  ID                string
  Platform          string                    // android, ios, etc.
  Identity          WorkerIdentity
  Hardware          HardwareCapabilities
  Runtime           RuntimeCapabilities
  Models            []ModelCapability
  Benchmark         BenchmarkInfo
  Network           NetworkCapabilities
  Power             PowerState
  Health            HealthStatus
  State             WorkerState
  LastHeartbeat     time.Time
}

type WorkerIdentity struct {
  DeviceID  string
  PublicKey string
}

type HardwareCapabilities struct {
  Architecture string      // arm64, arm64-v8a, x86_64, etc.
  RAMMB        uint64
  CPUCores     uint32
}

type RuntimeCapabilities struct {
  Name       string      // llama.cpp
  Version    string      // x.y.z
  Backends   []string    // cpu, vulkan, metal, ort, etc.
}

type ModelCapability struct {
  ModelID  string
  ShardIDs []string
}

type BenchmarkInfo struct {
  Status             string      // NOT_RUN, RUNNING, COMPLETED, FAILED
  TokensPerSecond    float64     // only if COMPLETED
  LastRunTime        time.Time
}

type NetworkCapabilities struct {
  Type string  // wifi, cellular, ethernet
}

type PowerState struct {
  BatteryPercent uint32
  Charging       bool
  TemperatureC   float32
}

type HealthStatus struct {
  Healthy            bool
  LastHeartbeatTime  time.Time
  MissedHeartbeats   uint32
}
```

## No Mandatory Benchmarking

Key principle: a worker can be registered and operated in the system without an existing benchmark.

For local development:

```
Worker can pass --benchmark-override=8.7 to bypass the requirement.
```

For production:

```
Scheduler policy:  UNBENCHMARKED workers are ineligible for standard inference jobs.
Scheduler option:  explicit override for testing/development.
```

## Deliverables

- ✅ Platform-independent Worker abstraction
- ✅ Worker state machine (UNKNOWN → REGISTERED → UNBENCHMARKED → READY)
- ✅ Capability discovery model (hardware, runtime, network, power)
- ✅ Registration protocol (works with or without benchmark)
- ✅ Server-side worker registry with state tracking
- ✅ Android stub implementation (ready for T013)
- ✅ iOS stub implementation (ready for future iOS support)
- ✅ WebSocket connection handler for persistent worker registration
- ✅ Heartbeat lifecycle management

## Next Steps (T004)

T004: Model Metadata Parser

- Extract minimum metadata from GGUF
- Parse layer_count, architecture, parameter_count, context_length
- Construct ModelMetadata struct
- No runtime loading yet; metadata only
