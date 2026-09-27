package mock

import (
    "context"
    "fmt"
    "time"

    "github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
    "github.com/akhil92kolli-hub/Intel-Hive/internal/model"
    "github.com/akhil92kolli-hub/Intel-Hive/internal/transport"
)

// WorkerPlatform defines a generic platform identifier.
type WorkerPlatform string

const (
    PlatformAndroid WorkerPlatform = "android"
    PlatformIOS     WorkerPlatform = "ios"
    PlatformMock    WorkerPlatform = "mock"
)

// WorkerState represents the runtime state of a worker.
type WorkerState string

const (
    StateUnknown       WorkerState = "UNKNOWN"
    StateRegistering   WorkerState = "REGISTERING"
    StateRegistered    WorkerState = "REGISTERED"
    StateUnbenchmarked WorkerState = "UNBENCHMARKED"
    StateReady         WorkerState = "READY"
    StateBusy          WorkerState = "BUSY"
    StateOffline       WorkerState = "OFFLINE"
)

// WorkerCapabilities summarizes mock worker features.
type WorkerCapabilities struct {
    ID          string
    Platform    WorkerPlatform
    State       WorkerState
    Runtime     string
    Backends    []string
    Layers      []model.LayerRange
    Benchmarks  map[string]float64
    MemoryMB    int
}

// Job represents a generic execution request.
type Job struct {
    ID       string
    ModelID  string
    Layers   model.LayerRange
    Payload  []byte
    Sequence uint64
}

// JobResult contains the simulated result from a worker.
type JobResult struct {
    JobID        string
    Status       string
    Output       []byte
    TokensPerSec float64
    LayerCount   int
}

// WorkerRuntime abstracts real and mock workers.
type WorkerRuntime interface {
    Start(ctx context.Context) error
    Stop(ctx context.Context) error
    Register(ctx context.Context) error
    Heartbeat(ctx context.Context) error
    Capabilities(ctx context.Context) WorkerCapabilities
    Execute(ctx context.Context, job Job) (JobResult, error)
}

// MockWorker is an in-memory test worker that simulates distributed execution.
type MockWorker struct {
    ID          string
    Platform    WorkerPlatform
    State       WorkerState
    Transport   *transport.InMemoryTransport
    Capabilities WorkerCapabilities
    Layers      []model.LayerRange
}

func NewMockWorker(id string, platform WorkerPlatform, layers []model.LayerRange) *MockWorker {
    return &MockWorker{
        ID:        id,
        Platform:  platform,
        State:     StateUnbenchmarked,
        Transport: transport.NewInMemoryTransport(),
        Capabilities: WorkerCapabilities{
            ID:          id,
            Platform:    platform,
            State:       StateUnbenchmarked,
            Runtime:     "llama.cpp",
            Backends:    []string{"cpu"},
            Layers:      layers,
            Benchmarks: map[string]float64{"mock": 10.0},
            MemoryMB:    4096,
        },
        Layers: layers,
    }
}

func (w *MockWorker) Start(ctx context.Context) error {
    w.State = StateReady
    return nil
}

func (w *MockWorker) Stop(ctx context.Context) error {
    w.State = StateOffline
    return nil
}

func (w *MockWorker) Register(ctx context.Context) error {
    w.State = StateRegistered
    return nil
}

func (w *MockWorker) Heartbeat(ctx context.Context) error {
    if w.State == StateOffline {
        return fmt.Errorf("worker offline")
    }
    return nil
}

func (w *MockWorker) Capabilities(ctx context.Context) WorkerCapabilities {
    return w.Capabilities
}

func (w *MockWorker) Execute(ctx context.Context, job Job) (JobResult, error) {
    if w.State == StateOffline {
        return JobResult{}, fmt.Errorf("worker offline")
    }

    // Validate shard coverage.
    covered := false
    for _, r := range w.Layers {
        if r.Start == job.Layers.Start && r.End == job.Layers.End {
            covered = true
            break
        }
    }
    if !covered {
        return JobResult{}, fmt.Errorf("worker %s cannot execute layers %d-%d", w.ID, job.Layers.Start, job.Layers.End)
    }

    output := []byte(fmt.Sprintf("mock-output-%s-layer-%d-%d", w.ID, job.Layers.Start, job.Layers.End))
    act := activation.NewActivation(
        job.ID,
        fmt.Sprintf("req-%s", job.ID),
        job.Sequence,
        job.Layers.Start,
        "f16",
        []int64{1, 128, 3072},
        output,
    )

    packet := transport.ActivationPacket{
        JobID:             job.ID,
        RequestID:         fmt.Sprintf("req-%s", job.ID),
        SequenceID:        job.Sequence,
        SourceWorker:      w.ID,
        DestinationWorker: "scheduler",
        Layer:             job.Layers.Start,
        DType:            act.DType,
        Shape:            act.Shape,
        Payload:          act.Payload,
        Checksum:         act.Checksum,
    }
    if err := w.Transport.Send(packet); err != nil {
        return JobResult{}, err
    }

    return JobResult{
        JobID:        job.ID,
        Status:      "completed",
        Output:      output,
        TokensPerSec: 10.0,
        LayerCount:   job.Layers.Size(),
    }, nil
}

func (w *MockWorker) SimulateBenchmark(score float64) {
    w.Capabilities.Benchmarks["mock"] = score
    w.State = StateReady
}

func NewMockWorkers() map[string]*MockWorker {
    return map[string]*MockWorker{
        "worker-a": NewMockWorker("worker-a", PlatformMock, []model.LayerRange{{Start: 0, End: 9}}),
        "worker-b": NewMockWorker("worker-b", PlatformMock, []model.LayerRange{{Start: 10, End: 19}}),
        "worker-c": NewMockWorker("worker-c", PlatformMock, []model.LayerRange{{Start: 20, End: 29}}),
    }
}

func SimulateDistributedExecution() error {
    workers := NewMockWorkers()
    for _, w := range workers {
        if err := w.Register(context.Background()); err != nil {
            return err
        }
        if err := w.Start(context.Background()); err != nil {
            return err
        }
    }

    job := Job{
        ID:      "job-42",
        ModelID: "llama-3b",
        Layers:  model.LayerRange{Start: 0, End: 9},
        Payload: []byte("prompt"),
        Sequence: 1,
    }

    _, err := workers["worker-a"].Execute(context.Background(), job)
    if err != nil {
        return err
    }
    return nil
}

func (w *MockWorker) LastHeartbeat() time.Time {
    return time.Now()
}
