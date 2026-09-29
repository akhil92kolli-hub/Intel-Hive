package mock

import (
	"context"
	"fmt"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/transport"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
)

type WorkerPlatform = worker.Platform

const (
	PlatformAndroid = worker.PlatformAndroid
	PlatformIOS     = worker.PlatformIOS
	PlatformMock    = worker.PlatformMock
)

type WorkerState = worker.State

const (
	StateUnknown       = worker.StateUnknown
	StateRegistering   = worker.StateRegistering
	StateRegistered    = worker.StateRegistered
	StateUnbenchmarked = worker.StateUnbenchmarked
	StateReady         = worker.StateReady
	StateBusy          = worker.StateBusy
	StateOffline       = worker.StateOffline
)

type WorkerCapabilities = worker.Capabilities
type Job = worker.Job
type JobResult = worker.JobResult
type WorkerRuntime = worker.Runtime

// MockWorker is an in-memory test worker that simulates distributed execution.
type MockWorker struct {
	ID               string
	Platform         WorkerPlatform
	State            WorkerState
	Transport        *transport.InMemoryTransport
	CapabilityReport WorkerCapabilities
	Layers           []model.LayerRange
}

var _ worker.Runtime = (*MockWorker)(nil)

func NewMockWorker(id string, platform WorkerPlatform, layers []model.LayerRange) *MockWorker {
	return &MockWorker{
		ID:        id,
		Platform:  platform,
		State:     StateUnbenchmarked,
		Transport: transport.NewInMemoryTransport(),
		CapabilityReport: WorkerCapabilities{
			ID:         id,
			Platform:   platform,
			State:      StateUnbenchmarked,
			Runtime:    "llama.cpp",
			Backends:   []string{"cpu"},
			Layers:     layers,
			Benchmarks: map[string]float64{"mock": 10.0},
			MemoryMB:   4096,
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
		return fmt.Errorf("%w: %s", worker.ErrUnavailable, w.ID)
	}
	return nil
}

func (w *MockWorker) Capabilities(ctx context.Context) WorkerCapabilities {
	return w.CapabilityReport
}

func (w *MockWorker) Execute(ctx context.Context, job Job) (JobResult, error) {
	if w.State == StateOffline {
		return JobResult{}, fmt.Errorf("%w: %s", worker.ErrUnavailable, w.ID)
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
		DType:             act.DType,
		Shape:             act.Shape,
		Payload:           act.Payload,
		Checksum:          act.Checksum,
	}
	if err := w.Transport.Send(packet); err != nil {
		return JobResult{}, err
	}

	return JobResult{
		JobID:        job.ID,
		Status:       "completed",
		Output:       output,
		TokensPerSec: 10.0,
		LayerCount:   job.Layers.Size(),
	}, nil
}

func (w *MockWorker) SimulateBenchmark(score float64) {
	w.CapabilityReport.Benchmarks["mock"] = score
	w.State = StateReady
}

func NewMockWorkers() map[string]*MockWorker {
	return map[string]*MockWorker{
		"worker-a": NewMockWorker("worker-a", PlatformMock, []model.LayerRange{{Start: 0, End: 9}}),
		"worker-b": NewMockWorker("worker-b", PlatformMock, []model.LayerRange{{Start: 10, End: 19}}),
		"worker-c": NewMockWorker("worker-c", PlatformMock, []model.LayerRange{{Start: 20, End: 29}}),
	}
}

func NewMockWorkerRuntimes() map[string]worker.Runtime {
	runtimes := make(map[string]worker.Runtime)
	for id, mockWorker := range NewMockWorkers() {
		runtimes[id] = mockWorker
	}
	return runtimes
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
		ID:       "job-42",
		ModelID:  "llama-3b",
		Layers:   model.LayerRange{Start: 0, End: 9},
		Payload:  []byte("prompt"),
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
