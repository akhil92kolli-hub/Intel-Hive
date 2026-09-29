package worker

import (
	"context"
	"errors"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

var ErrUnavailable = errors.New("worker unavailable")

type Platform string

const (
	PlatformAndroid Platform = "android"
	PlatformIOS     Platform = "ios"
	PlatformMock    Platform = "mock"
)

type State string

const (
	StateUnknown       State = "UNKNOWN"
	StateRegistering   State = "REGISTERING"
	StateRegistered    State = "REGISTERED"
	StateUnbenchmarked State = "UNBENCHMARKED"
	StateReady         State = "READY"
	StateBusy          State = "BUSY"
	StateOffline       State = "OFFLINE"
)

type Capabilities struct {
	ID         string
	Platform   Platform
	State      State
	Runtime    string
	Backends   []string
	Layers     []model.LayerRange
	Benchmarks map[string]float64
	MemoryMB   int
}

type Job struct {
	RequestID      string
	ModelVersion   string
	WorkerID       string
	PreviousWorker string
	NextWorker     string
	ID             string
	ModelID        string
	ShardID        string
	Layers         model.LayerRange
	Payload        []byte
	Activation     *activation.Envelope
	Phase          string
	SequenceID     string
	Position       uint32
	Prompt         string
	InputTokenIDs  []uint32
	FinalShard     bool
	Sequence       uint64
}

type JobResult struct {
	JobID          string
	Status         string
	Output         []byte
	Activation     *activation.Envelope
	SampledTokenID *uint32
	EndOfSequence  bool
	GeneratedText  []byte
	TokensPerSec   float64
	LayerCount     int
}

// Runtime is the scheduler-facing lifecycle and execution contract. Mock,
// local-device, and remote-device workers can all implement this interface.
type Runtime interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	Register(ctx context.Context) error
	Heartbeat(ctx context.Context) error
	Capabilities(ctx context.Context) Capabilities
	Execute(ctx context.Context, job Job) (JobResult, error)
}

// SequenceLifecycle is implemented by runtimes that retain per-request
// inference state and need an explicit cleanup signal.
type SequenceLifecycle interface {
	EndSequence(ctx context.Context, jobID, sequenceID string, completed bool) error
}
