package scheduler

import (
	"fmt"
	"sync"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

type WorkerState string

const (
	WorkerStateReady         WorkerState = "READY"
	WorkerStateBusy          WorkerState = "BUSY"
	WorkerStateOffline       WorkerState = "OFFLINE"
	WorkerStateUnbenchmarked WorkerState = "UNBENCHMARKED"
)

type WorkerPlatform string

const (
	WorkerPlatformMock    WorkerPlatform = "mock"
	WorkerPlatformAndroid WorkerPlatform = "android"
	WorkerPlatformIOS     WorkerPlatform = "ios"
)

// WorkerRecord is the scheduler's platform-independent view of a worker.
type WorkerRecord struct {
	ID           string
	State        WorkerState
	Platform     WorkerPlatform
	Layers       []model.LayerRange
	Benchmarks   map[string]float64
	MemoryMB     int
	Load         int
	LastSeen     time.Time
	RegisteredAt time.Time
}

func (w *WorkerRecord) CanServeLayer(layer model.LayerRange) bool {
	for _, covered := range w.Layers {
		if covered.Start <= layer.Start && layer.End <= covered.End {
			return true
		}
	}
	return false
}

func (w *WorkerRecord) IsEligible() bool {
	return w.State == WorkerStateReady
}

func (w *WorkerRecord) Score() float64 {
	throughput := 0.0
	for _, value := range w.Benchmarks {
		if value > throughput {
			throughput = value
		}
	}
	if throughput == 0 {
		throughput = 1
	}
	loadFactor := float64(100-w.Load) / 100
	return throughput * loadFactor
}

type Registry struct {
	mu      sync.RWMutex
	workers map[string]*WorkerRecord
}

func NewRegistry() *Registry {
	return &Registry{workers: make(map[string]*WorkerRecord)}
}

func (r *Registry) Register(worker WorkerRecord) error {
	if worker.ID == "" {
		return fmt.Errorf("worker ID cannot be empty")
	}
	if len(worker.Layers) == 0 {
		return fmt.Errorf("worker must advertise at least one layer range")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now().UTC()
	if existing, ok := r.workers[worker.ID]; ok {
		worker.RegisteredAt = existing.RegisteredAt
	}
	worker.LastSeen = now
	if worker.Benchmarks == nil {
		worker.Benchmarks = map[string]float64{}
	}
	r.workers[worker.ID] = &worker
	return nil
}

func (r *Registry) Unregister(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.workers[id]; !ok {
		return fmt.Errorf("worker %s not found", id)
	}
	delete(r.workers, id)
	return nil
}

func (r *Registry) Get(id string) (WorkerRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	worker, ok := r.workers[id]
	if !ok {
		return WorkerRecord{}, fmt.Errorf("worker %s not found", id)
	}
	return *worker, nil
}

func (r *Registry) List() []WorkerRecord {
	r.mu.RLock()
	defer r.mu.RUnlock()
	workers := make([]WorkerRecord, 0, len(r.workers))
	for _, worker := range r.workers {
		workers = append(workers, *worker)
	}
	return workers
}

func (r *Registry) UpdateState(id string, state WorkerState) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	worker, ok := r.workers[id]
	if !ok {
		return fmt.Errorf("worker %s not found", id)
	}
	worker.State = state
	worker.LastSeen = time.Now().UTC()
	return nil
}

func (r *Registry) UpdateLoad(id string, load int) error {
	if load < 0 || load > 100 {
		return fmt.Errorf("load must be between 0 and 100")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	worker, ok := r.workers[id]
	if !ok {
		return fmt.Errorf("worker %s not found", id)
	}
	worker.Load = load
	worker.LastSeen = time.Now().UTC()
	return nil
}

func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.workers)
}
