package scheduler

import (
	"context"
	"testing"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

func TestBuildPlanSelectsEligibleWorkers(t *testing.T) {
	scheduler := New(nil)
	workers := mock.NewMockWorkers()
	for id, runtime := range workers {
		if err := scheduler.RegisterWorker(WorkerRecord{
			ID: id, State: WorkerStateReady, Platform: WorkerPlatformMock,
			Layers: runtime.Layers, Benchmarks: map[string]float64{"mock": 10}, MemoryMB: 4096,
		}, runtime); err != nil {
			t.Fatal(err)
		}
	}

	shards := []model.ModelShard{
		{ID: "s0", Layers: model.LayerRange{Start: 0, End: 9}},
		{ID: "s1", Layers: model.LayerRange{Start: 10, End: 19}},
		{ID: "s2", Layers: model.LayerRange{Start: 20, End: 29}},
	}
	plan, err := scheduler.BuildPlan("job-1", "qwen2.5-3b", shards)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Shards) != 3 || plan.Shards[0].WorkerID != "worker-a" || plan.Shards[1].WorkerID != "worker-b" || plan.Shards[2].WorkerID != "worker-c" {
		t.Fatalf("unexpected plan: %+v", plan)
	}
}

func TestUnbenchmarkedWorkersAreNotSelected(t *testing.T) {
	scheduler := New(nil)
	worker := mock.NewMockWorker("worker-a", mock.PlatformMock, []model.LayerRange{{Start: 0, End: 9}})
	if err := scheduler.RegisterWorker(WorkerRecord{ID: "worker-a", State: WorkerStateUnbenchmarked, Layers: worker.Layers}, worker); err != nil {
		t.Fatal(err)
	}
	_, err := scheduler.BuildPlan("job-1", "model", []model.ModelShard{{ID: "s0", Layers: model.LayerRange{Start: 0, End: 9}}})
	if err == nil {
		t.Fatal("expected unbenchmarked worker to be rejected")
	}
}

func TestExecutePlan(t *testing.T) {
	scheduler := New(nil)
	workers := mock.NewMockWorkers()
	for id, runtime := range workers {
		if err := scheduler.RegisterWorker(WorkerRecord{ID: id, State: WorkerStateReady, Layers: runtime.Layers, Benchmarks: map[string]float64{"mock": 10}}, runtime); err != nil {
			t.Fatal(err)
		}
	}
	shards := []model.ModelShard{
		{ID: "s0", Layers: model.LayerRange{Start: 0, End: 9}},
		{ID: "s1", Layers: model.LayerRange{Start: 10, End: 19}},
		{ID: "s2", Layers: model.LayerRange{Start: 20, End: 29}},
	}
	plan, err := scheduler.BuildPlan("job-1", "model", shards)
	if err != nil {
		t.Fatal(err)
	}
	result, err := scheduler.Execute(context.Background(), plan, []byte("prompt"))
	if err != nil || len(result.FinalOutput) == 0 {
		t.Fatalf("execution failed: result=%+v err=%v", result, err)
	}
}
