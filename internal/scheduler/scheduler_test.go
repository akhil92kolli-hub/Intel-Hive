package scheduler

import (
	"context"
	"errors"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"testing"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/mock"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
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

	plan, err := scheduler.BuildPlan("job-1", "qwen2.5-3b", testThreeShards())
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Shards) != 3 || plan.Shards[0].WorkerID != "worker-a" ||
		plan.Shards[1].WorkerID != "worker-b" || plan.Shards[2].WorkerID != "worker-c" {
		t.Fatalf("unexpected plan: %+v", plan)
	}
}

func TestUnbenchmarkedWorkersAreNotSelected(t *testing.T) {
	scheduler := New(nil)
	runtime := mock.NewMockWorker("worker-a", mock.PlatformMock, []model.LayerRange{{Start: 0, End: 9}})
	if err := scheduler.RegisterWorker(WorkerRecord{
		ID: "worker-a", State: WorkerStateUnbenchmarked, Layers: runtime.Layers,
	}, runtime); err != nil {
		t.Fatal(err)
	}
	_, err := scheduler.BuildPlan("job-1", "model", testThreeShards()[:1])
	if err == nil {
		t.Fatal("expected unbenchmarked worker to be rejected")
	}
}

func TestBuildPlanDoesNotAssignShardAcrossModels(t *testing.T) {
	s := New(nil)
	runtime := mock.NewMockWorker("worker-a", mock.PlatformMock, []model.LayerRange{{Start: 0, End: 9}})
	if err := s.RegisterWorker(WorkerRecord{
		ID: runtime.ID, State: WorkerStateReady, Layers: runtime.Layers,
		ModelShards: map[string]map[string]model.LayerRange{
			"model-a": {
				"shard-a": model.LayerRange{Start: 0, End: 9},
			},
		},
	}, runtime); err != nil {
		t.Fatal(err)
	}
	_, err := s.BuildPlan("job-cross-model", "model-b", []model.ModelShard{
		{Version: "v1", ID: "shard-a", ModelID: "model-b", Layers: model.LayerRange{Start: 0, End: 9}},
	})
	if err == nil {
		t.Fatal("expected a model-specific shard mismatch to be rejected")
	}
}

func TestExecutePlan(t *testing.T) {
	scheduler := New(nil)
	for id, runtime := range mock.NewMockWorkers() {
		if err := scheduler.RegisterWorker(WorkerRecord{
			ID: id, State: WorkerStateReady, Layers: runtime.Layers,
			Benchmarks: map[string]float64{"mock": 10},
		}, runtime); err != nil {
			t.Fatal(err)
		}
	}
	plan, err := scheduler.BuildPlan("job-1", "model", testThreeShards())
	if err != nil {
		t.Fatal(err)
	}
	result, err := scheduler.Execute(context.Background(), plan, []byte("prompt"))
	if err != nil || len(result.FinalOutput) == 0 {
		t.Fatalf("execution failed: result=%+v err=%v", result, err)
	}
}

func TestExecuteRebuildsPlanAfterWorkerFailure(t *testing.T) {
	scheduler := New(nil)
	for id, runtime := range mock.NewMockWorkers() {
		if id == "worker-b" {
			runtime.State = mock.StateOffline
		}
		if err := scheduler.RegisterWorker(WorkerRecord{
			ID: id, State: WorkerStateReady, Platform: WorkerPlatformMock,
			Layers: runtime.Layers, Benchmarks: map[string]float64{"mock": 10},
		}, runtime); err != nil {
			t.Fatal(err)
		}
	}

	replacement := mock.NewMockWorker("worker-d", mock.PlatformMock, []model.LayerRange{{Start: 10, End: 19}})
	if err := scheduler.RegisterWorker(WorkerRecord{
		ID: replacement.ID, State: WorkerStateReady, Platform: WorkerPlatformMock,
		Layers: replacement.Layers, Benchmarks: map[string]float64{"mock": 5},
	}, replacement); err != nil {
		t.Fatal(err)
	}

	plan, err := scheduler.BuildPlan("job-retry", "model", testThreeShards())
	if err != nil {
		t.Fatal(err)
	}
	if plan.Shards[1].WorkerID != "worker-b" {
		t.Fatalf("expected initial plan to select worker-b, got %+v", plan.Shards)
	}

	result, err := scheduler.Execute(context.Background(), plan, []byte("prompt"))
	if err != nil {
		t.Fatalf("execution did not recover from worker failure: %v", err)
	}
	if result.JobID != "job-retry" || len(result.FinalOutput) == 0 {
		t.Fatalf("unexpected recovered result: %+v", result)
	}
	if got := result.ShardMetrics["s1"].WorkerID; got != "worker-d" {
		t.Fatalf("expected replacement worker-d for shard s1, got %q", got)
	}
	offline, err := scheduler.Registry().Get("worker-b")
	if err != nil {
		t.Fatal(err)
	}
	if offline.State != WorkerStateOffline {
		t.Fatalf("failed worker should be marked offline, got %q", offline.State)
	}
}

func TestExecuteFailsWhenNoReplacementWorkerExists(t *testing.T) {
	scheduler := New(nil)
	runtime := mock.NewMockWorker("worker-a", mock.PlatformMock, []model.LayerRange{{Start: 0, End: 9}})
	runtime.State = mock.StateOffline
	if err := scheduler.RegisterWorker(WorkerRecord{
		ID: runtime.ID, State: WorkerStateReady, Layers: runtime.Layers,
		Benchmarks: map[string]float64{"mock": 10},
	}, runtime); err != nil {
		t.Fatal(err)
	}
	plan, err := scheduler.BuildPlan("job-no-replacement", "model", testThreeShards()[:1])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := scheduler.Execute(context.Background(), plan, []byte("prompt")); err == nil {
		t.Fatal("expected execution to fail without a replacement")
	}
}

func TestExecuteDoesNotMarkWorkerOfflineForInferenceError(t *testing.T) {
	scheduler := New(nil)
	runtime := &failingMockWorker{
		MockWorker: mock.NewMockWorker(
			"worker-a",
			mock.PlatformMock,
			[]model.LayerRange{{Start: 0, End: 9}},
		),
	}
	if err := scheduler.RegisterWorker(WorkerRecord{
		ID: runtime.ID, State: WorkerStateReady, Layers: runtime.Layers,
		Benchmarks: map[string]float64{"mock": 10},
	}, runtime); err != nil {
		t.Fatal(err)
	}
	plan, err := scheduler.BuildPlan("job-inference-error", "model", testThreeShards()[:1])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := scheduler.Execute(context.Background(), plan, []byte("prompt")); err == nil {
		t.Fatal("expected inference error")
	}
	record, err := scheduler.Registry().Get(runtime.ID)
	if err != nil {
		t.Fatal(err)
	}
	if record.State != WorkerStateReady {
		t.Fatalf("inference error should not mark worker offline, got %q", record.State)
	}
}

func TestGenerateRunsPrefillAndDecodeAcrossAllShardsUntilEOS(t *testing.T) {
	s, calls := newTokenLoopScheduler()
	plan, err := s.BuildPlan("job-generate", "model", testThreeShards())
	if err != nil {
		t.Fatal(err)
	}

	result, err := s.Generate(context.Background(), plan, "hello", 5)
	if err != nil {
		t.Fatal(err)
	}
	if got := result.TokenIDs; len(got) != 2 || got[0] != 100 || got[1] != 101 {
		t.Fatalf("unexpected generated tokens: %v", got)
	}
	if string(result.Text) != "AB" {
		t.Fatalf("unexpected generated text: %q", result.Text)
	}
	if len(*calls) != 9 {
		t.Fatalf("expected three shard passes, got %d executions", len(*calls))
	}
	for i, job := range *calls {
		pass := i / 3
		wantPhase := "DECODE"
		if pass == 0 {
			wantPhase = "PREFILL"
		}
		if job.Phase != wantPhase || job.Position != uint32(pass) ||
			job.SequenceID != "job-generate:0" {
			t.Fatalf("unexpected step at index %d: %+v", i, job)
		}
		if i%3 == 0 {
			if pass == 0 && job.Prompt != "hello" {
				t.Fatalf("first prefill shard did not receive prompt: %+v", job)
			}
			if pass > 0 && (len(job.InputTokenIDs) != 1 || job.InputTokenIDs[0] != uint32(99+pass)) {
				t.Fatalf("first decode shard did not receive the prior token: %+v", job)
			}
		} else if job.Activation == nil {
			t.Fatalf("downstream shard did not receive activation: %+v", job)
		}
	}
}

func TestGenerateStopsAtMaxTokens(t *testing.T) {
	s, calls := newTokenLoopScheduler()
	plan, err := s.BuildPlan("job-limit", "model", testThreeShards())
	if err != nil {
		t.Fatal(err)
	}
	result, err := s.Generate(context.Background(), plan, "hello", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.TokenIDs) != 1 || result.TokenIDs[0] != 100 {
		t.Fatalf("expected exactly one generated token, got %v", result.TokenIDs)
	}
	if len(*calls) != 3 {
		t.Fatalf("max_tokens=1 should execute only prefill, got %d shard executions", len(*calls))
	}
}

type failingMockWorker struct {
	*mock.MockWorker
}

func (w *failingMockWorker) Execute(context.Context, mock.Job) (mock.JobResult, error) {
	return mock.JobResult{}, errors.New("inference failed")
}

type tokenLoopWorker struct {
	*mock.MockWorker
	calls *[]worker.Job
}

func (w *tokenLoopWorker) Execute(_ context.Context, job worker.Job) (worker.JobResult, error) {
	*w.calls = append(*w.calls, job)
	if !job.FinalShard {
		a := activation.NewEnvelope(job.ID, job.RequestID, job.SequenceID, job.ModelID, job.ModelVersion, job.WorkerID, job.NextWorker, job.Layers.End, "uint8", []int64{1}, []byte{1})
		a.Position = job.Position
		return worker.JobResult{
			JobID: job.ID, Activation: &a,
		}, nil
	}
	if job.Position >= 2 {
		return worker.JobResult{JobID: job.ID, EndOfSequence: true}, nil
	}
	token := uint32(100 + job.Position)
	return worker.JobResult{
		JobID: job.ID, SampledTokenID: &token,
		GeneratedText: []byte{byte('A' + job.Position)},
	}, nil
}

func newTokenLoopScheduler() (*Scheduler, *[]worker.Job) {
	s := New(nil)
	calls := make([]worker.Job, 0)
	for id, runtime := range mock.NewMockWorkers() {
		tokenWorker := &tokenLoopWorker{MockWorker: runtime, calls: &calls}
		if err := s.RegisterWorker(WorkerRecord{
			ID: id, State: WorkerStateReady, Platform: WorkerPlatformMock,
			Layers: runtime.Layers, Benchmarks: map[string]float64{"mock": 10},
		}, tokenWorker); err != nil {
			panic(err)
		}
	}
	return s, &calls
}

func testThreeShards() []model.ModelShard {
	return []model.ModelShard{
		{Version: "v1", ID: "s0", Layers: model.LayerRange{Start: 0, End: 9}},
		{Version: "v1", ID: "s1", Layers: model.LayerRange{Start: 10, End: 19}},
		{Version: "v1", ID: "s2", Layers: model.LayerRange{Start: 20, End: 29}},
	}
}
