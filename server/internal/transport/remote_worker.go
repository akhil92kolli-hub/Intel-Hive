package transport

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/worker"
	"github.com/akhil92kolli-hub/Intel-Hive/server/internal/protocol"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

type remoteJobResult struct {
	result worker.JobResult
	err    error
}

type pendingRemoteJob struct {
	job        worker.Job
	jobID      string
	sequenceID string
	position   uint32
	finalShard bool
	result     chan remoteJobResult
}

var ErrStaleAssignment = errors.New("stale job assignment response")

const staleAssignmentRetention = 2 * time.Minute

// RemoteWorker adapts a registered WebSocket peer to the scheduler runtime.
type RemoteWorker struct {
	id           string
	capabilities worker.Capabilities
	conn         *websocket.Conn
	writeMu      *sync.Mutex
	pendingMu    sync.Mutex
	pending      map[string]pendingRemoteJob
	retired      map[string]time.Time
	closed       chan struct{}
	closeOnce    sync.Once
	ready        chan struct{}
	readyOnce    sync.Once
}

func NewRemoteWorker(id string, capabilities worker.Capabilities, conn *websocket.Conn) *RemoteWorker {
	remote := NewRemoteWorkerWithWriteLock(id, capabilities, conn, &sync.Mutex{})
	remote.Activate()
	return remote
}

func NewRemoteWorkerWithWriteLock(
	id string,
	capabilities worker.Capabilities,
	conn *websocket.Conn,
	writeMu *sync.Mutex,
) *RemoteWorker {
	if writeMu == nil {
		writeMu = &sync.Mutex{}
	}
	return &RemoteWorker{
		id:           id,
		capabilities: capabilities,
		conn:         conn,
		writeMu:      writeMu,
		pending:      make(map[string]pendingRemoteJob),
		retired:      make(map[string]time.Time),
		closed:       make(chan struct{}),
		ready:        make(chan struct{}),
	}
}

func (w *RemoteWorker) Activate() {
	w.readyOnce.Do(func() { close(w.ready) })
}

func (w *RemoteWorker) Start(context.Context) error { return nil }

func (w *RemoteWorker) Stop(context.Context) error {
	w.Close()
	return nil
}

func (w *RemoteWorker) Register(context.Context) error { return nil }

func (w *RemoteWorker) Heartbeat(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-w.closed:
		return fmt.Errorf("%w: %s", worker.ErrUnavailable, w.id)
	default:
		return nil
	}
}

func (w *RemoteWorker) Capabilities(context.Context) worker.Capabilities {
	return w.capabilities
}

func (w *RemoteWorker) EndSequence(ctx context.Context, jobID, sequenceID string, completed bool) error {
	if strings.TrimSpace(jobID) == "" || strings.TrimSpace(sequenceID) == "" {
		return fmt.Errorf("job_id and sequence_id are required")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-w.closed:
		return fmt.Errorf("%w: %s", worker.ErrUnavailable, w.id)
	default:
	}
	if err := w.write(workerReply{Type: "sequence_end", Payload: protocol.SequenceEnd{
		JobID: jobID, SequenceID: sequenceID, Completed: completed,
	}}); err != nil {
		w.Close()
		return fmt.Errorf("%w: ending sequence: %v", worker.ErrUnavailable, err)
	}
	return nil
}

func (w *RemoteWorker) Execute(ctx context.Context, job worker.Job) (worker.JobResult, error) {
	select {
	case <-ctx.Done():
		return worker.JobResult{}, ctx.Err()
	case <-w.closed:
		return worker.JobResult{}, fmt.Errorf("%w: %s", worker.ErrUnavailable, w.id)
	case <-w.ready:
	}

	assignmentID := uuid.NewString()
	result := make(chan remoteJobResult, 1)
	w.pendingMu.Lock()
	select {
	case <-w.closed:
		w.pendingMu.Unlock()
		return worker.JobResult{}, fmt.Errorf("%w: %s", worker.ErrUnavailable, w.id)
	default:
	}
	w.pending[assignmentID] = pendingRemoteJob{
		job: job, jobID: job.ID, sequenceID: job.SequenceID, position: job.Position,
		finalShard: job.FinalShard, result: result,
	}
	w.pendingMu.Unlock()

	assignment := protocol.JobAssignment{
		RequestID: job.RequestID, ModelVersion: job.ModelVersion, ModelArtifactDigest: job.ModelArtifactDigest, WorkerID: w.id, PreviousWorker: job.PreviousWorker, NextWorker: job.NextWorker,
		AssignmentID:  assignmentID,
		JobID:         job.ID,
		ModelID:       job.ModelID,
		ShardID:       job.ShardID,
		Phase:         protocol.InferencePhase(job.Phase),
		SequenceID:    job.SequenceID,
		Position:      job.Position,
		PassOrdinal:   job.PassOrdinal,
		KVTokenOffset: job.KVTokenOffset,
		TokenCount:    job.TokenCount,
		Prompt:        job.Prompt,
		InputTokenIDs: job.InputTokenIDs,
		Activation:    job.Activation,
		FinalShard:    job.FinalShard,
		LayerStart:    job.Layers.Start,
		LayerEnd:      job.Layers.End,
		Sequence:      job.Sequence,
		Payload:       job.Payload,
	}
	if job.Phase != "" {
		if err := assignment.ValidateInference(); err != nil {
			w.removePending(assignmentID)
			return worker.JobResult{}, err
		}
	}
	if err := w.write(workerReply{Type: "job_assignment", Payload: assignment}); err != nil {
		w.removePending(assignmentID)
		w.Close()
		return worker.JobResult{}, fmt.Errorf("%w: sending assignment: %v", worker.ErrUnavailable, err)
	}

	select {
	case completed := <-result:
		return completed.result, completed.err
	case <-ctx.Done():
		w.removePending(assignmentID)
		return worker.JobResult{}, ctx.Err()
	case <-w.closed:
		w.removePending(assignmentID)
		return worker.JobResult{}, fmt.Errorf("%w: %s disconnected during assignment", worker.ErrUnavailable, w.id)
	}
}

func (w *RemoteWorker) Complete(message protocol.JobComplete) error {
	if message.WorkerID != w.id {
		return fmt.Errorf("job completion worker_id does not match connection")
	}
	if message.Status != "completed" && message.Status != "COMPLETED" {
		return fmt.Errorf("unsupported job completion status %q", message.Status)
	}
	w.pendingMu.Lock()
	pending, ok := w.pending[message.AssignmentID]
	if ok && pending.jobID != message.JobID {
		w.pendingMu.Unlock()
		return fmt.Errorf("job completion job_id does not match assignment")
	}
	if ok && pending.sequenceID != "" &&
		(message.SequenceID != pending.sequenceID || message.Position != pending.position) {
		w.pendingMu.Unlock()
		return fmt.Errorf("job completion sequence_id or position does not match assignment")
	}
	if ok && pending.sequenceID != "" {
		job := pending.job
		if message.PassOrdinal != job.PassOrdinal ||
			message.KVTokenOffsetBefore != job.KVTokenOffset ||
			message.KVTokenOffsetAfter <= message.KVTokenOffsetBefore {
			w.pendingMu.Unlock()
			return fmt.Errorf("job completion execution metadata does not match assignment")
		}
		actualTokenCount := message.KVTokenOffsetAfter - message.KVTokenOffsetBefore
		if job.TokenCount > 0 && actualTokenCount != job.TokenCount {
			w.pendingMu.Unlock()
			return fmt.Errorf("job completion token count does not match assignment")
		}
		if job.TokenCount == 0 && job.Prompt == "" {
			w.pendingMu.Unlock()
			return fmt.Errorf("job completion resolved a token count for a non-prompt assignment")
		}
	}
	if ok && pending.sequenceID != "" {
		if pending.finalShard {
			if (message.SampledTokenID != nil) == message.EndOfSequence || message.Activation != nil {
				w.pendingMu.Unlock()
				return fmt.Errorf("final shard completion must contain a sampled token or EOS")
			}
		} else if message.Activation == nil || message.SampledTokenID != nil || message.EndOfSequence {
			w.pendingMu.Unlock()
			return fmt.Errorf("non-final shard completion must contain only an activation")
		}
	}
	if ok && pending.sequenceID != "" && message.Activation != nil {
		j := pending.job
		if err := message.Activation.ValidateBoundary(j.ID, j.RequestID, j.SequenceID, j.ModelID, j.ModelVersion, w.id, j.NextWorker, j.Layers.End, j.Position); err != nil {
			w.pendingMu.Unlock()
			return err
		}
		if err := message.Activation.ValidateCanonical(); err != nil {
			w.pendingMu.Unlock()
			return err
		}
		if message.Activation.ModelArtifactDigest != j.ModelArtifactDigest ||
			message.Activation.PassOrdinal != message.PassOrdinal ||
			message.Activation.KVTokenOffset != message.KVTokenOffsetBefore ||
			message.Activation.TokenCount != message.KVTokenOffsetAfter-message.KVTokenOffsetBefore {
			w.pendingMu.Unlock()
			return fmt.Errorf("activation execution metadata does not match completion")
		}
	}
	if ok {
		delete(w.pending, message.AssignmentID)
		w.retireLocked(message.AssignmentID)
	} else if w.isRetiredLocked(message.AssignmentID) {
		w.pendingMu.Unlock()
		return fmt.Errorf("%w: %s", ErrStaleAssignment, message.AssignmentID)
	}
	w.pendingMu.Unlock()
	if !ok {
		return fmt.Errorf("unknown or expired assignment %s", message.AssignmentID)
	}
	pending.result <- remoteJobResult{result: worker.JobResult{
		JobID: message.JobID, Status: "completed", Output: message.Output,
		Activation: message.Activation, SampledTokenID: message.SampledTokenID,
		EndOfSequence: message.EndOfSequence, GeneratedText: message.GeneratedText,
		TokensPerSec: message.TokensPerSec, LayerCount: message.LayerCount,
		PassOrdinal: message.PassOrdinal, KVTokenOffsetBefore: message.KVTokenOffsetBefore,
		KVTokenOffsetAfter: message.KVTokenOffsetAfter,
	}}
	return nil
}

func (w *RemoteWorker) Accept(message protocol.JobAccepted) error {
	w.pendingMu.Lock()
	pending, ok := w.pending[message.AssignmentID]
	if !ok {
		if w.isRetiredLocked(message.AssignmentID) {
			w.pendingMu.Unlock()
			return fmt.Errorf("%w: %s", ErrStaleAssignment, message.AssignmentID)
		}
		w.pendingMu.Unlock()
		return fmt.Errorf("unknown or expired assignment %s", message.AssignmentID)
	}
	if message.Accepted {
		w.pendingMu.Unlock()
		return nil
	}
	delete(w.pending, message.AssignmentID)
	w.retireLocked(message.AssignmentID)
	w.pendingMu.Unlock()
	reason := message.Reason
	if reason == "" {
		reason = "worker rejected the assignment"
	}
	pending.result <- remoteJobResult{err: errors.New(reason)}
	return nil
}

func (w *RemoteWorker) Fail(message protocol.JobFailed) error {
	if message.WorkerID != w.id {
		return fmt.Errorf("job failure worker_id does not match connection")
	}
	w.pendingMu.Lock()
	pending, ok := w.pending[message.AssignmentID]
	if ok && pending.jobID != message.JobID {
		w.pendingMu.Unlock()
		return fmt.Errorf("job failure job_id does not match assignment")
	}
	if ok {
		delete(w.pending, message.AssignmentID)
		w.retireLocked(message.AssignmentID)
	} else if w.isRetiredLocked(message.AssignmentID) {
		w.pendingMu.Unlock()
		return fmt.Errorf("%w: %s", ErrStaleAssignment, message.AssignmentID)
	}
	w.pendingMu.Unlock()
	if !ok {
		return fmt.Errorf("unknown or expired assignment %s", message.AssignmentID)
	}
	if message.Unavailable {
		pending.result <- remoteJobResult{err: fmt.Errorf("%w: %s", worker.ErrUnavailable, message.Reason)}
	} else {
		reason := message.Reason
		if reason == "" {
			reason = "worker reported an inference failure"
		}
		pending.result <- remoteJobResult{err: errors.New(reason)}
	}
	return nil
}

func (w *RemoteWorker) Close() {
	w.closeOnce.Do(func() {
		close(w.closed)
		w.pendingMu.Lock()
		for assignmentID, pending := range w.pending {
			delete(w.pending, assignmentID)
			pending.result <- remoteJobResult{err: fmt.Errorf("%w: %s disconnected", worker.ErrUnavailable, w.id)}
		}
		w.pendingMu.Unlock()
		_ = w.conn.Close()
	})
}

func (w *RemoteWorker) write(message workerReply) error {
	w.writeMu.Lock()
	defer w.writeMu.Unlock()
	if err := w.conn.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
		return err
	}
	return w.conn.WriteJSON(message)
}

func (w *RemoteWorker) removePending(assignmentID string) {
	w.pendingMu.Lock()
	if _, exists := w.pending[assignmentID]; exists {
		delete(w.pending, assignmentID)
		w.retireLocked(assignmentID)
	}
	w.pendingMu.Unlock()
}

func (w *RemoteWorker) retireLocked(assignmentID string) {
	now := time.Now()
	for id, expiresAt := range w.retired {
		if !now.Before(expiresAt) {
			delete(w.retired, id)
		}
	}
	w.retired[assignmentID] = now.Add(staleAssignmentRetention)
}

func (w *RemoteWorker) isRetiredLocked(assignmentID string) bool {
	expiresAt, exists := w.retired[assignmentID]
	if exists && !time.Now().Before(expiresAt) {
		delete(w.retired, assignmentID)
		return false
	}
	return exists
}

func workerCapabilitiesFromRegistration(
	request protocol.WorkerRegisterRequest,
) worker.Capabilities {
	ranges := make([]model.LayerRange, 0, len(request.LoadedShards))
	for _, shard := range request.LoadedShards {
		ranges = append(ranges, model.LayerRange{Start: shard.LayerStart, End: shard.LayerEnd})
	}
	benchmark := make(map[string]float64)
	if request.Inference.Performance != nil {
		benchmark["tokens_per_second"] = request.Inference.Performance.TokensPerSecond
	}
	state := worker.StateUnbenchmarked
	if strings.EqualFold(request.Inference.BenchmarkStatus, "COMPLETED") && request.Inference.Performance != nil {
		state = worker.StateReady
	}
	return worker.Capabilities{
		ID: request.WorkerID, Platform: worker.Platform(request.Platform), State: state,
		Runtime: request.Inference.Runtime, Backends: []string{request.Inference.Backend},
		Layers: ranges, Benchmarks: benchmark, MemoryMB: int(request.Memory.TotalMB),
	}
}
