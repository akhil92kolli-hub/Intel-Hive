package protocol

import "fmt"

// ValidateInference validates the v2 assignment and binds input to its source
// layer and worker. Legacy mock jobs without a phase use the separate payload.
func (a JobAssignment) ValidateInference() error {
	if a.AssignmentID == "" || a.RequestID == "" || a.ModelVersion == "" || a.WorkerID == "" || a.ShardID == "" {
		return fmt.Errorf("assignment, request, model version, worker, and shard IDs are required")
	}
	if a.LayerStart < 0 || a.LayerEnd < a.LayerStart {
		return fmt.Errorf("invalid layer range")
	}
	if !a.FinalShard && a.NextWorker == "" {
		return fmt.Errorf("non-final shard requires next_worker")
	}
	step := InferenceStep{JobID: a.JobID, ModelID: a.ModelID, SequenceID: a.SequenceID, Phase: a.Phase, Position: a.Position, Prompt: a.Prompt, InputTokenIDs: a.InputTokenIDs, Activation: a.Activation, FinalShard: a.FinalShard}
	if err := step.Validate(); err != nil {
		return err
	}
	if (a.LayerStart == 0) == (a.Activation != nil) {
		return fmt.Errorf("input does not match shard layer boundary")
	}
	if a.Activation != nil {
		return a.Activation.ValidateBoundary(a.JobID, a.RequestID, a.SequenceID, a.ModelID, a.ModelVersion, a.PreviousWorker, a.WorkerID, a.LayerStart-1, a.Position)
	}
	return nil
}
