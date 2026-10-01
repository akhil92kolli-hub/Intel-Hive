package protocol

import (
	"fmt"
	"regexp"
)

var sha256DigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func ValidSHA256Digest(value string) bool { return sha256DigestPattern.MatchString(value) }

// ValidateInference validates the v2 assignment and binds input to its source
// layer and worker. Legacy mock jobs without a phase use the separate payload.
func (a JobAssignment) ValidateInference() error {
	if a.AssignmentID == "" || a.RequestID == "" || a.ModelVersion == "" || a.WorkerID == "" || a.ShardID == "" {
		return fmt.Errorf("assignment, request, model version, worker, and shard IDs are required")
	}
	if !ValidSHA256Digest(a.ModelArtifactDigest) {
		return fmt.Errorf("model_artifact_digest must be a SHA-256 digest")
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
	if a.Phase == InferencePhasePrefill && a.PassOrdinal != 0 {
		return fmt.Errorf("prefill pass_ordinal must be zero")
	}
	if a.Phase == InferencePhaseDecode && a.PassOrdinal == 0 {
		return fmt.Errorf("decode pass_ordinal must be greater than zero")
	}
	if a.Prompt != "" {
		if a.KVTokenOffset != 0 || a.TokenCount != 0 || a.LayerStart != 0 {
			return fmt.Errorf("prompt prefill must start at layer zero with unresolved token_count")
		}
	} else if a.TokenCount == 0 {
		return fmt.Errorf("token_count must be positive")
	}
	if len(a.InputTokenIDs) > 0 && uint32(len(a.InputTokenIDs)) != a.TokenCount {
		return fmt.Errorf("token_count does not match input_token_ids")
	}
	if a.Activation != nil {
		if err := a.Activation.ValidateBoundary(a.JobID, a.RequestID, a.SequenceID, a.ModelID, a.ModelVersion, a.PreviousWorker, a.WorkerID, a.LayerStart-1, a.Position); err != nil {
			return err
		}
		if err := a.Activation.ValidateCanonical(); err != nil {
			return err
		}
		if a.Activation.ModelArtifactDigest != a.ModelArtifactDigest ||
			a.Activation.PassOrdinal != a.PassOrdinal ||
			a.Activation.KVTokenOffset != a.KVTokenOffset ||
			a.Activation.TokenCount != a.TokenCount {
			return fmt.Errorf("activation execution metadata does not match assignment")
		}
	}
	return nil
}
