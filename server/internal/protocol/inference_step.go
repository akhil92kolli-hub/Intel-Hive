package protocol

import (
	"fmt"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"strings"
)

type InferencePhase string

const (
	InferencePhasePrefill InferencePhase = "PREFILL"
	InferencePhaseDecode  InferencePhase = "DECODE"
)

// InferenceStep is one model pass through a single shard. SequenceID remains
// stable for the request so each worker can retain its shard-local KV cache.
// Position is the pass ordinal (prefill is zero); workers track the actual
// model token offset after tokenizing the prompt.
type InferenceStep struct {
	JobID         string               `json:"job_id"`
	ModelID       string               `json:"model_id"`
	SequenceID    string               `json:"sequence_id"`
	Phase         InferencePhase       `json:"phase"`
	Position      uint32               `json:"position"`
	Prompt        string               `json:"prompt,omitempty"`
	InputTokenIDs []uint32             `json:"input_token_ids,omitempty"`
	Activation    *activation.Envelope `json:"activation,omitempty"`
	FinalShard    bool                 `json:"final_shard"`
}

func (s InferenceStep) Validate() error {
	if strings.TrimSpace(s.JobID) == "" || strings.TrimSpace(s.ModelID) == "" ||
		strings.TrimSpace(s.SequenceID) == "" {
		return fmt.Errorf("job_id, model_id, and sequence_id are required")
	}
	if s.Phase != InferencePhasePrefill && s.Phase != InferencePhaseDecode {
		return fmt.Errorf("unsupported inference phase %q", s.Phase)
	}
	inputForms := 0
	if s.Prompt != "" {
		inputForms++
	}
	if len(s.InputTokenIDs) > 0 {
		inputForms++
	}
	if s.Activation != nil {
		if err := s.Activation.Validate(); err != nil {
			return err
		}
		if s.Activation.JobID != s.JobID || s.Activation.ModelID != s.ModelID || s.Activation.SequenceID != s.SequenceID || s.Activation.Position != s.Position {
			return fmt.Errorf("activation does not match inference step")
		}
		inputForms++
	}
	if inputForms != 1 {
		return fmt.Errorf("exactly one of prompt, input_token_ids, or activation is required")
	}
	if s.Phase == InferencePhasePrefill && s.Position != 0 {
		return fmt.Errorf("prefill position must be zero")
	}
	if s.Phase == InferencePhaseDecode && s.Position == 0 {
		return fmt.Errorf("decode position must be greater than zero")
	}
	if s.Phase == InferencePhaseDecode && len(s.InputTokenIDs) > 0 && len(s.InputTokenIDs) != 1 {
		return fmt.Errorf("decode steps must contain exactly one input token")
	}
	if s.Phase == InferencePhaseDecode && s.Prompt != "" {
		return fmt.Errorf("decode steps cannot contain prompt text")
	}
	return nil
}

// InferenceStepResult is the output from one shard pass. Non-final shards
// return an activation; the final shard returns one sampled token or EOS.
type InferenceStepResult struct {
	JobID          string               `json:"job_id"`
	SequenceID     string               `json:"sequence_id"`
	Position       uint32               `json:"position"`
	Activation     *activation.Envelope `json:"activation,omitempty"`
	SampledTokenID *uint32              `json:"sampled_token_id,omitempty"`
	EndOfSequence  bool                 `json:"end_of_sequence,omitempty"`
}

func (r InferenceStepResult) ValidateFor(step InferenceStep) error {
	if r.JobID != step.JobID || r.SequenceID != step.SequenceID || r.Position != step.Position {
		return fmt.Errorf("inference result does not match job, sequence, and position")
	}
	if step.FinalShard {
		hasToken := r.SampledTokenID != nil
		if hasToken == r.EndOfSequence {
			return fmt.Errorf("final shard must return exactly one sampled token or EOS")
		}
		if r.Activation != nil {
			return fmt.Errorf("final shard must not return an activation")
		}
		return nil
	}
	if r.Activation == nil || r.SampledTokenID != nil || r.EndOfSequence {
		return fmt.Errorf("non-final shard must return only a non-empty activation")
	}
	if err := r.Activation.Validate(); err != nil {
		return err
	}
	if r.Activation.JobID != r.JobID || r.Activation.SequenceID != r.SequenceID || r.Activation.Position != r.Position {
		return fmt.Errorf("activation does not match result")
	}
	return nil
}
