package protocol

import (
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"testing"
)

func TestInferenceStepValidate(t *testing.T) {
	tests := []struct {
		name    string
		step    InferenceStep
		wantErr bool
	}{
		{
			name: "prefill prompt tokens",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhasePrefill, InputTokenIDs: []uint32{11, 12}},
		},
		{
			name: "prefill prompt text",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhasePrefill, Prompt: "hello"},
		},
		{
			name: "prefill activation",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhasePrefill, Activation: testActivation()},
		},
		{
			name: "decode token",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhaseDecode, Position: 1, InputTokenIDs: []uint32{12}},
		},
		{
			name: "both input forms",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhaseDecode, Position: 1, InputTokenIDs: []uint32{12}, Activation: testActivation()},
			wantErr: true,
		},
		{
			name: "no input",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhaseDecode, Position: 1},
			wantErr: true,
		},
		{
			name: "decode multiple tokens",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhaseDecode, Position: 1, InputTokenIDs: []uint32{12, 13}},
			wantErr: true,
		},
		{
			name: "decode at prefill position",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhaseDecode, InputTokenIDs: []uint32{12}},
			wantErr: true,
		},
		{
			name: "prefill nonzero position",
			step: InferenceStep{JobID: "job", ModelID: "model", SequenceID: "seq",
				Phase: InferencePhasePrefill, Position: 1, InputTokenIDs: []uint32{12}},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.step.Validate()
			if (err != nil) != test.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestInferenceStepResultValidateFor(t *testing.T) {
	token := uint32(42)
	tests := []struct {
		name    string
		step    InferenceStep
		result  InferenceStepResult
		wantErr bool
	}{
		{
			name:   "intermediate activation",
			step:   InferenceStep{JobID: "job", SequenceID: "seq", FinalShard: false},
			result: InferenceStepResult{JobID: "job", SequenceID: "seq", Activation: testActivation()},
		},
		{
			name:   "final sampled token",
			step:   InferenceStep{JobID: "job", SequenceID: "seq", FinalShard: true},
			result: InferenceStepResult{JobID: "job", SequenceID: "seq", SampledTokenID: &token},
		},
		{
			name:   "final EOS",
			step:   InferenceStep{JobID: "job", SequenceID: "seq", FinalShard: true},
			result: InferenceStepResult{JobID: "job", SequenceID: "seq", EndOfSequence: true},
		},
		{
			name:    "mismatched token position",
			step:    InferenceStep{JobID: "job", SequenceID: "seq", Position: 2, FinalShard: true},
			result:  InferenceStepResult{JobID: "job", SequenceID: "seq", Position: 1, SampledTokenID: &token},
			wantErr: true,
		},
		{
			name:    "final without token or EOS",
			step:    InferenceStep{JobID: "job", SequenceID: "seq", FinalShard: true},
			result:  InferenceStepResult{JobID: "job", SequenceID: "seq"},
			wantErr: true,
		},
		{
			name:    "intermediate token rejected",
			step:    InferenceStep{JobID: "job", SequenceID: "seq", FinalShard: false},
			result:  InferenceStepResult{JobID: "job", SequenceID: "seq", SampledTokenID: &token},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.result.ValidateFor(test.step)
			if (err != nil) != test.wantErr {
				t.Fatalf("ValidateFor() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func testActivation() *activation.Envelope {
	a := activation.NewEnvelope("job", "job", "seq", "model", "v1", "a", "b", 0, "uint8", []int64{1}, []byte{1})
	return &a
}
