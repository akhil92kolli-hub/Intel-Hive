package protocol

import (
	"encoding/json"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
	"testing"
)

func TestJobAssignmentActivationUsesSharedEnvelopeJSON(t *testing.T) {
	envelope := activation.NewEnvelope(
		"job-1", "request-1", "job-1:attempt-0", "model-1", "v1",
		"worker-a", "worker-b", 9, "uint8", []int64{3}, []byte{1, 2, 3},
	)
	envelope.Position = 4
	assignment := JobAssignment{
		RequestID: "request-1", ModelVersion: "v1", WorkerID: "worker-b",
		PreviousWorker: "worker-a", NextWorker: "worker-c",
		AssignmentID: "assignment-1", JobID: "job-1", ModelID: "model-1",
		ShardID: "shard-10-19", Phase: InferencePhaseDecode,
		SequenceID: "job-1:attempt-0", Position: 4, Activation: &envelope,
		LayerStart: 10, LayerEnd: 19,
	}

	encoded, err := json.Marshal(assignment)
	if err != nil {
		t.Fatalf("marshal websocket assignment: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatalf("unmarshal assignment fields: %v", err)
	}
	var activationFields map[string]json.RawMessage
	if err := json.Unmarshal(fields["activation"], &activationFields); err != nil {
		t.Fatalf("activation must be a JSON object: %v", err)
	}
	var sequenceID string
	if err := json.Unmarshal(activationFields["sequence_id"], &sequenceID); err != nil {
		t.Fatalf("activation sequence_id must be a string: %v", err)
	}
	if sequenceID != assignment.SequenceID || string(activationFields["payload"]) != `"AQID"` {
		t.Fatalf("unexpected activation JSON: %s", fields["activation"])
	}
	var decoded JobAssignment
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal websocket assignment: %v", err)
	}
	if decoded.Activation == nil {
		t.Fatal("activation envelope was omitted")
	}
	if err := decoded.Activation.ValidateBoundary(
		assignment.JobID, assignment.RequestID, assignment.SequenceID,
		assignment.ModelID, assignment.ModelVersion, assignment.PreviousWorker,
		assignment.WorkerID, assignment.LayerStart-1, assignment.Position,
	); err != nil {
		t.Fatalf("activation envelope did not survive websocket round trip: %v", err)
	}
	if string(decoded.Activation.Payload) != string([]byte{1, 2, 3}) {
		t.Fatalf("activation payload changed: %v", decoded.Activation.Payload)
	}
}

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
