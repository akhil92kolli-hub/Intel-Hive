package activation

import (
	"encoding/json"
	"testing"
)

func TestEnvelopeJSONRoundTripUsesStringSequenceAndBase64Payload(t *testing.T) {
	original := NewEnvelope(
		"job-1",
		"request-1",
		"job-1:attempt-0",
		"qwen2.5-3b-instruct",
		"q4_k_m-v1",
		"worker-a",
		"worker-b",
		9,
		"float32",
		[]int64{1, 1, 1},
		[]byte{1, 2, 3, 4},
	)

	encoded, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal activation: %v", err)
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatalf("unmarshal activation fields: %v", err)
	}
	var sequenceID string
	if err := json.Unmarshal(fields["sequence_id"], &sequenceID); err != nil {
		t.Fatalf("sequence_id is not a JSON string: %v", err)
	}
	if sequenceID != original.SequenceID {
		t.Fatalf("sequence_id = %q, want %q", sequenceID, original.SequenceID)
	}

	var decoded Envelope
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal activation: %v", err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("validate round-tripped activation: %v", err)
	}
	if decoded.ModelID != original.ModelID || decoded.ModelVersion != original.ModelVersion {
		t.Fatal("model identity changed during JSON round trip")
	}
	if string(decoded.Payload) != string(original.Payload) {
		t.Fatal("payload changed during JSON round trip")
	}
}

func TestEnvelopeRejectsCorruptPayload(t *testing.T) {
	envelope := NewEnvelope(
		"job-1", "request-1", "sequence-1", "model-1", "v1",
		"worker-a", "worker-b", 3, "float16", []int64{1, 2}, []byte{1, 2, 3, 4},
	)
	envelope.Payload[0] ^= 0xff
	if err := envelope.Validate(); err == nil {
		t.Fatal("expected checksum validation failure")
	}
}
