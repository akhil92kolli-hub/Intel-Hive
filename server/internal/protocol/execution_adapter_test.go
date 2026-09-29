package protocol

import "testing"

const executionDigest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func TestBuildExecutionRequestPreservesPrefillKVRange(t *testing.T) {
	descriptor := &TensorDescriptor{DType: "f16", Shape: []int64{37, 2}, Layout: "row_major_contiguous", ByteOrder: "little_endian", ByteLength: 148}
	request, err := BuildExecutionRequest(JobAssignment{RequestID: "request", JobID: "job", SequenceID: "sequence", ModelID: "qwen", ModelVersion: "1", ModelArtifactDigest: executionDigest, ShardID: "s0", LayerStart: 0, LayerEnd: 11, Phase: InferencePhasePrefill, PassOrdinal: 0, KVTokenOffset: 0, TokenCount: 37, InputTokenIDs: make([]uint32, 37), InputTensor: descriptor, OutputTensor: descriptor})
	if err != nil {
		t.Fatal(err)
	}
	if request.PassOrdinal != 0 || request.KVTokenOffset != 0 || request.TokenCount != 37 {
		t.Fatalf("execution metadata changed: %+v", request)
	}
}
