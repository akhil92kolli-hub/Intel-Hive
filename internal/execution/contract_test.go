package execution

import (
	"strings"
	"testing"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

const digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func request(mode Mode, pass uint64, offset, count uint32) Request {
	spec := TensorSpec{DType: F16, Shape: []uint32{count, 2}, Layout: RowMajorContiguous, ByteOrder: LittleEndian, ByteLength: uint64(count) * 4}
	shard := ShardSpec{ID: "s0", ModelID: "qwen", ModelVersion: "1", ModelArtifactDigest: digest, Layers: model.LayerRange{Start: 0, End: 11}, Input: spec, Output: spec}
	tokens := make([]int32, count)
	return Request{RequestID: "r", SequenceID: "s", ModelID: "qwen", ModelVersion: "1", ModelArtifactDigest: digest, Shard: shard, PassOrdinal: pass, Mode: mode, KVTokenOffset: offset, TokenCount: count, Input: ShardInput{Tokens: tokens}}
}

func TestPrefillAndDecodeUseIndependentPassAndKVOffsets(t *testing.T) {
	prefill := request(Prefill, 0, 0, 37)
	if err := prefill.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (Result{RequestID: "r", SequenceID: "s", PassOrdinal: 0, KVTokenOffsetBefore: 0, KVTokenOffsetAfter: 37, Output: Output{Type: OutputActivation}}).ValidateFor(prefill, false); err != nil {
		t.Fatal(err)
	}
	decode := request(Decode, 1, 37, 1)
	if err := decode.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (Result{RequestID: "r", SequenceID: "s", PassOrdinal: 1, KVTokenOffsetBefore: 37, KVTokenOffsetAfter: 38, Output: Output{Type: OutputToken}}).ValidateFor(decode, true); err != nil {
		t.Fatal(err)
	}
	secondDecode := request(Decode, 2, 38, 1)
	if err := (Result{RequestID: "r", SequenceID: "s", PassOrdinal: 2, KVTokenOffsetBefore: 38, KVTokenOffsetAfter: 39, Output: Output{Type: OutputToken}}).ValidateFor(secondDecode, true); err != nil {
		t.Fatal(err)
	}
}

func TestValidateNextRejectsInvalidKVOffset(t *testing.T) {
	previous := Result{RequestID: "r", SequenceID: "s", PassOrdinal: 0, KVTokenOffsetAfter: 37}
	if err := ValidateNext(previous, request(Decode, 1, 100, 1)); err == nil || !strings.Contains(err.Error(), "KV token offset") {
		t.Fatalf("expected non-contiguous KV offset failure, got %v", err)
	}
}

func TestTensorValidatesExactF16ByteLength(t *testing.T) {
	const byteLength = 1 * 37 * 2048 * 2
	tensor := Tensor{DType: DTypeF16, Shape: []uint32{1, 37, 2048}, Layout: LayoutRowMajorContiguous, ByteOrder: ByteOrderLittleEndian, ByteLength: byteLength, Data: make([]byte, byteLength)}
	if err := tensor.Validate(); err != nil {
		t.Fatal(err)
	}
	tensor.ByteLength--
	if err := tensor.Validate(); err == nil {
		t.Fatal("expected byte-length mismatch")
	}
}

func TestRejectsPassOrdinalAsKVOffset(t *testing.T) {
	req := request(Decode, 1, 37, 1)
	err := (Result{RequestID: "r", SequenceID: "s", PassOrdinal: 1, KVTokenOffsetBefore: 1, KVTokenOffsetAfter: 2, Output: Output{Type: OutputActivation}}).ValidateFor(req, false)
	if err == nil || !strings.Contains(err.Error(), "KV token range") {
		t.Fatalf("expected KV range failure, got %v", err)
	}
}
