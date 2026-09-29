package execution

import (
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
	"strings"
	"testing"
)

const digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func request(mode Mode, pass uint64, offset, count uint32) Request {
	spec := TensorSpec{DType: F16, Shape: []uint32{count, 2}, Layout: RowMajorContiguous, ByteOrder: LittleEndian, ByteLength: uint64(count) * 4}
	shard := ShardSpec{ID: "s0", ModelID: "qwen", ModelVersion: "1", ModelArtifactDigest: digest, Layers: model.LayerRange{Start: 0, End: 11}, Input: spec, Output: spec}
	return Request{RequestID: "r", SequenceID: "s", ModelID: "qwen", ModelVersion: "1", ModelArtifactDigest: digest, Shard: shard, PassOrdinal: pass, Mode: mode, KVTokenOffset: offset, TokenCount: count}
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
}

func TestRejectsPassOrdinalAsKVOffset(t *testing.T) {
	req := request(Decode, 1, 37, 1)
	err := (Result{RequestID: "r", SequenceID: "s", PassOrdinal: 1, KVTokenOffsetBefore: 1, KVTokenOffsetAfter: 2, Output: Output{Type: OutputActivation}}).ValidateFor(req, false)
	if err == nil || !strings.Contains(err.Error(), "KV token range") {
		t.Fatalf("expected KV range failure, got %v", err)
	}
}
