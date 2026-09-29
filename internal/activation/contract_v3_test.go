package activation

import "testing"

func TestCanonicalEnvelopeRequiresDeterministicTensorMetadata(t *testing.T) {
	a := NewEnvelope("job", "request", "sequence", "model", "1", "a", "b", 11, "F16", []int64{1, 2}, []byte{0, 0, 0, 0})
	a.ModelArtifactDigest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	a.TokenCount = 1
	a.Layout = "ROW_MAJOR_CONTIGUOUS"
	a.ByteOrder = "LITTLE_ENDIAN"
	a.ByteLength = 4
	if err := a.ValidateCanonical(); err != nil {
		t.Fatal(err)
	}
	a.ByteLength = 3
	if err := a.ValidateCanonical(); err == nil {
		t.Fatal("expected byte length validation failure")
	}
}
