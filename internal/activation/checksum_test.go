package activation

import (
	"crypto/sha256"
	"fmt"
	"testing"
)

func TestValidateChecksum(t *testing.T) {
	payload := []byte("activation")
	digest := sha256.Sum256(payload)
	checksum := fmt.Sprintf("sha256:%x", digest[:])

	if err := ValidateChecksum(payload, checksum); err != nil {
		t.Fatalf("valid checksum rejected: %v", err)
	}
	if err := ValidateChecksum(payload, "sha256:invalid"); err == nil {
		t.Fatal("invalid checksum accepted")
	}
}
