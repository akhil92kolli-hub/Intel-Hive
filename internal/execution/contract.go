// Package execution defines the backend-facing shard execution contract.
//
// Transport packages adapt WebSocket/protobuf messages into these values. No
// native backend should accept a transport DTO directly.
package execution

import (
	"fmt"
	"math"
	"strings"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

type Mode string

const (
	Prefill Mode = "prefill"
	Decode  Mode = "decode"
)

type TensorDType string

const (
	F16 TensorDType = "F16"
	F32 TensorDType = "F32"
)

type TensorLayout string

const RowMajorContiguous TensorLayout = "ROW_MAJOR_CONTIGUOUS"

type ByteOrder string

const LittleEndian ByteOrder = "LITTLE_ENDIAN"

// TensorSpec is deliberately constrained for M1. Arbitrary strides and
// platform-dependent byte order are excluded from the wire contract.
type TensorSpec struct {
	DType      TensorDType
	Shape      []uint32
	Layout     TensorLayout
	ByteOrder  ByteOrder
	ByteLength uint64
}

func (s TensorSpec) Validate() error {
	if s.DType != F16 && s.DType != F32 {
		return fmt.Errorf("unsupported tensor dtype %q", s.DType)
	}
	if s.Layout != RowMajorContiguous {
		return fmt.Errorf("unsupported tensor layout %q", s.Layout)
	}
	if s.ByteOrder != LittleEndian {
		return fmt.Errorf("unsupported tensor byte order %q", s.ByteOrder)
	}
	if len(s.Shape) == 0 {
		return fmt.Errorf("tensor shape is required")
	}
	bytes := uint64(2)
	if s.DType == F32 {
		bytes = 4
	}
	for _, dimension := range s.Shape {
		if dimension == 0 || bytes > math.MaxUint64/uint64(dimension) {
			return fmt.Errorf("invalid tensor shape")
		}
		bytes *= uint64(dimension)
	}
	if s.ByteLength != bytes {
		return fmt.Errorf("tensor byte length %d does not match shape and dtype (%d)", s.ByteLength, bytes)
	}
	return nil
}

type ShardSpec struct {
	ID                  string
	ModelID             string
	ModelVersion        string
	ModelArtifactDigest string
	Layers              model.LayerRange
	Input               TensorSpec
	Output              TensorSpec
}

func (s ShardSpec) Validate() error {
	if strings.TrimSpace(s.ID) == "" || strings.TrimSpace(s.ModelID) == "" || strings.TrimSpace(s.ModelVersion) == "" {
		return fmt.Errorf("shard identity is required")
	}
	if !validDigest(s.ModelArtifactDigest) {
		return fmt.Errorf("shard artifact digest must be sha256:<64 hex characters>")
	}
	if s.Layers.Start < 0 || s.Layers.End < s.Layers.Start {
		return fmt.Errorf("invalid shard layer range")
	}
	if err := s.Input.Validate(); err != nil {
		return fmt.Errorf("input tensor: %w", err)
	}
	return s.Output.Validate()
}

type Input struct {
	TokenIDs   []uint32
	Activation []byte
}

type Request struct {
	RequestID           string
	SequenceID          string
	ModelID             string
	ModelVersion        string
	ModelArtifactDigest string
	Shard               ShardSpec
	PassOrdinal         uint64
	Mode                Mode
	KVTokenOffset       uint32
	TokenCount          uint32
	Input               Input
}

func (r Request) Validate() error {
	if strings.TrimSpace(r.RequestID) == "" || strings.TrimSpace(r.SequenceID) == "" {
		return fmt.Errorf("request and sequence IDs are required")
	}
	if r.Mode != Prefill && r.Mode != Decode {
		return fmt.Errorf("unsupported execution mode %q", r.Mode)
	}
	if r.TokenCount == 0 {
		return fmt.Errorf("token count must be positive")
	}
	if r.Mode == Decode && r.TokenCount != 1 {
		return fmt.Errorf("decode must process exactly one token")
	}
	if r.ModelID != r.Shard.ModelID || r.ModelVersion != r.Shard.ModelVersion || r.ModelArtifactDigest != r.Shard.ModelArtifactDigest {
		return fmt.Errorf("request model identity does not match shard")
	}
	if err := r.Shard.Validate(); err != nil {
		return err
	}
	return nil
}

type OutputType string

const (
	OutputActivation OutputType = "activation"
	OutputLogits     OutputType = "logits"
	OutputToken      OutputType = "token"
	OutputEOS        OutputType = "eos"
)

type Output struct {
	Type    OutputType
	Tensor  []byte
	TokenID *uint32
}

type Result struct {
	RequestID           string
	SequenceID          string
	PassOrdinal         uint64
	KVTokenOffsetBefore uint32
	KVTokenOffsetAfter  uint32
	Output              Output
}

func (r Result) ValidateFor(request Request, finalShard bool) error {
	if r.RequestID != request.RequestID || r.SequenceID != request.SequenceID || r.PassOrdinal != request.PassOrdinal {
		return fmt.Errorf("result identity does not match request")
	}
	if r.KVTokenOffsetBefore != request.KVTokenOffset || r.KVTokenOffsetAfter != request.KVTokenOffset+request.TokenCount {
		return fmt.Errorf("result KV token range is not contiguous")
	}
	if finalShard {
		if r.Output.Type != OutputLogits && r.Output.Type != OutputToken && r.Output.Type != OutputEOS {
			return fmt.Errorf("final shard must return logits, token, or eos")
		}
	} else if r.Output.Type != OutputActivation {
		return fmt.Errorf("intermediate shard must return an activation")
	}
	return nil
}

func validDigest(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return false
	}
	for _, r := range value[len("sha256:"):] {
		if !(r >= '0' && r <= '9' || r >= 'a' && r <= 'f') {
			return false
		}
	}
	return true
}
