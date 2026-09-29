// Package execution defines the backend-facing shard execution contract.
//
// Transport packages adapt WebSocket/protobuf messages into these values. No
// native backend should accept a transport DTO directly.
package execution

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

type ExecutionMode string

const (
	ExecutionPrefill ExecutionMode = "prefill"
	ExecutionDecode  ExecutionMode = "decode"
)

// Mode and its values remain aliases while callers migrate to the explicit
// execution terminology.
type Mode = ExecutionMode

const (
	Prefill = ExecutionPrefill
	Decode  = ExecutionDecode
)

type TensorDType string

const (
	DTypeF16 TensorDType = "f16"
	DTypeF32 TensorDType = "f32"
	F16                  = DTypeF16
	F32                  = DTypeF32
)

type TensorLayout string

const LayoutRowMajorContiguous TensorLayout = "row_major_contiguous"

const RowMajorContiguous = LayoutRowMajorContiguous

type ByteOrder string

const ByteOrderLittleEndian ByteOrder = "little_endian"

const LittleEndian = ByteOrderLittleEndian

// TensorSpec is deliberately constrained for M1. Arbitrary strides and
// platform-dependent byte order are excluded from the wire contract.
type TensorSpec struct {
	DType      TensorDType
	Shape      []uint32
	Layout     TensorLayout
	ByteOrder  ByteOrder
	ByteLength uint64
}

// Tensor is the canonical M1 activation representation. The M1 contract has
// no strides: data must be row-major, contiguous, and little-endian.
type Tensor struct {
	DType      TensorDType
	Shape      []uint32
	ByteLength uint64
	Layout     TensorLayout
	ByteOrder  ByteOrder
	Data       []byte
}

func (t Tensor) Spec() TensorSpec {
	return TensorSpec{DType: t.DType, Shape: append([]uint32(nil), t.Shape...), Layout: t.Layout, ByteOrder: t.ByteOrder, ByteLength: t.ByteLength}
}

func (t Tensor) Validate() error {
	if err := t.Spec().Validate(); err != nil {
		return err
	}
	if uint64(len(t.Data)) != t.ByteLength {
		return fmt.Errorf("tensor data length does not match byte length")
	}
	return nil
}

type Activation struct{ Tensor Tensor }

func (a Activation) Validate() error { return a.Tensor.Validate() }

func (s TensorSpec) Validate() error {
	if s.DType != DTypeF16 && s.DType != DTypeF32 {
		return fmt.Errorf("unsupported tensor dtype %q", s.DType)
	}
	if s.Layout != LayoutRowMajorContiguous {
		return fmt.Errorf("unsupported tensor layout %q", s.Layout)
	}
	if s.ByteOrder != ByteOrderLittleEndian {
		return fmt.Errorf("unsupported tensor byte order %q", s.ByteOrder)
	}
	if len(s.Shape) == 0 {
		return fmt.Errorf("tensor shape is required")
	}
	bytes := uint64(2)
	if s.DType == DTypeF32 {
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

type ShardInput struct {
	Activation *Activation
	Tokens     []int32
}

func (i ShardInput) Validate(tokenCount uint32) error {
	if (i.Activation == nil) == (len(i.Tokens) == 0) {
		return fmt.Errorf("exactly one of activation or tokens is required")
	}
	if i.Activation != nil {
		return i.Activation.Validate()
	}
	if len(i.Tokens) != int(tokenCount) {
		return fmt.Errorf("token count does not match token input")
	}
	return nil
}

type Input = ShardInput

type ShardExecutionRequest struct {
	RequestID           string
	SequenceID          string
	ModelID             string
	ModelVersion        string
	ModelArtifactDigest string
	Shard               ShardSpec
	PassOrdinal         uint64
	Mode                ExecutionMode
	KVTokenOffset       uint32
	TokenCount          uint32
	Input               ShardInput
}

type Request = ShardExecutionRequest

func (r ShardExecutionRequest) Validate() error {
	if strings.TrimSpace(r.RequestID) == "" || strings.TrimSpace(r.SequenceID) == "" {
		return fmt.Errorf("request and sequence IDs are required")
	}
	if r.Mode != ExecutionPrefill && r.Mode != ExecutionDecode {
		return fmt.Errorf("unsupported execution mode %q", r.Mode)
	}
	if r.TokenCount == 0 {
		return fmt.Errorf("token count must be positive")
	}
	if r.ModelID != r.Shard.ModelID || r.ModelVersion != r.Shard.ModelVersion || r.ModelArtifactDigest != r.Shard.ModelArtifactDigest {
		return fmt.Errorf("request model identity does not match shard")
	}
	if err := r.Shard.Validate(); err != nil {
		return err
	}
	return r.Input.Validate(r.TokenCount)
}

type OutputType string

const (
	OutputActivation OutputType = "activation"
	OutputLogits     OutputType = "logits"
	OutputToken      OutputType = "token"
	OutputEOS        OutputType = "eos"
)

type ShardOutput struct {
	Type       OutputType
	Activation *Activation
	Tensor     *Tensor
	TokenID    *uint32
}

type Output = ShardOutput

type ShardExecutionResult struct {
	RequestID           string
	SequenceID          string
	PassOrdinal         uint64
	KVTokenOffsetBefore uint32
	KVTokenOffsetAfter  uint32
	Output              ShardOutput
}

type Result = ShardExecutionResult

func (r ShardExecutionResult) ValidateFor(request ShardExecutionRequest, finalShard bool) error {
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

// ShardExecutor is the backend boundary. Transport adapters are responsible
// for constructing and validating ShardExecutionRequest before this call.
type ShardExecutor interface {
	Execute(ctx context.Context, req ShardExecutionRequest) (ShardExecutionResult, error)
}

// ValidateNext enforces sequence-local pass and KV continuity. A backend can
// call it with its retained prior result before it touches native state.
func ValidateNext(previous ShardExecutionResult, next ShardExecutionRequest) error {
	if previous.RequestID != next.RequestID || previous.SequenceID != next.SequenceID {
		return fmt.Errorf("request or sequence does not match prior result")
	}
	if next.PassOrdinal != previous.PassOrdinal+1 {
		return fmt.Errorf("pass ordinal is not contiguous")
	}
	if next.KVTokenOffset != previous.KVTokenOffsetAfter {
		return fmt.Errorf("KV token offset is not contiguous")
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
