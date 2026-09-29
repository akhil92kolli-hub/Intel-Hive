package protocol

import (
	"fmt"
	"math"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/execution"
	"github.com/akhil92kolli-hub/Intel-Hive/internal/model"
)

// BuildExecutionRequest is the server-side transport boundary. It is the only
// place a WebSocket assignment is converted into a backend execution request.
func BuildExecutionRequest(assignment JobAssignment) (execution.ShardExecutionRequest, error) {
	mode, err := executionMode(assignment.Phase)
	if err != nil {
		return execution.ShardExecutionRequest{}, err
	}
	inputSpec, err := tensorSpec(assignment.InputTensor)
	if err != nil {
		return execution.ShardExecutionRequest{}, fmt.Errorf("input tensor: %w", err)
	}
	outputSpec, err := tensorSpec(assignment.OutputTensor)
	if err != nil {
		return execution.ShardExecutionRequest{}, fmt.Errorf("output tensor: %w", err)
	}
	input, err := shardInput(assignment)
	if err != nil {
		return execution.ShardExecutionRequest{}, err
	}
	req := execution.ShardExecutionRequest{
		RequestID: assignment.RequestID, SequenceID: assignment.SequenceID,
		ModelID: assignment.ModelID, ModelVersion: assignment.ModelVersion,
		ModelArtifactDigest: assignment.ModelArtifactDigest,
		Shard: execution.ShardSpec{ID: assignment.ShardID, ModelID: assignment.ModelID,
			ModelVersion: assignment.ModelVersion, ModelArtifactDigest: assignment.ModelArtifactDigest,
			Layers: model.LayerRange{Start: assignment.LayerStart, End: assignment.LayerEnd}, Input: inputSpec, Output: outputSpec},
		PassOrdinal: assignment.PassOrdinal, KVTokenOffset: assignment.KVTokenOffset,
		TokenCount: assignment.TokenCount, Mode: mode, Input: input,
	}
	if err := req.Validate(); err != nil {
		return execution.ShardExecutionRequest{}, err
	}
	return req, nil
}

func executionMode(phase InferencePhase) (execution.ExecutionMode, error) {
	switch phase {
	case InferencePhasePrefill:
		return execution.ExecutionPrefill, nil
	case InferencePhaseDecode:
		return execution.ExecutionDecode, nil
	default:
		return "", fmt.Errorf("unsupported execution phase %q", phase)
	}
}

func tensorSpec(value *TensorDescriptor) (execution.TensorSpec, error) {
	if value == nil {
		return execution.TensorSpec{}, fmt.Errorf("is required")
	}
	shape := make([]uint32, len(value.Shape))
	for i, dimension := range value.Shape {
		if dimension <= 0 || dimension > math.MaxUint32 {
			return execution.TensorSpec{}, fmt.Errorf("invalid shape")
		}
		shape[i] = uint32(dimension)
	}
	spec := execution.TensorSpec{DType: execution.TensorDType(value.DType), Shape: shape,
		Layout: execution.TensorLayout(value.Layout), ByteOrder: execution.ByteOrder(value.ByteOrder), ByteLength: value.ByteLength}
	return spec, spec.Validate()
}

func shardInput(assignment JobAssignment) (execution.ShardInput, error) {
	if assignment.Activation != nil {
		if err := assignment.Activation.ValidateCanonical(); err != nil {
			return execution.ShardInput{}, err
		}
		spec, err := tensorSpec(&TensorDescriptor{DType: lowerDType(assignment.Activation.DType), Shape: assignment.Activation.Shape,
			Layout: lowerLayout(assignment.Activation.Layout), ByteOrder: lowerByteOrder(assignment.Activation.ByteOrder), ByteLength: assignment.Activation.ByteLength})
		if err != nil {
			return execution.ShardInput{}, err
		}
		return execution.ShardInput{Activation: &execution.Activation{Tensor: execution.Tensor{DType: spec.DType, Shape: spec.Shape, Layout: spec.Layout, ByteOrder: spec.ByteOrder, ByteLength: spec.ByteLength, Data: append([]byte(nil), assignment.Activation.Payload...)}}}, nil
	}
	if len(assignment.InputTokenIDs) == 0 {
		return execution.ShardInput{}, fmt.Errorf("execution requires input token IDs or activation")
	}
	tokens := make([]int32, len(assignment.InputTokenIDs))
	for i, token := range assignment.InputTokenIDs {
		if token > math.MaxInt32 {
			return execution.ShardInput{}, fmt.Errorf("token ID exceeds int32 range")
		}
		tokens[i] = int32(token)
	}
	return execution.ShardInput{Tokens: tokens}, nil
}

func lowerDType(value string) string {
	if value == "F16" {
		return "f16"
	}
	if value == "F32" {
		return "f32"
	}
	return value
}
func lowerLayout(value string) string {
	if value == "ROW_MAJOR_CONTIGUOUS" {
		return "row_major_contiguous"
	}
	return value
}
func lowerByteOrder(value string) string {
	if value == "LITTLE_ENDIAN" {
		return "little_endian"
	}
	return value
}
