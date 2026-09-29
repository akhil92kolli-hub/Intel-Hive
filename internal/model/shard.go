package model

import (
	"fmt"
)

// LayerRange describes an inclusive range of model layers.
type LayerRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

func (r LayerRange) Size() int {
	if r.End < r.Start {
		return 0
	}
	return r.End - r.Start + 1
}

func (r LayerRange) Contains(layer int) bool {
	return layer >= r.Start && layer <= r.End
}

// ModelMetadata is the minimum metadata structure required for layer planning.
type ModelMetadata struct {
	ID             string
	Architecture   string
	ParameterCount int64
	LayerCount     int
	HiddenSize     int
	ContextLength  int
	Quantization   string
}

// Layer describes a single logical layer in a model.
type Layer struct {
	Index       int
	MemoryBytes int64
	Tensors     []Tensor
}

// Tensor describes a model tensor used inside a layer.
type Tensor struct {
	Name      string
	Shape     []int64
	DataType  string
	SizeBytes int64
}

// Model describes the full logical model.
type Model struct {
	ID             string
	Version        string
	ArtifactDigest string

	Metadata ModelMetadata
	Layers   []Layer
}

// ModelShard describes a contiguous set of layers assigned to a worker.
type ModelShard struct {
	ID             string
	ModelID        string
	Version        string
	ArtifactDigest string
	Layers         LayerRange
	MemoryBytes    int64
}

// NewModelShard creates a valid shard for a contiguous range.
func NewModelShard(model *Model, start int, end int) (*ModelShard, error) {
	if start < 0 {
		return nil, fmt.Errorf("invalid start layer: %d", start)
	}
	if end < start {
		return nil, fmt.Errorf("invalid layer range: %d-%d", start, end)
	}
	if model == nil {
		return nil, fmt.Errorf("model is nil")
	}
	if end >= len(model.Layers) {
		return nil, fmt.Errorf("layer %d exceeds model layer count %d", end, len(model.Layers))
	}

	var memory int64
	for i := start; i <= end; i++ {
		memory += model.Layers[i].MemoryBytes
	}

	return &ModelShard{
		ID:             fmt.Sprintf("%s-%d-%d", model.ID, start, end),
		ModelID:        model.ID,
		Version:        model.Version,
		ArtifactDigest: model.ArtifactDigest,
		Layers:         LayerRange{Start: start, End: end},
		MemoryBytes:    memory,
	}, nil
}

// ShardPlanner splits a model into logical ranges.
type ShardPlanner struct{}

func (p *ShardPlanner) Split(model *Model, ranges []LayerRange) ([]ModelShard, error) {
	if model == nil {
		return nil, fmt.Errorf("model is nil")
	}

	if err := ValidatePartition(len(model.Layers), ranges); err != nil {
		return nil, err
	}

	shards := make([]ModelShard, 0, len(ranges))
	for _, r := range ranges {
		shard, err := NewModelShard(model, r.Start, r.End)
		if err != nil {
			return nil, err
		}
		shards = append(shards, *shard)
	}
	return shards, nil
}

// ValidatePartition ensures ranges are contiguous and non-overlapping.
func ValidatePartition(layerCount int, ranges []LayerRange) error {
	if len(ranges) == 0 {
		return fmt.Errorf("no layer ranges provided")
	}
	if layerCount <= 0 {
		return fmt.Errorf("invalid layer count: %d", layerCount)
	}

	expected := 0
	for _, r := range ranges {
		if r.Start < 0 || r.End < r.Start {
			return fmt.Errorf("invalid layer range: %+v", r)
		}
		if r.Start != expected {
			return fmt.Errorf("partition gap or overlap detected; expected start %d, got %d", expected, r.Start)
		}
		if r.End >= layerCount {
			return fmt.Errorf("range %+v exceeds model layer count %d", r, layerCount)
		}
		expected = r.End + 1
	}

	if expected != layerCount {
		return fmt.Errorf("partition covers %d layers, but model has %d", expected, layerCount)
	}
	return nil
}
