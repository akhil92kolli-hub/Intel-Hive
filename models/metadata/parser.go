package metadata

import (
    "encoding/json"
    "fmt"
    "os"
)

// ModelMetadata stores the minimum required metadata for distributed model planning.
type ModelMetadata struct {
    ModelID         string `json:"model_id"`
    Architecture    string `json:"architecture"`
    ParameterCount  uint64 `json:"parameter_count"`
    ContextLength   uint32 `json:"context_length"`
    EmbeddingLength uint32 `json:"embedding_length"`
    LayerCount      uint32 `json:"layer_count"`
    AttentionHeads  uint32 `json:"attention_heads"`
    KVHeads         uint32 `json:"kv_heads"`
    VocabSize       uint32 `json:"vocab_size"`
    TensorCount     uint32 `json:"tensor_count"`
    Quantization    string `json:"quantization"`
}

// ModelShard describes a contiguous layer range.
type ModelShard struct {
    ModelID    string `json:"model_id"`
    ShardID    string `json:"shard_id"`
    LayerStart uint32 `json:"layer_start"`
    LayerEnd   uint32 `json:"layer_end"`
    Tensors    []string `json:"tensors,omitempty"`
}

// Model describes the full model and all shard definitions.
type Model struct {
    Metadata ModelMetadata `json:"metadata"`
    Shards   []ModelShard `json:"shards"`
}

// LoadFromFile reads a model metadata manifest and returns a Model instance.
func LoadFromFile(path string) (Model, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Model{}, fmt.Errorf("read model metadata: %w", err)
    }

    var meta ModelMetadata
    if err := json.Unmarshal(data, &meta); err != nil {
        return Model{}, fmt.Errorf("decode model metadata: %w", err)
    }

    return BuildModel(meta), nil
}

// BuildModel constructs a model plus shard layout using default Phase-0 planning.
func BuildModel(meta ModelMetadata) Model {
    if meta.LayerCount == 0 {
        meta.LayerCount = 36
    }

    shards := make([]ModelShard, 0, 3)
    shardCount := 3
    layersPerShard := int(meta.LayerCount) / shardCount
    remainder := int(meta.LayerCount) % shardCount

    start := uint32(0)
    for i := 0; i < shardCount; i++ {
        extra := 0
        if i == shardCount-1 {
            extra = remainder
        }
        count := layersPerShard + extra
        end := start + uint32(count-1)
        if count == 0 {
            break
        }

        shards = append(shards, ModelShard{
            ModelID:    meta.ModelID,
            ShardID:    fmt.Sprintf("%s-s%d", meta.ModelID, i),
            LayerStart: start,
            LayerEnd:   end,
        })
        start = end + 1
    }

    return Model{
        Metadata: meta,
        Shards:   shards,
    }
}
