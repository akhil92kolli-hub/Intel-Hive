# T004: Model Metadata Parser

## Objective

Extract the minimum metadata required to build a distributed model description without yet forcing a full GGUF runtime integration.

The system should be able to answer:

- What is the model architecture?
- How many layers does it have?
- How large is the context window?
- What is the embedding size?
- What is the quantization format?
- How should the model be partitioned into shards?

## Design Goal

This is a metadata-first layer, not a runtime loader. We are not implementing full GGUF tensor evaluation yet. We are only extracting enough information to construct a valid distributed model description.

```text
ModelMetadata
  ├── model_id
  ├── architecture
  ├── parameter_count
  ├── context_length
  ├── embedding_length
  ├── layer_count
  ├── attention_heads
  ├── kv_heads
  ├── vocab_size
  ├── tensor_count
  └── quantization
```

## Qwen2.5 3B Requirements

For Qwen2.5-3B-Instruct, the eventual metadata should support:

```text
layer_count = 36
embedding_length = 3072
attention_heads = 24
kv_heads = 8
context_length = 32768
vocab_size = 152064
architecture = qwen2
quantization = Q4_K_M
```

This lets the system create:

```text
Shard 0 = 0–11
Shard 1 = 12–23
Shard 2 = 24–35
```

## Phase-0 Abstraction

```go
type ModelMetadata struct {
  ModelID         string
  Architecture    string
  ParameterCount  uint64
  ContextLength   uint32
  EmbeddingLength uint32
  LayerCount      uint32
  AttentionHeads  uint32
  KVHeads         uint32
  VocabSize       uint32
  TensorCount     uint32
  Quantization    string
}

type ModelShard struct {
  ModelID     string
  ShardID     string
  LayerStart  uint32
  LayerEnd    uint32
  Tensors     []string
}

type Model struct {
  Metadata ModelMetadata
  Shards   []ModelShard
}
```

## Parsing Approach

A JSON or metadata manifest can be used during Phase-0. The parser should only read the minimum fields needed to build `Model` and `ModelShard` objects.

This can be implemented in two stages:

### Stage 1: Manifest-based parser

A static JSON file with explicit metadata. This is the fastest route for Phase-0 and allows the distributed scheduler to operate before mobile runtime support is complete.

### Stage 2: GGUF metadata extraction

Later, a dedicated GGUF parser can read the model header and extract the same fields directly from the file.

## Example JSON

```json
{
  "model_id": "qwen2.5-3b",
  "architecture": "qwen2",
  "parameter_count": 3000000000,
  "context_length": 32768,
  "embedding_length": 3072,
  "layer_count": 36,
  "attention_heads": 24,
  "kv_heads": 8,
  "vocab_size": 152064,
  "tensor_count": 512,
  "quantization": "Q4_K_M"
}
```

## Model Sharding Rules

Given `layer_count = 36`, default Phase-0 sharding can be:

```go
func DefaultShardPlan(layerCount uint32) []ModelShard {
  // 3 shards
  // maintain even division as much as possible
  // if layer_count % 3 != 0, use remainder distribution to the last shard
}
```

Example output:

```text
Shard 0 = 0..11
Shard 1 = 12..23
Shard 2 = 24..35
```

## File Structure

```text
models/
  manifests/
    qwen2.5-3b.json
  metadata/
    parser.go
    model.go
```

## Deliverables

- ✅ `ModelMetadata` struct
- ✅ `ModelShard` struct
- ✅ `Model` abstraction
- ✅ sharding helper for a 36-layer model
- ✅ manifest example for Qwen2.5-3B
- ✅ parser entry point that returns a `Model` instance
- ✅ no full GGUF loader yet

## Next step

T005: Layer-Range Model Abstraction

This will focus on the model object lifecycle and how layer ranges map to shards without yet demanding runtime tensor loading.
