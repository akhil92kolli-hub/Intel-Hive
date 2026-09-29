# T005: Layer-Range Model Abstraction

## Objective

Create the model abstraction that sits between raw model metadata and the distributed scheduler.

The key concept is:

> A model is one logical object, and a shard is a contiguous range of layers that a worker can execute.

This task deliberately keeps the abstraction logical and clean before runtime loading is implemented.

## Core Types

```go
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
```

```go
type Model struct {
    ID      string
    Version string

    Metadata ModelMetadata
    Layers   []Layer
}

type Layer struct {
    Index      int
    MemoryBytes int64
    Tensors    []Tensor
}

type Tensor struct {
    Name      string
    Shape     []int64
    DataType  string
    SizeBytes int64
}
```

```go
type ModelShard struct {
    ID          string
    ModelID     string
    Version     string
    Layers      LayerRange
    MemoryBytes int64
}
```

## Construction

A shard can only be created for valid contiguous layer ranges:

```go
func NewModelShard(model *Model, start int, end int) (*ModelShard, error)
```

Rules:
- start must be >= 0
- end must be >= start
- end must not be beyond model.Layers length - 1
- memory is aggregated from all layer memory entries in the range

## Partitioning

Use a `ShardPlanner` to create logical shards:

```go
type ShardPlanner struct{}

func (p *ShardPlanner) Split(model *Model, ranges []LayerRange) ([]ModelShard, error)
```

This task does not decide worker assignment. It creates the logical shard definitions only.

## Validation

```go
func ValidatePartition(layerCount int, ranges []LayerRange) error
```

Validation rejects:
- missing layers
- overlapping ranges
- empty ranges
- ranges that exceed the model's layer count

## Example

For a 30-layer model:

```go
ranges := []LayerRange{
    {Start: 0, End: 9},
    {Start: 10, End: 19},
    {Start: 20, End: 29},
}
```

This produces:

- `model-0-9`
- `model-10-19`
- `model-20-29`

## Tests

The test suite should validate:
- `LayerRange.Size()`
- `LayerRange.Contains()`
- shard creation for valid ranges
- rejected invalid ranges
- partition validation for contiguous ranges
- partition validation failure for overlaps or gaps

## Why this matters

This abstraction creates the scheduler's basic working unit:

```text
Model
    ├── Shard 0: 0–9
    ├── Shard 1: 10–19
    └── Shard 2: 20–29
```

The scheduler later selects a worker for each shard without needing to understand tensor-level details.
