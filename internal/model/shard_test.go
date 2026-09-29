package model

import "testing"

func TestLayerRangeSize(t *testing.T) {
    r := LayerRange{Start: 10, End: 19}
    if got := r.Size(); got != 10 {
        t.Fatalf("expected size 10, got %d", got)
    }
}

func TestLayerRangeContains(t *testing.T) {
    r := LayerRange{Start: 10, End: 19}
    if !r.Contains(10) {
        t.Fatal("expected layer 10 to be in range")
    }
    if !r.Contains(19) {
        t.Fatal("expected layer 19 to be in range")
    }
    if r.Contains(20) {
        t.Fatal("layer 20 should not be in range")
    }
}

func TestValidatePartition(t *testing.T) {
    ranges := []LayerRange{
        {Start: 0, End: 9},
        {Start: 10, End: 19},
        {Start: 20, End: 29},
    }
    if err := ValidatePartition(30, ranges); err != nil {
        t.Fatalf("expected valid partition, got error: %v", err)
    }
}

func TestInvalidPartition(t *testing.T) {
    ranges := []LayerRange{
        {Start: 0, End: 9},
        {Start: 11, End: 19},
        {Start: 20, End: 29},
    }
    if err := ValidatePartition(30, ranges); err == nil {
        t.Fatal("expected invalid partition for gap/overlap")
    }
}

func TestNewModelShard(t *testing.T) {
    model := &Model{
        ID:      "llama-3b",
        Version: "1",
        Layers: []Layer{
            {Index: 0, MemoryBytes: 100},
            {Index: 1, MemoryBytes: 110},
            {Index: 2, MemoryBytes: 120},
        },
    }

    shard, err := NewModelShard(model, 0, 1)
    if err != nil {
        t.Fatalf("expected valid shard, got err: %v", err)
    }
    if shard.MemoryBytes != 210 {
        t.Fatalf("expected memory 210, got %d", shard.MemoryBytes)
    }
}
