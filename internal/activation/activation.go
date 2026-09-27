package activation

import (
    "bytes"
    "crypto/sha256"
    "encoding/base64"
    "fmt"
    "testing"
)

// Activation represents a logical tensor output from one layer in a pipeline.
type Activation struct {
    JobID       string   `json:"job_id"`
    RequestID   string   `json:"request_id"`
    SequenceID  uint64   `json:"sequence_id"`
    SourceWorker string  `json:"source_worker"`
    DestinationWorker string `json:"destination_worker"`
    Layer       int      `json:"layer"`
    DType       string   `json:"dtype"`
    Shape       []int64  `json:"shape"`
    Payload     []byte   `json:"payload"`
    Checksum    string   `json:"checksum"`
}

func NewActivation(jobID string, requestID string, seq uint64, layer int, dtype string, shape []int64, payload []byte) *Activation {
    checksum := sha256.Sum256(payload)
    return &Activation{
        JobID:       jobID,
        RequestID:   requestID,
        SequenceID:  seq,
        Layer:       layer,
        DType:       dtype,
        Shape:       shape,
        Payload:     payload,
        Checksum:    fmt.Sprintf("sha256:%x", checksum[:]),
    }
}

func (a *Activation) Serialize() ([]byte, error) {
    // Minimal binary format for testing and round-trips.
    // Format: job_id_len | job_id | request_id_len | request_id | sequence | layer | dtype_len | dtype | shape_count | shape values | payload_len | payload | checksum_len | checksum
    var buf bytes.Buffer
    writeString := func(s string) {
        buf.Write([]byte{byte(len(s) >> 8), byte(len(s))})
        buf.WriteString(s)
    }
    writeInt64 := func(v int64) {
        var b [8]byte
        // use a tiny big-endian encoding for test simplicity
        for i := 0; i < 8; i++ {
            b[i] = byte(v >> (8 * uint(7-i)))
        }
        buf.Write(b[:])
    }

    writeString(a.JobID)
    writeString(a.RequestID)
    writeInt64(int64(a.SequenceID))
    writeInt64(int64(a.Layer))
    writeString(a.DType)
    buf.Write([]byte{byte(len(a.Shape))})
    for _, dim := range a.Shape {
        writeInt64(dim)
    }
    writeInt64(int64(len(a.Payload)))
    buf.Write(a.Payload)
    writeString(a.Checksum)
    return buf.Bytes(), nil
}

func DeserializeActivation(data []byte) (*Activation, error) {
    // This is intentionally small and strict for Phase-0 tests.
    // It reads the same format written by Serialize().
    // The implementation is intentionally straightforward and rejects malformed payloads.
    idx := 0
    readString := func() (string, error) {
        if idx+2 > len(data) {
            return "", fmt.Errorf("short string length")
        }
        length := int(data[idx])<<8 | int(data[idx+1])
        idx += 2
        if idx+length > len(data) {
            return "", fmt.Errorf("string exceeds buffer")
        }
        s := string(data[idx : idx+length])
        idx += length
        return s, nil
    }
    readInt64 := func() (int64, error) {
        if idx+8 > len(data) {
            return 0, fmt.Errorf("short int64")
        }
        v := int64(0)
        for i := 0; i < 8; i++ {
            v = (v << 8) | int64(data[idx+i])
        }
        idx += 8
        return v, nil
    }

    jobID, err := readString()
    if err != nil { return nil, err }
    requestID, err := readString()
    if err != nil { return nil, err }
    seq, err := readInt64()
    if err != nil { return nil, err }
    layer, err := readInt64()
    if err != nil { return nil, err }
    dtype, err := readString()
    if err != nil { return nil, err }

    shapeCount := int(data[idx])
    idx++
    shape := make([]int64, 0, shapeCount)
    for i := 0; i < shapeCount; i++ {
        dim, err := readInt64()
        if err != nil { return nil, err }
        shape = append(shape, dim)
    }
    payloadLen, err := readInt64()
    if err != nil { return nil, err }
    if idx+int(payloadLen) > len(data) {
        return nil, fmt.Errorf("payload exceeds buffer")
    }
    payload := append([]byte(nil), data[idx:idx+int(payloadLen)]...)
    idx += int(payloadLen)
    checksum, err := readString()
    if err != nil { return nil, err }

    return &Activation{
        JobID:       jobID,
        RequestID:   requestID,
        SequenceID:  uint64(seq),
        Layer:       int(layer),
        DType:       dtype,
        Shape:       shape,
        Payload:     payload,
        Checksum:    checksum,
    }, nil
}

func (a *Activation) Validate() error {
    computed := sha256.Sum256(a.Payload)
    expected := fmt.Sprintf("sha256:%x", computed[:])
    if a.Checksum != expected {
        return fmt.Errorf("checksum mismatch")
    }
    return nil
}

func (a *Activation) ToBase64() string {
    return base64.StdEncoding.EncodeToString(a.Payload)
}

func TestActivationRoundTrip(t *testing.T) {
    payload := []byte{1, 2, 3, 4, 5, 6, 7, 8}
    act := NewActivation("job-1", "request-1", 17, 12, "f16", []int64{1, 17, 2048}, payload)

    data, err := act.Serialize()
    if err != nil {
        t.Fatalf("serialize failed: %v", err)
    }

    roundTrip, err := DeserializeActivation(data)
    if err != nil {
        t.Fatalf("deserialize failed: %v", err)
    }

    if roundTrip.JobID != act.JobID || roundTrip.RequestID != act.RequestID || roundTrip.SequenceID != act.SequenceID {
        t.Fatalf("round trip produced different metadata")
    }

    if !bytes.Equal(roundTrip.Payload, payload) {
        t.Fatal("payload changed during round trip")
    }

    if err := roundTrip.Validate(); err != nil {
        t.Fatalf("checksum validation failed: %v", err)
    }
}
