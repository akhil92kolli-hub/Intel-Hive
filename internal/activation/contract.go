package activation

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

// Envelope is the JSON activation contract shared by the scheduler and mobile
// workers. Sequence IDs are strings to match inference-step assignments.
type Envelope struct {
	Position          uint32  `json:"position"`
	JobID             string  `json:"job_id"`
	RequestID         string  `json:"request_id"`
	SequenceID        string  `json:"sequence_id"`
	ModelID           string  `json:"model_id"`
	ModelVersion      string  `json:"model_version"`
	SourceWorker      string  `json:"source_worker"`
	DestinationWorker string  `json:"destination_worker"`
	Layer             int     `json:"layer"`
	DType             string  `json:"dtype"`
	Shape             []int64 `json:"shape"`
	Payload           []byte  `json:"payload"`
	Checksum          string  `json:"checksum"`
}

func NewEnvelope(jobID, requestID, sequenceID, modelID, modelVersion string,
	sourceWorker, destinationWorker string, layer int, dtype string, shape []int64, payload []byte,
) Envelope {
	data := append([]byte(nil), payload...)
	return Envelope{
		JobID:             jobID,
		RequestID:         requestID,
		SequenceID:        sequenceID,
		ModelID:           modelID,
		ModelVersion:      modelVersion,
		SourceWorker:      sourceWorker,
		DestinationWorker: destinationWorker,
		Layer:             layer,
		DType:             dtype,
		Shape:             append([]int64(nil), shape...),
		Payload:           data,
		Checksum:          envelopeChecksum(data),
	}
}

func (a Envelope) Validate() error {
	switch {
	case strings.TrimSpace(a.JobID) == "":
		return fmt.Errorf("activation job_id is required")
	case strings.TrimSpace(a.RequestID) == "":
		return fmt.Errorf("activation request_id is required")
	case strings.TrimSpace(a.SequenceID) == "":
		return fmt.Errorf("activation sequence_id is required")
	case strings.TrimSpace(a.ModelID) == "":
		return fmt.Errorf("activation model_id is required")
	case strings.TrimSpace(a.ModelVersion) == "":
		return fmt.Errorf("activation model_version is required")
	case strings.TrimSpace(a.SourceWorker) == "":
		return fmt.Errorf("activation source_worker is required")
	case strings.TrimSpace(a.DestinationWorker) == "":
		return fmt.Errorf("activation destination_worker is required")
	case a.Layer < 0:
		return fmt.Errorf("activation layer cannot be negative")
	case strings.TrimSpace(a.DType) == "":
		return fmt.Errorf("activation dtype is required")
	case len(a.Shape) == 0:
		return fmt.Errorf("activation shape is required")
	}
	for _, dimension := range a.Shape {
		if dimension <= 0 {
			return fmt.Errorf("activation dimensions must be positive")
		}
	}
	if len(a.Payload) == 0 {
		return fmt.Errorf("activation payload cannot be empty")
	}
	bytesPerElement := map[string]int64{"float32": 4, "float16": 2, "uint8": 1}[a.DType]
	if bytesPerElement == 0 {
		return fmt.Errorf("unsupported activation dtype %q", a.DType)
	}
	expected := bytesPerElement
	for _, dimension := range a.Shape {
		if dimension > int64(len(a.Payload))/expected {
			return fmt.Errorf("activation shape does not match payload length")
		}
		expected *= dimension
	}
	if expected != int64(len(a.Payload)) {
		return fmt.Errorf("activation shape does not match payload length")
	}
	if a.Checksum != envelopeChecksum(a.Payload) {
		return fmt.Errorf("activation checksum mismatch")
	}
	return nil
}

func envelopeChecksum(payload []byte) string {
	sum := sha256.Sum256(payload)
	return fmt.Sprintf("sha256:%x", sum[:])
}

// ValidateBoundary binds a tensor to the exact pass and adjacent workers.
func (a Envelope) ValidateBoundary(job, request, sequence, model, version, source, destination string, layer int, position uint32) error {
	if err := a.Validate(); err != nil {
		return err
	}
	if a.JobID != job || a.RequestID != request || a.SequenceID != sequence || a.ModelID != model || a.ModelVersion != version || a.SourceWorker != source || a.DestinationWorker != destination || a.Layer != layer || a.Position != position {
		return fmt.Errorf("activation identity, route, layer, or position does not match assignment")
	}
	return nil
}
