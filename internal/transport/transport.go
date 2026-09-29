package transport

import (
	"fmt"
	"sync"

	"github.com/akhil92kolli-hub/Intel-Hive/internal/activation"
)

// ActivationPacket is the transport-level representation of a logical activation.
type ActivationPacket struct {
	JobID             string  `json:"job_id"`
	RequestID         string  `json:"request_id"`
	SequenceID        uint64  `json:"sequence_id"`
	SourceWorker      string  `json:"source_worker"`
	DestinationWorker string  `json:"destination_worker"`
	Layer             int     `json:"layer"`
	DType             string  `json:"dtype"`
	Shape             []int64 `json:"shape"`
	Payload           []byte  `json:"payload"`
	Checksum          string  `json:"checksum"`
}

// Transport moves activation data from one worker to another.
type Transport interface {
	Send(packet ActivationPacket) error
	Receive() (ActivationPacket, error)
	Validate(packet ActivationPacket) error
}

// InMemoryTransport is the default Phase-0 test transport.
type InMemoryTransport struct {
	mu    sync.Mutex
	queue chan ActivationPacket
}

func NewInMemoryTransport() *InMemoryTransport {
	return &InMemoryTransport{queue: make(chan ActivationPacket, 64)}
}

func (t *InMemoryTransport) Send(packet ActivationPacket) error {
	if packet.Checksum == "" {
		return fmt.Errorf("missing checksum")
	}
	if err := ValidateActivationPacket(packet); err != nil {
		return err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.queue <- packet
	return nil
}

func (t *InMemoryTransport) Receive() (ActivationPacket, error) {
	packet, ok := <-t.queue
	if !ok {
		return ActivationPacket{}, fmt.Errorf("transport closed")
	}
	if err := ValidateActivationPacket(packet); err != nil {
		return ActivationPacket{}, err
	}
	return packet, nil
}

func (t *InMemoryTransport) Validate(packet ActivationPacket) error {
	return ValidateActivationPacket(packet)
}

func ValidateActivationPacket(packet ActivationPacket) error {
	if packet.JobID == "" || packet.RequestID == "" {
		return fmt.Errorf("missing job or request identifiers")
	}
	if packet.DestinationWorker == "" {
		return fmt.Errorf("missing destination worker")
	}
	if len(packet.Payload) == 0 {
		return fmt.Errorf("activation payload is empty")
	}
	if packet.Checksum == "" {
		return fmt.Errorf("missing checksum")
	}
	if err := activation.ValidateChecksum(packet.Payload, packet.Checksum); err != nil {
		return err
	}
	return nil
}
