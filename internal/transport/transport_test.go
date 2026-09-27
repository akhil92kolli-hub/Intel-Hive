package transport

import "testing"

func TestTransportRoundTrip(t *testing.T) {
    transport := NewInMemoryTransport()
    payload := []byte{1, 2, 3, 4, 5}
    packet := ActivationPacket{
        JobID:             "job-1",
        RequestID:         "request-1",
        SequenceID:        7,
        SourceWorker:      "worker-a",
        DestinationWorker: "worker-b",
        Layer:             12,
        DType:             "f16",
        Shape:             []int64{1, 17, 2048},
        Payload:           payload,
        Checksum:          "sha256:yes",
    }

    if err := transport.Send(packet); err == nil {
        t.Fatal("expected checksum validation to fail for placeholder checksum")
    }

    packet.Checksum = "sha256:53994b3d7d52f55f4b4d7d445db23412357ff12f3ae6f5b785a4a64d1985929a"
    if err := transport.Send(packet); err != nil {
        t.Fatalf("expected valid packet to succeed, got: %v", err)
    }

    received, err := transport.Receive()
    if err != nil {
        t.Fatalf("receive failed: %v", err)
    }
    if received.JobID != packet.JobID {
        t.Fatal("job id mismatch")
    }
    if received.DestinationWorker != "worker-b" {
        t.Fatal("destination worker mismatch")
    }
}
