# T006: Activation Envelope Serialization

## Objective

Create a platform-neutral activation representation and serialization logic that can later be transported between workers.

The activation should be independent of Android or iOS. It is a pure data structure with:

- job id
- request id
- sequence id
- source/destination worker
- layer index
- dtype
- shape
- payload
- checksum

## Activation Data Model

```go
type Envelope struct {
    JobID             string
    RequestID         string
    SequenceID        string
    ModelID           string
    ModelVersion      string
    SourceWorker      string
    DestinationWorker string
    Position          uint32
    Layer             int
    DType             string
    Shape             []int64
    Payload           []byte
    Checksum          string
}
```

## Serialization

Provide:

```go
func (a Envelope) Validate() error
func (a Envelope) ValidateBoundary(...) error
```

The serialization implementation should be:
- deterministic
- easily testable
- easy to validate
- independent of hardware backend

## Why this is important

The scheduler and the worker runtime should both understand the same activation format before any real phone-to-phone transfer begins.

This gives us a clean path to:
- mock worker transport
- no-phone development
- Android/iOS runtime integration later

## Validation Rules

- payload must be non-empty for realistic activation packets
- checksum must match the payload hash
- dtype and shape must exactly account for the payload bytes
- job, request, model/version, position, layer, and worker route must match the
  receiving assignment
- dtype and shape must remain stable when round tripping

## Recommended checksum

Use SHA-256:

```text
checksum = sha256(payload)
```

## Round-trip Test

A round-trip test should prove:

1. serialize succeeds
2. deserialize succeeds
3. payload remains identical
4. checksum validates successfully

## Next milestone

This is the last data-level task before actual transport and the mock worker pipeline.
