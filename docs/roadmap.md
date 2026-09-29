# IntelHive Roadmap

This roadmap uses milestone acceptance gates to sequence engineering work. A
task being implemented is not itself a milestone: each milestone is complete
only when its end-to-end acceptance test passes.

## Milestones at a glance

| Milestone | Phase | Acceptance gate |
| --- | --- | --- |
| M1 — Distributed LAN inference | Phase 0 | Three ordinary Android phones run a quantized 3B model together over a local network and return a generated response. |
| M2 — Dynamic worker replacement | Phase 0 | A worker can disappear during an inference request and the scheduler selects a suitable replacement and rebuilds the pipeline without a second client request. |
| M3 — Heterogeneous worker network | Phase 1 | 10–20 heterogeneous phones automatically form and measure inference pipelines. |
| M4 — Remote Internet workers | Phase 2 | Remote workers participate securely, with network costs, contributions, and usage measured. |

Do not begin Internet-scale architecture before M1–M3 are demonstrated reliably.
M4 is a significant network, security, and economics transition, not a
prerequisite for the first mobile-app release.

## Phase 0 — Prove distributed inference

**Goal:** Three ordinary Android phones jointly execute a quantized 3B model
over a local network.

### M1 — Three-phone inference

The scheduler assigns logical layer shards to workers. The initial scheduler
must remain deterministic and capability-based; it is not an AI/LLM scheduler.
For a 30-layer example, the demonstration should show:

```text
Phone A              Phone B              Phone C
Layers 0–9           Layers 10–19         Layers 20–29
    │                     │                     │
    └──── activation ────►└──── activation ────►
                                                   │
                                                   ▼
                                            Generated response
```

The client sends one inference request to the scheduler. The scheduler assigns
the model shards, activations move between workers, and generated tokens return
to the client.

| Task | Work | Gate |
| --- | --- | --- |
| T001 | Repository, Go services, Android project, and shared protocol foundation | Foundation |
| T002A | IntelHive-owned `InferenceEngine` and logical shard execution contract | M1 |
| T002B | Portable activation metadata, checksum, and sequence/KV-state contract | M1 |
| T002C | llama.cpp/ggml feasibility spike for executing one logical layer range | M1 |
| T003 | Android Worker Runtime | M1 |
| T004 | Model metadata and layer discovery | M1 |
| T005 | Logical `ModelShard` / layer-range abstraction | M1 |
| T006 | Activation serialization | M1 |
| T007 | Worker-to-worker activation transport | M1 |
| T008 | Mock Worker | M1 |
| T009 | Mock distributed inference | M1 |
| T010 | Scheduler Core | M1 |
| T011 | Dynamic Worker Assignment | M1 initial deterministic assignment; M2 failure-time reassignment |
| T013 | Android llama.cpp native integration | M1 |
| T014 | Android GPU/backend integration | M1 |
| T015 | Real-device hardware benchmark | M1 |
| T016 | Replace `MockWorker` with Android Worker | M1 |
| T017 | Real two-phone distributed inference | M1 |
| T018 | Real three-phone distributed inference | **M1 acceptance test** |

T011 provides the initial shard-to-worker assignment needed by M1. M2 extends
that capability to reassign a shard after a worker failure. T012 is the
failure/reconnection foundation for M2.

T002A and T002B define the Android-side engine and activation contracts. T002C
now has a native ggml feasibility prototype under `native/layer-range`: it
builds and verifies split-range equivalence, prefill/decode KV positions, and
sequence reset using deterministic synthetic weights. This proves that
IntelHive can own partial-graph boundaries around ggml; it does not load GGUF
weights or execute Qwen, and does not complete Android native inference. The
scheduler owns pipeline assignment; IntelHive's worker runtime owns local
shard execution and sequence state, while llama.cpp/ggml is only a local
compute backend. M1 remains blocked on a real model-weight loader and Android
backend integration (T013 onward).

### M2 — Worker failure and replacement

The client makes exactly one `POST /inference`; it does not need to know that a
worker disappeared. For example, if B goes offline in `A → B → C`, the
scheduler selects a ready replacement X for B's required model and layer range
and reconstructs `A → X → C`.

| Task | Work | Gate |
| --- | --- | --- |
| T012 | Failure detection and pipeline rebuild foundation | M2 |
| T019 | Worker health monitoring and heartbeat timeout (`READY` → `OFFLINE`) | M2 |
| T020 | Replacement-worker selection by model, shard, readiness, and capacity | M2 |
| T021 | Pipeline reconstruction after replacement | M2 |
| T022 | End-to-end three-phone failure and replacement test | **M2 acceptance test** |

Heartbeat reports include worker ID, timestamp, state, loaded shards, and
performance. Replacement selection must reject offline workers, incompatible
shards, and workers with insufficient memory.

## Phase 1 — Heterogeneous worker network

**Start after M2.**

### M3 — Automatic pipelines across 10–20 devices

Move from fixed device assignments to selecting the best eligible worker for
each required shard. Keep selection deterministic and measurable.

| Task | Work |
| --- | --- |
| T023 | Android device capability profiler |
| T024 | CPU/GPU/backend capability reporting |
| T025 | RAM and available-memory reporting |
| T026 | Model/shard capability advertisement |
| T027 | Worker performance benchmark |
| T028 | Heterogeneous worker registry |
| T029 | Scheduler candidate matching |
| T030 | Deterministic worker selection |
| T031 | Multiple candidate workers per shard |
| T032 | Worker reuse prevention within a pipeline |
| T033 | Dynamic pipeline construction |
| T034 | Worker join/leave handling |
| T035 | Pipeline rebuild |
| T036 | 10-worker integration test |
| T037 | 20-worker integration test |
| T038 | Network latency measurement |
| T039 | Pipeline throughput measurement |
| T040 | End-to-end M3 demonstration |

The scheduler should consider worker availability, required model/shard,
backend, memory, benchmarked performance, and network latency. No AI/LLM-based
scheduling is in scope for this phase.

## Phase 2 — Remote Internet workers

**Start after M3.**

### M4 — Remote workers with measurable economics

Internet participation introduces NAT traversal, TLS, authentication,
unreliable connections, variable latency, bandwidth costs, churn, malicious
workers, relays, regional latency, accounting, and usage measurement. These
concerns should be addressed after the LAN milestones are reliable.

| Task | Work |
| --- | --- |
| T041 | Worker authentication |
| T042 | Secure remote worker protocol |
| T043 | Internet connectivity |
| T044 | NAT/firewall strategy |
| T045 | Remote worker discovery |
| T046 | Regional worker selection |
| T047 | Network latency measurement |
| T048 | Bandwidth measurement |
| T049 | Inference cost measurement |
| T050 | Worker contribution accounting |
| T051 | Developer usage accounting |
| T052 | Credit/economic model |
| T053 | Abuse detection |
| T054 | Worker reputation/health |
| T055 | Remote failure recovery |
| T056 | M4 Internet test |
| T057 | M4 economics dashboard |

## Recommended immediate execution sequence

1. T010 — Scheduler Core
2. T011 — Dynamic Worker Assignment
3. T012 — Failure/reconnection foundation
4. T008/T009 — Mock distributed validation
5. T013 — Android llama.cpp integration
6. T015 — Real-device benchmark
7. T016 — Android Worker integration
8. T017 — Real two-phone inference
9. T018 — Real three-phone inference and M1 demonstration

T014 (Android GPU/backend integration) is also required within M1 where the
selected device/backend needs it. Once M1 passes, complete T019–T022 to reach
M2; then proceed to T023–T040 for M3 and T041–T057 for M4.

## Mobile app release track

App publication is a product-readiness track, not a substitute for a technical
milestone.

| Artifact | Target |
| --- | --- |
| Android engineering/test app | During Phase 0 to support M1/M2 testing |
| Android internal/beta build | After M1 |
| Android closed testing and public release | Around M2 to early M3, after recovery and product-readiness checks |
| iOS engineering worker | Begin during Phase 1; initially validate as a foreground/controlled worker |
| iOS TestFlight | After Android M1/M2 is stable and the iOS execution model is validated |
| iOS public App Store release | After background/execution constraints, privacy, security, and store requirements are validated |

The public compute-provider app additionally needs onboarding, identity and
consent, privacy controls, battery/charging/Wi-Fi controls, thermal protection,
worker status, model-download handling, secure authentication, diagnostics,
privacy policy, terms, security documentation, support, and store compliance.
Do not recruit a large public Android worker network before M3.

## Acceptance-gate order

```text
Phase 0: LAN
  M1: three Android phones jointly generate a response
   ↓
  M2: a failed worker is transparently replaced
   ↓
  Android beta/public release track
   ↓
Phase 1
  M3: 10–20 heterogeneous phones form pipelines automatically
   ↓
  iOS validation and release track
   ↓
Phase 2
  M4: remote workers participate with measurable economics
```
