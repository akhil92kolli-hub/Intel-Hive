# Android native compute backends

The current Android Qwen2.5 layer-range executor is a CPU backend. The build
sets `GGML_VULKAN=OFF`, links `ggml-cpu`, and executes each graph with
`ggml_graph_compute_with_ctx`. It now uses up to eight available CPU threads;
worker registration and benchmark telemetry therefore report `cpu` and
`llama.cpp-cpu-layer-range` respectively.

Enabling `GGML_VULKAN` at compile time is insufficient for this executor. Its
GGUF tensors point directly into an mmap-backed model file and its graph tensors
are created in a CPU ggml context. A Vulkan implementation must:

1. initialize and capability-test an Android Vulkan ggml backend;
2. allocate backend buffers for the selected shard's weights and graph tensors;
3. copy or stage mmap-backed GGUF tensors into those buffers without loading
   unrelated layers;
4. schedule the layer-range graph through `ggml_backend_sched`;
5. retain shard-local KV state on the selected backend;
6. fall back to CPU when required Vulkan operations or memory are unavailable;
7. pass hidden-state, logit, prefill, decode, and KV-position parity tests before
   it is advertised as a worker capability.

Benchmark records identify whether one phone ran every logical shard or a real
network pipeline was used:

- `single_device_all_shards` is the Android device benchmark;
- `distributed_pipeline` is reserved for the two/three-device transport test.

A single-device result must not be used as evidence that activation transport
between independent workers has passed.
