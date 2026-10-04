# Android native backends and resumable benchmarks

The Android benchmark reuses IntelHive's existing layered-shard executor and
llama.cpp JNI path. It runs three profiles sequentially, with generation targets
of 32, 64, 128, 256, and 512 tokens for each profile:

- **CPU-only** runs every shard on the CPU.
- **Vulkan-targeted** requests Vulkan for all three shards.
- **Hybrid** requests Vulkan for `s0` and `s1` (layers 0–19), and CPU for `s2`
  (layers 20–35).

Each profile uses four graph threads for a consistent comparison. Vulkan API
availability is recorded separately from inference. A Vulkan-targeted stage is
only counted as completed when the executor reports Vulkan execution for every
shard requested by that profile. Missing Vulkan execution is recorded as a
failed step with the missing shard IDs; a device without Vulkan records those
stages as skipped. Backend setup or inference failures are isolated to their
stage so later measurements can proceed.

The suite checkpoint is atomically persisted on-device. Completed stages are
not repeated when the user resumes. A stage interrupted while running is
recovered as cancelled with an explicit reason, and users can retry failed,
cancelled, or skipped stages individually. The notification and vertically
scrollable activity show stage progress, generated-token counts, error details,
and Supabase upload status. Reset requires confirmation, creates a new local
suite ID, and does not remove append-only Supabase history.

Every attempt is appended to the existing `benchmark_results` table using the
mobile insert-only policy. Its `raw_result` includes the suite/step/attempt
identity, configured profile, observed Vulkan shards, inference metrics,
thermal and memory telemetry, and any failure or cancellation reason.
Unsuccessful uploads stay in the local checkpoint and are retried when the
suite is next resumed. The app contains only the Supabase publishable key; it
does not read or update historical rows and must never contain a service-role
credential.

The device profile includes Android version, SoC/ABI, CPU core count and
available frequency data, total/available RAM, and Vulkan API/device/vendor/
driver data when exposed by the native capability probe. Sampling during an
inference step captures battery temperature and Android thermal status, app
PSS/RSS, native/Java heap, available RAM, major page faults, and current CPU
frequencies where the device permits access. Battery temperature is not a CPU
or GPU temperature sensor. Missing telemetry is represented as unavailable,
not treated as a benchmark failure.

The benchmark flow is foreground and cancellable between native inference
passes. Longer token targets can take substantial time on slower phones; users
can stop and resume without losing finished stages. The Qwen2.5-3B model and
native runtime must be installed before a run. Only a completed CPU-only
measurement updates worker readiness; Vulkan/hybrid results do not replace the
CPU readiness baseline.

Android Vulkan builds use the NDK Vulkan loader and require host `glslc`,
Vulkan-Headers (including Vulkan-Hpp), and SPIRV-Headers. The Vulkan path builds
for `arm64-v8a`. Native build success does not establish runtime compatibility
or performance on every Android GPU; actual execution still requires
on-device validation.

Benchmark records identify whether one phone ran every logical shard or a real
network pipeline was used:

- `single_device_all_shards` is the Android device benchmark;
- `distributed_pipeline` is reserved for the multi-device transport test.

A single-device result must not be used as evidence that activation transport
between independent workers has passed.
