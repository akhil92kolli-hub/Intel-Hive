# T002: Single-Device Benchmark Specification

## Objective

Establish the baseline performance of Qwen2.5-3B-Instruct Q4_K_M on a single Android device using Vulkan GPU acceleration via llama.cpp.

## Model

- **Name:** Qwen2.5-3B-Instruct
- **Format:** GGUF
- **Quantization:** Q4_K_M (~2.2 GB)
- **Architecture:** Transformer (30 layers, 3072 hidden size)
- **Context:** 2K–4K tokens (MVP focus)
- **License:** Check Qwen terms; ensure compliance

## Setup

### 1. Model Preparation

Download the quantized model:

```bash
# Option A: From Hugging Face
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct-q4_k_m.gguf \
  --local-dir ./models

# Option B: From ollama library (if available)
```

Verify SHA-256 checksum before deployment.

### 2. Android Environment

**Minimum Requirements:**
- Android 14 (API 34+)
- ARM64-v8a architecture
- 4 GB RAM (recommended 6+ GB)
- Vulkan-capable GPU
- 3 GB free storage for model + intermediate state

**Target Devices:**
- Pixel 6 or newer (Adreno)
- Samsung Galaxy S21+ (Mali)
- OnePlus 9+ (Adreno)

### 3. Build & Deploy

```bash
cd android-worker
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/intel-hive-worker-debug.apk
```

### 4. JNI Bindings

The app requires native bindings for:
- Model loading and context creation
- Tokenization
- Forward pass (layer-by-layer)
- Activation export
- KV-cache management

**Phase-0 Implementation:**
- Use pre-built llama.cpp Android NDK bindings (or build from source)
- Vulkan backend: `-DGGML_VULKAN=ON`
- GPU layer offload: 30 (all layers)

## Benchmark Protocol

### Execution Flow

1. **Model Load:** Time to load GGUF and initialize Vulkan context
2. **Prefill Phase:** Process N tokens of the prompt in one batch
3. **Generation Phase:** Auto-regressive token generation, one token at a time
4. **Measurement:** Separate timing for prefill and generation
5. **Export:** One activation tensor from a mid-layer (e.g., layer 15)

### Input

**Prompt:** "The future of AI is"

**Configuration:**
- Prefill tokens: 128
- Generated tokens: 100
- Total tokens: 228

### Metrics Collected

```json
{
  "benchmark": {
    "prefill_tokens": 128,
    "generated_tokens": 100,
    "total_time_ms": <int>,
    "tokens_per_second": <float>,
    "prefill_speed_tokens_per_second": <float>,
    "generation_speed_tokens_per_second": <float>
  },
  "memory": {
    "peak_rss_mb": <int>,
    "peak_gpu_mb": <int>
  },
  "thermal": {
    "initial_temp_c": <float>,
    "peak_temp_c": <float>,
    "final_temp_c": <float>
  },
  "activations": {
    "layer_output_size_bytes": <int>,
    "dtype": "float32",
    "shape": [1, 128, 3072]
  }
}
```

## Expected Results (Baseline Estimates)

### Pixel 6 (Adreno 650)

```
prefill_speed:   80–120 tok/s
generation_speed: 5–8 tok/s
end_to_end:      6–8 tok/s
memory:          ~3.5 GB RAM + 1.2 GB GPU
thermal_delta:   +10–15°C
```

### Flagship (e.g., Pixel 8)

```
prefill_speed:   150–200 tok/s
generation_speed: 8–12 tok/s
end_to_end:      8–10 tok/s
memory:          ~3.5 GB RAM + 1.5 GB GPU
thermal_delta:   +8–12°C
```

### Mid-Range (e.g., Galaxy A54)

```
prefill_speed:   40–70 tok/s
generation_speed: 3–5 tok/s
end_to_end:      3–5 tok/s
memory:          ~3.5 GB RAM + 800 MB GPU
thermal_delta:   +12–18°C
```

## Output Format

All benchmarks export JSON to:

```
/data/data/com.intellihive.worker/cache/benchmark_result.json
```

Retrieve via:

```bash
adb pull /data/data/com.intellihive.worker/cache/benchmark_result.json
```

## Deliverables

- ✅ Android project with llama.cpp JNI bindings
- ✅ InferenceEngine abstraction (Phase-0: single device)
- ✅ BenchmarkService for measurement and export
- ✅ MainActivity UI for starting benchmark
- ✅ Benchmark specification and expected results
- 📝 JSON results from ≥3 different devices

## Next Steps (T003)

1. Deploy to 3 reference devices
2. Collect baseline benchmarks
3. Export results to `benchmark/results/`
4. Analyze bottlenecks
5. Proceed to two-device activation transfer (no inference yet)
