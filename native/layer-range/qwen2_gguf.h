#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace intelhive {

enum class InferenceBackend {
    Cpu,
    Vulkan,
};

struct Qwen2Config {
    int32_t layer_count;
    int32_t embedding_size;
    int32_t feed_forward_size;
    int32_t attention_heads;
    int32_t kv_heads;
    int32_t context_length;
    int32_t vocabulary_size;
    float rms_norm_epsilon;
    float rope_frequency_base;
};

struct BackendExecutionStats {
    InferenceBackend requested_backend = InferenceBackend::Cpu;
    uint64_t cpu_graph_nodes = 0;
    uint64_t vulkan_graph_nodes = 0;
    uint64_t gpu_weight_bytes = 0;
    uint64_t gpu_kv_bytes = 0;
    uint64_t host_kv_bytes = 0;
    uint64_t activation_transfer_bytes = 0;
    bool fallback_used = false;
};

class Qwen2GgufModel;

class Qwen2GgufShard {
public:
    Qwen2GgufShard(
        std::shared_ptr<const Qwen2GgufModel> model,
        int32_t first_layer,
        int32_t last_layer,
        int32_t graph_threads = 0,
        InferenceBackend backend = InferenceBackend::Cpu,
        int64_t max_duration_ms = 0);
    ~Qwen2GgufShard();
    Qwen2GgufShard(Qwen2GgufShard&&) noexcept;
    Qwen2GgufShard& operator=(Qwen2GgufShard&&) noexcept;
    Qwen2GgufShard(const Qwen2GgufShard&) = delete;
    Qwen2GgufShard& operator=(const Qwen2GgufShard&) = delete;

    std::vector<float> execute(
        const std::vector<float>& token_major_input,
        int32_t token_offset,
        int64_t max_duration_ms = 0);
    void reset();
    int32_t cached_tokens() const;
    size_t mapped_weight_bytes() const;
    bool used_vulkan() const;
    BackendExecutionStats backend_stats() const;
    int32_t first_layer() const;
    int32_t last_layer() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Qwen2GgufModel : public std::enable_shared_from_this<Qwen2GgufModel> {
public:
    explicit Qwen2GgufModel(const std::string& path);
    ~Qwen2GgufModel();
    Qwen2GgufModel(Qwen2GgufModel&&) noexcept;
    Qwen2GgufModel& operator=(Qwen2GgufModel&&) noexcept;
    Qwen2GgufModel(const Qwen2GgufModel&) = delete;
    Qwen2GgufModel& operator=(const Qwen2GgufModel&) = delete;

    const Qwen2Config& config() const;
    std::vector<int32_t> tokenize(const std::string& text) const;
    std::vector<float> embed_token(uint32_t token_id) const;
    std::vector<float> project_logits(
        const std::vector<float>& hidden_state,
        int32_t graph_threads = 0) const;
    size_t layer_weight_bytes(int32_t first_layer, int32_t last_layer) const;
    const std::string& path() const;

private:
    friend class Qwen2GgufShard;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
