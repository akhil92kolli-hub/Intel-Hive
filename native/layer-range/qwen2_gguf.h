#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace intelhive {

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

class Qwen2GgufModel;

class Qwen2GgufShard {
public:
    Qwen2GgufShard(std::shared_ptr<const Qwen2GgufModel> model, int32_t first_layer, int32_t last_layer);
    ~Qwen2GgufShard();
    Qwen2GgufShard(Qwen2GgufShard&&) noexcept;
    Qwen2GgufShard& operator=(Qwen2GgufShard&&) noexcept;
    Qwen2GgufShard(const Qwen2GgufShard&) = delete;
    Qwen2GgufShard& operator=(const Qwen2GgufShard&) = delete;

    std::vector<float> execute(const std::vector<float>& token_major_input, int32_t token_offset);
    void reset();
    int32_t cached_tokens() const;
    size_t mapped_weight_bytes() const;
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
    std::vector<float> project_logits(const std::vector<float>& hidden_state) const;
    const std::string& path() const;

private:
    friend class Qwen2GgufShard;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
