#pragma once
#include <vector>

namespace intelhive {
// Feasibility backend only: a small deterministic float32 decoder, not a GGUF
// loader or a Qwen model. Each instance owns only its inclusive layer range.
class LayerRange {
public:
    static constexpr int width = 8;
    static constexpr int layer_count = 6;
    static constexpr int max_tokens = 32;
    LayerRange(int first, int last);
    // Token-major residual stream [tokens, width]. Offset is the actual KV
    // token offset, not the scheduler's pass ordinal. Prefill may have >1 token.
    std::vector<float> execute(const std::vector<float>& input, int token_offset);
    void reset();
    int cached_tokens() const { return tokens_; }
    int owned_layers() const { return last_ - first_ + 1; }
private:
    struct Cache { std::vector<float> keys, values; };
    int first_, last_, tokens_ = 0;
    std::vector<Cache> cache_;
    std::vector<float> layer(const std::vector<float>& input, int index, int position, Cache& cache);
};
}
