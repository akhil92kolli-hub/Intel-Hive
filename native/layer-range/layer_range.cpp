#include "layer_range.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace intelhive {
namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
Context context() {
    auto* p = ggml_init({4 * 1024 * 1024, nullptr, false});
    if (!p) throw std::runtime_error("ggml context allocation failed");
    return Context(p, ggml_free);
}
ggml_tensor* tensor(ggml_context* c, int rows, int cols, const std::vector<float>& data) {
    auto* t = ggml_new_tensor_2d(c, GGML_TYPE_F32, rows, cols);
    if (data.size() != static_cast<size_t>(rows * cols)) throw std::invalid_argument("tensor size");
    std::memcpy(t->data, data.data(), data.size() * sizeof(float));
    return t;
}
ggml_tensor* weights(ggml_context* c, int layer, int kind, int rows, int cols) {
    std::vector<float> data(rows * cols);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = 0.12f * std::sin(float((i + 1) * (kind + 1) + 17 * layer));
    return tensor(c, rows, cols, data);
}
void compute(ggml_context* c, ggml_tensor* output) {
    auto* graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(c, graph, 1) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("ggml graph execution failed");
}
std::vector<float> values(ggml_tensor* t) {
    auto* data = static_cast<float*>(t->data);
    return {data, data + ggml_nelements(t)};
}
// One attention head with standard RoPE; kept explicit to expose token offsets.
void rope(std::vector<float>& x, int position) {
    for (int i = 0; i < LayerRange::width; i += 2) {
        float angle = position * std::pow(10000.0f, -float(i) / LayerRange::width);
        float a = x[i], b = x[i + 1];
        x[i] = a * std::cos(angle) - b * std::sin(angle);
        x[i + 1] = a * std::sin(angle) + b * std::cos(angle);
    }
}
}
LayerRange::LayerRange(int first, int last): first_(first), last_(last) {
    if (first < 0 || last < first || last >= layer_count)
        throw std::invalid_argument("invalid inclusive layer range");
    cache_.resize(owned_layers());
}
void LayerRange::reset() {
    tokens_ = 0;
    for (auto& cache : cache_) { cache.keys.clear(); cache.values.clear(); }
}
std::vector<float> LayerRange::execute(const std::vector<float>& input, int token_offset) {
    if (input.empty() || input.size() % width || token_offset != tokens_)
        throw std::invalid_argument("invalid residual shape or KV token offset");
    const auto count = input.size() / width;
    if (count > static_cast<size_t>(max_tokens - tokens_))
        throw std::invalid_argument("context capacity exceeded");
    for (float x : input) if (!std::isfinite(x)) throw std::invalid_argument("non-finite input");
    // Commit cache changes only if the whole pass succeeds.
    auto pending = cache_;
    std::vector<float> output;
    output.reserve(input.size());
    for (size_t token = 0; token < count; ++token) {
        std::vector<float> x(input.begin() + token * width, input.begin() + (token + 1) * width);
        for (int i = first_; i <= last_; ++i)
            x = layer(x, i, tokens_ + static_cast<int>(token), pending[i - first_]);
        output.insert(output.end(), x.begin(), x.end());
    }
    cache_ = std::move(pending);
    tokens_ += static_cast<int>(count);
    return output;
}
std::vector<float> LayerRange::layer(const std::vector<float>& input, int index, int position, Cache& cache) {
    auto ctx = context(); auto* c = ctx.get();
    auto* x = tensor(c, width, 1, input);
    auto* norm = ggml_rms_norm(c, x, 1e-6f);
    auto* q = ggml_mul_mat(c, weights(c, index, 0, width, width), norm);
    auto* k = ggml_mul_mat(c, weights(c, index, 1, width, width), norm);
    auto* v = ggml_mul_mat(c, weights(c, index, 2, width, width), norm);
    compute(c, q); compute(c, k); compute(c, v);
    auto qv = values(q), kv = values(k), vv = values(v);
    rope(qv, position); rope(kv, position);
    cache.keys.insert(cache.keys.end(), kv.begin(), kv.end());
    cache.values.insert(cache.values.end(), vv.begin(), vv.end());
    auto* keys = tensor(c, width, position + 1, cache.keys);
    auto* query = tensor(c, width, 1, qv);
    auto* scores = ggml_scale(c, ggml_mul_mat(c, keys, query), 1.0f / std::sqrt(float(width)));
    auto* probabilities = ggml_soft_max(c, scores);
    auto* all_values = tensor(c, width, position + 1, cache.values);
    auto* attended = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, all_values)), probabilities);
    auto* projected = ggml_mul_mat(c, weights(c, index, 3, width, width), attended);
    auto* residual = ggml_add(c, x, projected);
    auto* fnorm = ggml_rms_norm(c, residual, 1e-6f);
    auto* gate = ggml_silu(c, ggml_mul_mat(c, weights(c, index, 4, width, width * 2), fnorm));
    auto* up = ggml_mul_mat(c, weights(c, index, 5, width, width * 2), fnorm);
    auto* down = ggml_mul_mat(c, weights(c, index, 6, width * 2, width), ggml_mul(c, gate, up));
    auto* result = ggml_add(c, residual, down);
    compute(c, result);
    return values(result);
}
}
