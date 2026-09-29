#include "qwen2_gguf.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void compare(const std::vector<float>& expected, const std::vector<float>& actual) {
    require(expected.size() == actual.size(), "split execution changed activation shape");
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite activation");
        max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
    }
    require(max_error <= 1e-4f, "split-vs-unsplit activation error exceeds 1e-4");
}

void append(std::vector<float>& destination, const std::vector<float>& token) {
    destination.insert(destination.end(), token.begin(), token.end());
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: qwen2_gguf_test <qwen2.5-3b-instruct.gguf>");
        auto model = std::make_shared<intelhive::Qwen2GgufModel>(argv[1]);
        const auto& config = model->config();
        require(config.layer_count == 36 && config.embedding_size == 2048 &&
                config.feed_forward_size == 11008 && config.attention_heads == 16 &&
                config.kv_heads == 2,
                "GGUF metadata does not match Qwen2.5-3B-Instruct");

        const int32_t layers_per_shard = config.layer_count / 3;
        require(config.layer_count % 3 == 0, "test expects three equal layer ranges");
        intelhive::Qwen2GgufShard full(model, 0, config.layer_count - 1);
        intelhive::Qwen2GgufShard first(model, 0, layers_per_shard - 1);
        intelhive::Qwen2GgufShard second(model, layers_per_shard, 2 * layers_per_shard - 1);
        intelhive::Qwen2GgufShard third(model, 2 * layers_per_shard, config.layer_count - 1);

        const std::vector<uint32_t> token_ids{17, 42, 1337, 314, 2718, 5};
        std::vector<float> prefill;
        for (size_t i = 0; i < 3; ++i) append(prefill, model->embed_token(token_ids[i]));
        const auto full_prefill = full.execute(prefill, 0);
        const auto first_prefill = first.execute(prefill, 0);
        const auto second_prefill = second.execute(first_prefill, 0);
        const auto split_prefill = third.execute(second_prefill, 0);
        compare(full_prefill, split_prefill);

        for (size_t i = 3; i < token_ids.size(); ++i) {
            const auto token = model->embed_token(token_ids[i]);
            const auto full_output = full.execute(token, static_cast<int32_t>(i));
            const auto a = first.execute(token, static_cast<int32_t>(i));
            const auto b = second.execute(a, static_cast<int32_t>(i));
            const auto split_output = third.execute(b, static_cast<int32_t>(i));
            compare(full_output, split_output);
        }

        require(first.cached_tokens() == static_cast<int32_t>(token_ids.size()) &&
                second.cached_tokens() == first.cached_tokens() &&
                third.cached_tokens() == first.cached_tokens(),
                "shard-local KV positions diverged");
        const auto expected = model->embed_token(token_ids.front());
        full.reset();
        const auto reset_output = full.execute(expected, 0);
        intelhive::Qwen2GgufShard isolated(model, 0, config.layer_count - 1);
        compare(reset_output, isolated.execute(expected, 0));

        std::cout << "PASS: real GGUF Qwen2 layer weights, three-shard parity, prefill/decode, "
                  << "sequence reset, and KV positions; full-range mapped bytes="
                  << full.mapped_weight_bytes() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
