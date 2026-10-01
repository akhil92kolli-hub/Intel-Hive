#include "qwen2_gguf.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
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
    require(max_error <= 1e-4f,
        "activation parity error exceeds 1e-4 (max_abs_error=" + std::to_string(max_error) + ")");
}

float max_abs_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    require(expected.size() == actual.size(), "reference activation changed shape");
    float result = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "reference activation is non-finite");
        result = std::max(result, std::abs(expected[i] - actual[i]));
    }
    return result;
}

void append(std::vector<float>& destination, const std::vector<float>& token) {
    destination.insert(destination.end(), token.begin(), token.end());
}

uint32_t argmax(const std::vector<float>& values) {
    require(!values.empty(), "logit vector is empty");
    const auto best = std::max_element(values.begin(), values.end());
    require(std::isfinite(*best), "logits contain no finite maximum");
    return static_cast<uint32_t>(std::distance(values.begin(), best));
}

struct LlamaReference {
    std::vector<std::vector<float>> logits;
    std::vector<float> first_layer_output;
};

float compare_reference(const std::vector<float>& expected, const std::vector<float>& actual) {
    require(expected.size() == actual.size(), "llama.cpp reference logits changed shape");
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "reference logits are non-finite");
        max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
    }
    if (argmax(expected) != argmax(actual)) {
        throw std::runtime_error("llama.cpp and layer-range greedy tokens differ: reference=" +
            std::to_string(argmax(expected)) + ", executor=" + std::to_string(argmax(actual)) +
            ", max_abs_error=" + std::to_string(max_error));
    }
    require(max_error <= 0.75f,
        "llama.cpp reference logit error exceeds 0.75 (max_abs_error=" +
        std::to_string(max_error) + ")");
    return max_error;
}

LlamaReference llama_reference_logits(
        const std::string& model_path,
        const std::vector<uint32_t>& token_ids,
        size_t prefill_tokens) {
    require(prefill_tokens > 0 && prefill_tokens <= token_ids.size(), "invalid reference prefill length");
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    struct BackendCleanup {
        ~BackendCleanup() { llama_backend_free(); }
    } backend_cleanup;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
    require(model != nullptr, "llama.cpp could not load the reference GGUF");

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 32;
    context_params.n_batch = static_cast<uint32_t>(prefill_tokens);
    context_params.n_ubatch = static_cast<uint32_t>(prefill_tokens);
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    LlamaReference result;
    struct EvalTrace {
        std::vector<float> first_layer_output;
    } trace;
    context_params.cb_eval = [](ggml_tensor* tensor, bool ask, void* user_data) {
        auto* eval_trace = static_cast<EvalTrace*>(user_data);
        if (std::string(tensor->name).find("l_out-0") == std::string::npos) return true;
        if (!ask) {
            const auto* begin = static_cast<const float*>(tensor->data);
            eval_trace->first_layer_output.assign(begin, begin + ggml_nelements(tensor));
        }
        return true;
    };
    context_params.cb_eval_user_data = &trace;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    require(context != nullptr, "llama.cpp could not create the reference context");

    const auto copy_logits = [&](int32_t index) {
        const float* logits = llama_get_logits_ith(context.get(), index);
        require(logits != nullptr, "llama.cpp did not return reference logits");
        const int32_t vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        return std::vector<float>(logits, logits + vocabulary_size);
    };

    std::vector<llama_token> prompt(token_ids.begin(), token_ids.begin() + prefill_tokens);
    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    require(llama_decode(context.get(), batch) == 0, "llama.cpp reference prefill failed");
    result.first_layer_output = trace.first_layer_output;
    result.logits.push_back(copy_logits(-1));
    for (size_t i = prefill_tokens; i < token_ids.size(); ++i) {
        llama_token next = static_cast<llama_token>(token_ids[i]);
        batch = llama_batch_get_one(&next, 1);
        require(llama_decode(context.get(), batch) == 0, "llama.cpp reference decode failed");
        result.logits.push_back(copy_logits(-1));
    }
    require(!result.first_layer_output.empty(), "llama.cpp did not expose the first-layer activation");
    return result;
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
        const auto prompt_tokens = model->tokenize("Hello from IntelHive");
        require(!prompt_tokens.empty() &&
                std::all_of(prompt_tokens.begin(), prompt_tokens.end(), [&](int32_t token) {
                    return token >= 0 && token < config.vocabulary_size;
                }), "GGUF tokenizer produced invalid token IDs");

        const std::vector<uint32_t> token_ids{17, 42, 1337, 314, 2718, 5};
        const auto reference = llama_reference_logits(argv[1], token_ids, 3);
        const int32_t layers_per_shard = config.layer_count / 3;
        require(config.layer_count % 3 == 0, "test expects three equal layer ranges");
        intelhive::Qwen2GgufShard full(model, 0, config.layer_count - 1);
        intelhive::Qwen2GgufShard first(model, 0, layers_per_shard - 1);
        intelhive::Qwen2GgufShard second(model, layers_per_shard, 2 * layers_per_shard - 1);
        intelhive::Qwen2GgufShard third(model, 2 * layers_per_shard, config.layer_count - 1);

        std::vector<float> prefill;
        for (size_t i = 0; i < 3; ++i) append(prefill, model->embed_token(token_ids[i]));
        intelhive::Qwen2GgufShard reference_layer(model, 0, 0);
        const auto first_layer_prefill = reference_layer.execute(prefill, 0);
        const float first_layer_reference_error = max_abs_error(reference.first_layer_output, first_layer_prefill);
        require(first_layer_reference_error <= 0.02f,
            "llama.cpp first-layer activation error exceeds 0.02 (max_abs_error=" +
            std::to_string(first_layer_reference_error) + ")");
        const auto full_prefill = full.execute(prefill, 0);
        const auto first_prefill = first.execute(prefill, 0);
        const auto second_prefill = second.execute(first_prefill, 0);
        const auto split_prefill = third.execute(second_prefill, 0);
        compare(full_prefill, split_prefill);
        const size_t last_hidden_offset = (3 - 1) * static_cast<size_t>(config.embedding_size);
        const std::vector<float> full_prefill_hidden(
            full_prefill.begin() + last_hidden_offset,
            full_prefill.begin() + last_hidden_offset + config.embedding_size);
        const std::vector<float> split_prefill_hidden(
            split_prefill.begin() + last_hidden_offset,
            split_prefill.begin() + last_hidden_offset + config.embedding_size);
        const auto full_prefill_logits = model->project_logits(full_prefill_hidden);
        const auto split_prefill_logits = model->project_logits(split_prefill_hidden);
        compare(full_prefill_logits, split_prefill_logits);
        const float prefill_reference_error = compare_reference(reference.logits[0], full_prefill_logits);
        compare_reference(reference.logits[0], split_prefill_logits);
        float max_reference_logit_error = prefill_reference_error;

        std::vector<float> full_decode_logits;
        std::vector<float> split_decode_logits;
        for (size_t i = 3; i < token_ids.size(); ++i) {
            const auto token = model->embed_token(token_ids[i]);
            const auto full_output = full.execute(token, static_cast<int32_t>(i));
            const auto a = first.execute(token, static_cast<int32_t>(i));
            const auto b = second.execute(a, static_cast<int32_t>(i));
            const auto split_output = third.execute(b, static_cast<int32_t>(i));
            compare(full_output, split_output);
            full_decode_logits = model->project_logits(full_output);
            split_decode_logits = model->project_logits(split_output);
            compare(full_decode_logits, split_decode_logits);
            max_reference_logit_error = std::max(max_reference_logit_error,
                compare_reference(reference.logits[i - 2], full_decode_logits));
            max_reference_logit_error = std::max(max_reference_logit_error,
                compare_reference(reference.logits[i - 2], split_decode_logits));
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

        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            throw std::runtime_error("failed to read process resource usage");
        }
        std::cout << "PASS: real GGUF Qwen2 layer weights, three-shard hidden/logit parity, "
                  << "independent llama.cpp logit parity, prefill/decode, sequence reset, and KV positions\n"
                  << "Logits: vocabulary=" << config.vocabulary_size
                  << ", final decode argmax=" << argmax(full_decode_logits)
                  << ", first-layer max reference error=" << first_layer_reference_error
                  << ", max reference logit error=" << max_reference_logit_error << '\n'
                  << "Mapped layer weight bytes: full=" << full.mapped_weight_bytes()
                  << ", per-shard=" << first.mapped_weight_bytes() << ','
                  << second.mapped_weight_bytes() << ',' << third.mapped_weight_bytes() << '\n'
                  << "Peak process RSS bytes: "
#if defined(__APPLE__)
                  << usage.ru_maxrss
#else
                  << static_cast<uint64_t>(usage.ru_maxrss) * 1024
#endif
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
