#include "qwen2_gguf.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace intelhive {
namespace {

using GgmlContext = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

GgmlContext create_context(size_t bytes, bool no_alloc = false) {
    auto* context = ggml_init({bytes, nullptr, no_alloc});
    if (context == nullptr) throw std::runtime_error("ggml context allocation failed");
    return GgmlContext(context, ggml_free);
}

void run_graph(ggml_context* context, ggml_tensor* output) {
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(context, graph, 1) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml graph execution failed");
    }
}

uint32_t read_u32(const gguf_context* metadata, const std::string& key) {
    const int64_t id = gguf_find_key(metadata, key.c_str());
    if (id < 0) throw std::runtime_error("GGUF metadata key missing: " + key);
    return gguf_get_val_u32(metadata, id);
}

float read_f32(const gguf_context* metadata, const std::string& key) {
    const int64_t id = gguf_find_key(metadata, key.c_str());
    if (id < 0) throw std::runtime_error("GGUF metadata key missing: " + key);
    return gguf_get_val_f32(metadata, id);
}

struct TensorDescriptor {
    ggml_type type;
    size_t bytes;
    size_t offset;
    int64_t dimensions[2];
};

}

struct Qwen2GgufModel::Impl {
    explicit Impl(std::string model_path) : model_path(std::move(model_path)) {
        try {
            file_descriptor = open(this->model_path.c_str(), O_RDONLY);
            if (file_descriptor < 0) {
                throw std::runtime_error("open GGUF model failed: " + this->model_path + ": " + std::strerror(errno));
            }

            struct stat status {};
            if (fstat(file_descriptor, &status) != 0 || status.st_size <= 0) {
                throw std::runtime_error("stat GGUF model failed: " + this->model_path + ": " + std::strerror(errno));
            }
            mapped_size = static_cast<size_t>(status.st_size);
            mapped_data = static_cast<const uint8_t*>(mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, file_descriptor, 0));
            if (mapped_data == MAP_FAILED) {
                mapped_data = nullptr;
                throw std::runtime_error("map GGUF model failed: " + this->model_path + ": " + std::strerror(errno));
            }

            gguf_init_params params {true, nullptr};
            metadata = gguf_init_from_file(this->model_path.c_str(), params);
            if (metadata == nullptr) throw std::runtime_error("parse GGUF metadata failed: " + this->model_path);

            const int64_t architecture_id = gguf_find_key(metadata, "general.architecture");
            if (architecture_id < 0 || std::string(gguf_get_val_str(metadata, architecture_id)) != "qwen2") {
                throw std::runtime_error("GGUF architecture must be qwen2");
            }

            config.layer_count = static_cast<int32_t>(read_u32(metadata, "qwen2.block_count"));
            config.embedding_size = static_cast<int32_t>(read_u32(metadata, "qwen2.embedding_length"));
            config.feed_forward_size = static_cast<int32_t>(read_u32(metadata, "qwen2.feed_forward_length"));
            config.attention_heads = static_cast<int32_t>(read_u32(metadata, "qwen2.attention.head_count"));
            config.kv_heads = static_cast<int32_t>(read_u32(metadata, "qwen2.attention.head_count_kv"));
            config.context_length = static_cast<int32_t>(read_u32(metadata, "qwen2.context_length"));
            config.rms_norm_epsilon = read_f32(metadata, "qwen2.attention.layer_norm_rms_epsilon");
            config.rope_frequency_base = read_f32(metadata, "qwen2.rope.freq_base");

            if (config.layer_count <= 0 || config.embedding_size <= 0 || config.feed_forward_size <= 0 ||
                config.attention_heads <= 0 || config.kv_heads <= 0 || config.context_length <= 0 ||
                config.embedding_size % config.attention_heads != 0 ||
                config.attention_heads % config.kv_heads != 0 ||
                !std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0f ||
                !std::isfinite(config.rope_frequency_base) || config.rope_frequency_base <= 0.0f) {
                throw std::runtime_error("invalid Qwen2 GGUF architecture metadata");
            }

            const int64_t embedding_id = gguf_find_tensor(metadata, "token_embd.weight");
            if (embedding_id < 0) throw std::runtime_error("GGUF token embedding tensor missing");
            const int64_t* embedding_shape = gguf_get_tensor_ne(metadata, embedding_id);
            if (embedding_shape[0] != config.embedding_size || embedding_shape[1] <= 0) {
                throw std::runtime_error("GGUF token embedding shape does not match architecture metadata");
            }
            config.vocabulary_size = static_cast<int32_t>(embedding_shape[1]);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() {
        cleanup();
    }

    void cleanup() noexcept {
        if (metadata != nullptr) gguf_free(metadata);
        metadata = nullptr;
        if (mapped_data != nullptr) munmap(const_cast<uint8_t*>(mapped_data), mapped_size);
        mapped_data = nullptr;
        if (file_descriptor >= 0) close(file_descriptor);
        file_descriptor = -1;
    }

    TensorDescriptor tensor_descriptor(const std::string& name, int64_t dim0, int64_t dim1) const {
        const int64_t id = gguf_find_tensor(metadata, name.c_str());
        if (id < 0) throw std::runtime_error("GGUF tensor missing: " + name);
        const int64_t* shape = gguf_get_tensor_ne(metadata, id);
        if (shape[0] != dim0 || shape[1] != dim1 || shape[2] != 1 || shape[3] != 1) {
            throw std::runtime_error("GGUF tensor shape mismatch: " + name);
        }

        const size_t data_offset = gguf_get_data_offset(metadata);
        const size_t tensor_offset = gguf_get_tensor_offset(metadata, id);
        const size_t bytes = gguf_get_tensor_size(metadata, id);
        if (tensor_offset > mapped_size || data_offset > mapped_size - tensor_offset ||
            bytes > mapped_size - data_offset - tensor_offset) {
            throw std::runtime_error("GGUF tensor data lies outside the model file: " + name);
        }
        return {gguf_get_tensor_type(metadata, id), bytes, data_offset + tensor_offset, {dim0, dim1}};
    }

    ggml_tensor* map_tensor(
            ggml_context* context,
            const std::string& name,
            int64_t dim0,
            int64_t dim1) const {
        const TensorDescriptor descriptor = tensor_descriptor(name, dim0, dim1);
        auto* tensor = ggml_new_tensor_2d(context, descriptor.type, dim0, dim1);
        if (ggml_nbytes(tensor) != descriptor.bytes) {
            throw std::runtime_error("GGUF tensor byte size mismatch: " + name);
        }
        tensor->data = const_cast<uint8_t*>(mapped_data + descriptor.offset);
        ggml_set_name(tensor, name.c_str());
        return tensor;
    }

    std::string model_path;
    int file_descriptor = -1;
    const uint8_t* mapped_data = nullptr;
    size_t mapped_size = 0;
    gguf_context* metadata = nullptr;
    Qwen2Config config {};
};

struct Qwen2GgufShard::Impl {
    struct LayerWeights {
        ggml_tensor* attention_norm;
        ggml_tensor* query;
        ggml_tensor* key;
        ggml_tensor* value;
        ggml_tensor* attention_output;
        ggml_tensor* feed_forward_norm;
        ggml_tensor* feed_forward_gate;
        ggml_tensor* feed_forward_up;
        ggml_tensor* feed_forward_down;
    };

    struct LayerCache {
        std::vector<float> keys;
        std::vector<float> values;
    };

    struct LayerResult {
        std::vector<float> activation;
        std::vector<float> rotated_key;
        std::vector<float> value;
    };

    LayerResult execute_layer(
            const LayerWeights& weights,
            const LayerCache& cache,
            const std::vector<float>& input,
            int32_t position,
            const Qwen2Config& config) const {
        const int32_t head_dimension = config.embedding_size / config.attention_heads;
        const int32_t previous_tokens = static_cast<int32_t>(
            cache.keys.size() / static_cast<size_t>(head_dimension * config.kv_heads));

        auto context = create_context(32 * 1024 * 1024);
        auto* ctx = context.get();
        auto* residual_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.embedding_size);
        std::memcpy(residual_input->data, input.data(), input.size() * sizeof(float));

        auto* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        static_cast<int32_t*>(positions->data)[0] = position;

        auto* attention_norm = ggml_mul(ctx,
            ggml_rms_norm(ctx, residual_input, config.rms_norm_epsilon),
            weights.attention_norm);
        auto* query = ggml_mul_mat(ctx, weights.query, attention_norm);
        auto* key = ggml_mul_mat(ctx, weights.key, attention_norm);
        auto* value = ggml_mul_mat(ctx, weights.value, attention_norm);
        query = ggml_reshape_3d(ctx, query, head_dimension, config.attention_heads, 1);
        key = ggml_reshape_3d(ctx, key, head_dimension, config.kv_heads, 1);
        value = ggml_reshape_3d(ctx, value, head_dimension, config.kv_heads, 1);

        constexpr float rope_scale = 1.0f;
        constexpr float rope_extension = 0.0f;
        constexpr float rope_attention = 1.0f;
        constexpr float rope_beta_fast = 32.0f;
        constexpr float rope_beta_slow = 1.0f;
        query = ggml_rope_ext(ctx, query, positions, nullptr, head_dimension, GGML_ROPE_TYPE_NORMAL,
            config.context_length, config.rope_frequency_base, rope_scale, rope_extension,
            rope_attention, rope_beta_fast, rope_beta_slow);
        key = ggml_rope_ext(ctx, key, positions, nullptr, head_dimension, GGML_ROPE_TYPE_NORMAL,
            config.context_length, config.rope_frequency_base, rope_scale, rope_extension,
            rope_attention, rope_beta_fast, rope_beta_slow);

        ggml_tensor* all_keys = key;
        ggml_tensor* all_values = value;
        if (previous_tokens > 0) {
            auto* old_keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dimension, config.kv_heads, previous_tokens);
            auto* old_values = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dimension, config.kv_heads, previous_tokens);
            std::memcpy(old_keys->data, cache.keys.data(), cache.keys.size() * sizeof(float));
            std::memcpy(old_values->data, cache.values.data(), cache.values.size() * sizeof(float));
            all_keys = ggml_concat(ctx, old_keys, key, 2);
            all_values = ggml_concat(ctx, old_values, value, 2);
        }

        auto* q_for_attention = ggml_permute(ctx, query, 0, 2, 1, 3);
        auto* k_for_attention = ggml_permute(ctx, all_keys, 0, 2, 1, 3);
        auto* v_for_attention = ggml_permute(ctx, all_values, 0, 2, 1, 3);
        if (k_for_attention->type == GGML_TYPE_F32) {
            k_for_attention = ggml_cast(ctx, k_for_attention, GGML_TYPE_F16);
        }
        if (v_for_attention->type == GGML_TYPE_F32) {
            v_for_attention = ggml_cast(ctx, v_for_attention, GGML_TYPE_F16);
        }
        auto* attended = ggml_flash_attn_ext(ctx, q_for_attention, k_for_attention,
            v_for_attention, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dimension)), 0.0f, 0.0f);
        ggml_prec_set_acc(attended, GGML_PREC_F32);
        attended = ggml_reshape_2d(ctx, attended, config.embedding_size, 1);

        auto* attention_output = ggml_mul_mat(ctx, weights.attention_output, attended);
        auto* attention_residual = ggml_add(ctx, residual_input, attention_output);
        auto* feed_forward_norm = ggml_mul(ctx,
            ggml_rms_norm(ctx, attention_residual, config.rms_norm_epsilon),
            weights.feed_forward_norm);
        auto* gate = ggml_silu(ctx, ggml_mul_mat(ctx, weights.feed_forward_gate, feed_forward_norm));
        auto* up = ggml_mul_mat(ctx, weights.feed_forward_up, feed_forward_norm);
        auto* down = ggml_mul_mat(ctx, weights.feed_forward_down, ggml_mul(ctx, gate, up));
        auto* output = ggml_add(ctx, attention_residual, down);

        run_graph(ctx, output);
        const auto* output_data = static_cast<const float*>(output->data);
        const auto* key_data = static_cast<const float*>(key->data);
        const auto* value_data = static_cast<const float*>(value->data);
        return {
            {output_data, output_data + config.embedding_size},
            {key_data, key_data + head_dimension * config.kv_heads},
            {value_data, value_data + head_dimension * config.kv_heads},
        };
    }

    std::shared_ptr<const Qwen2GgufModel> model;
    GgmlContext weights_context {nullptr, ggml_free};
    std::vector<LayerWeights> weights;
    std::vector<LayerCache> cache;
    int32_t first = 0;
    int32_t last = -1;
    int32_t token_count = 0;
    size_t weight_bytes = 0;
};

Qwen2GgufModel::Qwen2GgufModel(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}
Qwen2GgufModel::~Qwen2GgufModel() = default;
Qwen2GgufModel::Qwen2GgufModel(Qwen2GgufModel&&) noexcept = default;
Qwen2GgufModel& Qwen2GgufModel::operator=(Qwen2GgufModel&&) noexcept = default;

const Qwen2Config& Qwen2GgufModel::config() const {
    return impl_->config;
}

const std::string& Qwen2GgufModel::path() const {
    return impl_->model_path;
}

std::vector<float> Qwen2GgufModel::embed_token(uint32_t token_id) const {
    if (token_id >= static_cast<uint32_t>(impl_->config.vocabulary_size)) {
        throw std::invalid_argument("token ID is outside the GGUF vocabulary");
    }
    auto weight_context = create_context(1024 * 1024, true);
    auto* embedding = impl_->map_tensor(weight_context.get(), "token_embd.weight",
        impl_->config.embedding_size, impl_->config.vocabulary_size);
    auto graph_context = create_context(4 * 1024 * 1024);
    auto* ids = ggml_new_tensor_1d(graph_context.get(), GGML_TYPE_I32, 1);
    static_cast<int32_t*>(ids->data)[0] = static_cast<int32_t>(token_id);
    auto* row = ggml_get_rows(graph_context.get(), embedding, ids);
    run_graph(graph_context.get(), row);
    const auto* data = static_cast<const float*>(row->data);
    return {data, data + impl_->config.embedding_size};
}

Qwen2GgufShard::Qwen2GgufShard(
        std::shared_ptr<const Qwen2GgufModel> model,
        int32_t first_layer,
        int32_t last_layer) : impl_(std::make_unique<Impl>()) {
    if (!model) throw std::invalid_argument("GGUF model is required");
    const auto& config = model->config();
    if (first_layer < 0 || last_layer < first_layer || last_layer >= config.layer_count) {
        throw std::invalid_argument("invalid inclusive Qwen2 layer range");
    }

    impl_->model = std::move(model);
    impl_->first = first_layer;
    impl_->last = last_layer;
    impl_->weights_context = create_context(4 * 1024 * 1024, true);
    impl_->weights.reserve(static_cast<size_t>(last_layer - first_layer + 1));
    impl_->cache.resize(static_cast<size_t>(last_layer - first_layer + 1));
    const auto& model_impl = *impl_->model->impl_;

    const int32_t key_value_width =
        config.embedding_size / config.attention_heads * config.kv_heads;
    for (int32_t layer = first_layer; layer <= last_layer; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        Impl::LayerWeights layer_weights {
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_norm.weight",
                config.embedding_size, 1),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_q.weight",
                config.embedding_size, config.embedding_size),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_k.weight",
                config.embedding_size, key_value_width),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_v.weight",
                config.embedding_size, key_value_width),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_output.weight",
                config.embedding_size, config.embedding_size),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_norm.weight",
                config.embedding_size, 1),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_gate.weight",
                config.embedding_size, config.feed_forward_size),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_up.weight",
                config.embedding_size, config.feed_forward_size),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_down.weight",
                config.feed_forward_size, config.embedding_size),
        };
        const ggml_tensor* tensors[] = {
            layer_weights.attention_norm, layer_weights.query, layer_weights.key, layer_weights.value,
            layer_weights.attention_output, layer_weights.feed_forward_norm,
            layer_weights.feed_forward_gate, layer_weights.feed_forward_up, layer_weights.feed_forward_down,
        };
        for (const auto* tensor : tensors) impl_->weight_bytes += ggml_nbytes(tensor);
        impl_->weights.push_back(layer_weights);
    }
}

Qwen2GgufShard::~Qwen2GgufShard() = default;
Qwen2GgufShard::Qwen2GgufShard(Qwen2GgufShard&&) noexcept = default;
Qwen2GgufShard& Qwen2GgufShard::operator=(Qwen2GgufShard&&) noexcept = default;

std::vector<float> Qwen2GgufShard::execute(
        const std::vector<float>& token_major_input,
        int32_t token_offset) {
    const auto& config = impl_->model->config();
    if (token_major_input.empty() ||
        token_major_input.size() % static_cast<size_t>(config.embedding_size) != 0 ||
        token_offset != impl_->token_count) {
        throw std::invalid_argument("invalid activation shape or Qwen2 KV position");
    }
    const size_t token_count = token_major_input.size() / static_cast<size_t>(config.embedding_size);
    if (token_count > static_cast<size_t>(config.context_length - impl_->token_count)) {
        throw std::invalid_argument("Qwen2 context capacity exceeded");
    }
    for (float value : token_major_input) {
        if (!std::isfinite(value)) throw std::invalid_argument("activation contains a non-finite value");
    }

    std::vector<float> result;
    result.reserve(token_major_input.size());
    const int32_t head_dimension = config.embedding_size / config.attention_heads;
    const size_t cache_append_size = static_cast<size_t>(head_dimension * config.kv_heads);
    for (size_t token = 0; token < token_count; ++token) {
        const auto begin = token_major_input.begin() + token * config.embedding_size;
        std::vector<float> activation(begin, begin + config.embedding_size);
        std::vector<std::vector<float>> pending_keys(impl_->cache.size());
        std::vector<std::vector<float>> pending_values(impl_->cache.size());

        for (size_t index = 0; index < impl_->weights.size(); ++index) {
            const auto layer_result = impl_->execute_layer(impl_->weights[index], impl_->cache[index],
                activation, impl_->token_count, config);
            activation = layer_result.activation;
            pending_keys[index] = layer_result.rotated_key;
            pending_values[index] = layer_result.value;
        }

        for (auto& layer_cache : impl_->cache) {
            layer_cache.keys.reserve(layer_cache.keys.size() + cache_append_size);
            layer_cache.values.reserve(layer_cache.values.size() + cache_append_size);
        }
        for (size_t index = 0; index < impl_->cache.size(); ++index) {
            auto& layer_cache = impl_->cache[index];
            layer_cache.keys.insert(layer_cache.keys.end(), pending_keys[index].begin(), pending_keys[index].end());
            layer_cache.values.insert(layer_cache.values.end(), pending_values[index].begin(), pending_values[index].end());
        }

        result.insert(result.end(), activation.begin(), activation.end());
        ++impl_->token_count;
    }
    return result;
}

void Qwen2GgufShard::reset() {
    impl_->token_count = 0;
    for (auto& layer_cache : impl_->cache) {
        layer_cache.keys.clear();
        layer_cache.values.clear();
    }
}

int32_t Qwen2GgufShard::cached_tokens() const {
    return impl_->token_count;
}

size_t Qwen2GgufShard::mapped_weight_bytes() const {
    return impl_->weight_bytes;
}

int32_t Qwen2GgufShard::first_layer() const {
    return impl_->first;
}

int32_t Qwen2GgufShard::last_layer() const {
    return impl_->last;
}

}
