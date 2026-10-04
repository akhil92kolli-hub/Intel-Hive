#include "qwen2_gguf.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "ggml-backend.h"
#ifdef INTELHIVE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace intelhive {
namespace {

using GgmlContext = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using BackendHandle = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using BackendBuffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using BackendScheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
using ExecutionDeadline = std::optional<std::chrono::steady_clock::time_point>;

ExecutionDeadline execution_deadline(int64_t max_duration_ms) {
    if (max_duration_ms <= 0) return std::nullopt;
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(max_duration_ms);
}

bool deadline_expired(const ExecutionDeadline& deadline) {
    return deadline.has_value() && std::chrono::steady_clock::now() >= *deadline;
}

void require_before_deadline(const ExecutionDeadline& deadline) {
    if (deadline_expired(deadline)) {
        throw std::runtime_error("Native execution deadline exceeded");
    }
}

struct SchedulerDeadlineState {
    ExecutionDeadline deadline;
    size_t nodes_asked = 0;
    bool expired = false;
};

bool scheduler_deadline_callback(ggml_tensor*, bool ask, void* user_data) {
    auto& state = *static_cast<SchedulerDeadlineState*>(user_data);
    if (ask) {
        ++state.nodes_asked;
        state.expired = deadline_expired(state.deadline);
        return state.expired || state.nodes_asked % 32 == 0;
    }
    state.expired = deadline_expired(state.deadline);
    return !state.expired;
}

GgmlContext create_context(size_t bytes, bool no_alloc = false) {
    auto* context = ggml_init({bytes, nullptr, no_alloc});
    if (context == nullptr) throw std::runtime_error("ggml context allocation failed");
    return GgmlContext(context, ggml_free);
}

int graph_thread_count(int requested) {
    if (requested > 0) return requested;
    const unsigned int available = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1U, std::min(available == 0 ? 1U : available, 8U)));
}

void run_graph(ggml_context* context, ggml_tensor* output, int graph_threads = 0) {
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(context, graph, graph_thread_count(graph_threads)) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml graph execution failed");
    }
}

class GraphExecutor {
public:
    GraphExecutor(InferenceBackend kind, int32_t graph_threads) : kind_(kind), graph_threads_(graph_threads) {
        if (kind_ == InferenceBackend::Cpu) return;
#ifdef INTELHIVE_HAS_VULKAN
        if (ggml_backend_vk_get_device_count() <= 0) {
            throw std::runtime_error("GGML Vulkan backend found no usable Vulkan device");
        }
        vulkan_.reset(ggml_backend_vk_init(0));
        cpu_.reset(ggml_backend_cpu_init());
        if (!vulkan_ || !cpu_) {
            throw std::runtime_error("could not initialize GGML Vulkan and CPU backends");
        }
        ggml_backend_cpu_set_n_threads(cpu_.get(), graph_thread_count(graph_threads_));
        ggml_backend_t backends[] = {vulkan_.get(), cpu_.get()};
        scheduler_.reset(ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        if (!scheduler_) throw std::runtime_error("could not create GGML Vulkan backend scheduler");
#else
        (void) graph_threads_;
        throw std::runtime_error("This build does not include the GGML Vulkan backend");
#endif
    }

    void allocate_weights(
            ggml_context* context,
            const std::function<const void*(const ggml_tensor*)>& resolve_source,
            const ExecutionDeadline& deadline) {
#ifdef INTELHIVE_HAS_VULKAN
        if (kind_ != InferenceBackend::Vulkan) return;
        require_before_deadline(deadline);
        BackendBuffer buffer(ggml_backend_alloc_ctx_tensors(context, vulkan_.get()),
            ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("could not allocate Vulkan shard weight buffer");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto* tensor = ggml_get_first_tensor(context); tensor != nullptr;
                tensor = ggml_get_next_tensor(context, tensor)) {
            require_before_deadline(deadline);
            const auto* source = resolve_source(tensor);
            if (source == nullptr) throw std::runtime_error("could not resolve mapped GGUF tensor data");
            ggml_backend_tensor_set(tensor, source, 0, ggml_nbytes(tensor));
        }
        require_before_deadline(deadline);
        weight_buffer_ = std::move(buffer);
        gpu_weight_bytes_ = ggml_backend_buffer_get_size(weight_buffer_.get());
#else
        (void) context;
        (void) resolve_source;
#endif
    }

    void compute(
            ggml_context* context,
            ggml_tensor* output,
            const ExecutionDeadline& deadline) {
        require_before_deadline(deadline);
        auto* graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        if (kind_ == InferenceBackend::Cpu) {
            cpu_graph_nodes_ += static_cast<uint64_t>(ggml_graph_n_nodes(graph));
            if (ggml_graph_compute_with_ctx(
                    context, graph, graph_thread_count(graph_threads_)) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml graph execution failed");
            }
            require_before_deadline(deadline);
            return;
        }
#ifdef INTELHIVE_HAS_VULKAN
        ggml_backend_sched_reset(scheduler_.get());
        ggml_backend_sched_split_graph(scheduler_.get(), graph);
        bool graph_uses_vulkan = false;
        const int node_count = ggml_graph_n_nodes(graph);
        for (int index = 0; index < node_count; ++index) {
            const bool vulkan_node = ggml_backend_is_vk(ggml_backend_sched_get_tensor_backend(
                scheduler_.get(), ggml_graph_node(graph, index)));
            graph_uses_vulkan = graph_uses_vulkan || vulkan_node;
            if (vulkan_node) ++vulkan_graph_nodes_;
            else ++cpu_graph_nodes_;
        }
        if (!graph_uses_vulkan) {
            ggml_backend_sched_reset(scheduler_.get());
            throw std::runtime_error("GGML scheduled this graph entirely on CPU; no Vulkan inference was measured");
        }
        SchedulerDeadlineState deadline_state {deadline};
        if (deadline.has_value()) {
            ggml_backend_sched_set_eval_callback(
                scheduler_.get(), scheduler_deadline_callback, &deadline_state);
        }
        const auto status = ggml_backend_sched_graph_compute(scheduler_.get(), graph);
        if (deadline.has_value()) {
            ggml_backend_sched_set_eval_callback(scheduler_.get(), nullptr, nullptr);
        }
        if (deadline_state.expired) {
            ggml_backend_sched_reset(scheduler_.get());
            throw std::runtime_error("Native execution deadline exceeded");
        }
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_reset(scheduler_.get());
            throw std::runtime_error("GGML Vulkan graph execution failed");
        }
        used_vulkan_ = true;
        graph_pending_ = true;
#endif
    }

    std::vector<float> read_f32(
            const ggml_tensor* tensor,
            const ExecutionDeadline& deadline) {
        require_before_deadline(deadline);
        std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
#ifdef INTELHIVE_HAS_VULKAN
        if (kind_ == InferenceBackend::Vulkan) {
            activation_transfer_bytes_ += static_cast<uint64_t>(values.size() * sizeof(float));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            require_before_deadline(deadline);
            return values;
        }
#endif
        const auto* data = static_cast<const float*>(tensor->data);
        return {data, data + values.size()};
    }

    void finish() {
#ifdef INTELHIVE_HAS_VULKAN
        if (kind_ == InferenceBackend::Vulkan && graph_pending_) {
            ggml_backend_sched_synchronize(scheduler_.get());
            ggml_backend_sched_reset(scheduler_.get());
            graph_pending_ = false;
        }
#endif
    }

    bool used_vulkan() const { return used_vulkan_; }

    BackendExecutionStats stats() const {
        BackendExecutionStats value;
        value.requested_backend = kind_;
        value.cpu_graph_nodes = cpu_graph_nodes_;
        value.vulkan_graph_nodes = vulkan_graph_nodes_;
        value.gpu_weight_bytes = gpu_weight_bytes_;
        value.activation_transfer_bytes = activation_transfer_bytes_;
        value.fallback_used = kind_ == InferenceBackend::Vulkan && cpu_graph_nodes_ > 0;
        return value;
    }

private:
    InferenceBackend kind_;
    int32_t graph_threads_;
    bool used_vulkan_ = false;
    bool graph_pending_ = false;
    uint64_t cpu_graph_nodes_ = 0;
    uint64_t vulkan_graph_nodes_ = 0;
    uint64_t gpu_weight_bytes_ = 0;
    uint64_t activation_transfer_bytes_ = 0;
    BackendHandle vulkan_ {nullptr, ggml_backend_free};
    BackendHandle cpu_ {nullptr, ggml_backend_free};
    BackendScheduler scheduler_ {nullptr, ggml_backend_sched_free};
    BackendBuffer weight_buffer_ {nullptr, ggml_backend_buffer_free};
};

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
        if (tokenizer_model != nullptr) llama_model_free(tokenizer_model);
        tokenizer_model = nullptr;
        if (metadata != nullptr) gguf_free(metadata);
        metadata = nullptr;
        if (mapped_data != nullptr) munmap(const_cast<uint8_t*>(mapped_data), mapped_size);
        mapped_data = nullptr;
        if (file_descriptor >= 0) close(file_descriptor);
        file_descriptor = -1;
    }

    const llama_vocab* tokenizer_vocab() const {
        std::lock_guard<std::mutex> lock(tokenizer_mutex);
        if (tokenizer_model == nullptr) {
            static std::once_flag backend_once;
            std::call_once(backend_once, llama_backend_init);
            auto params = llama_model_default_params();
            params.vocab_only = true;
            tokenizer_model = llama_model_load_from_file(model_path.c_str(), params);
            if (tokenizer_model == nullptr) {
                throw std::runtime_error("load GGUF tokenizer vocabulary failed: " + model_path);
            }
        }
        return llama_model_get_vocab(tokenizer_model);
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
            int64_t dim1,
            bool backend_owned = false) const {
        const TensorDescriptor descriptor = tensor_descriptor(name, dim0, dim1);
        auto* tensor = ggml_new_tensor_2d(context, descriptor.type, dim0, dim1);
        if (ggml_nbytes(tensor) != descriptor.bytes) {
            throw std::runtime_error("GGUF tensor byte size mismatch: " + name);
        }
        if (!backend_owned) tensor->data = const_cast<uint8_t*>(mapped_data + descriptor.offset);
        ggml_set_name(tensor, name.c_str());
        return tensor;
    }

    ggml_tensor* map_optional_vector(
            ggml_context* context,
            const std::string& name,
            int64_t length,
            bool backend_owned = false) const {
        const int64_t id = gguf_find_tensor(metadata, name.c_str());
        if (id < 0) return nullptr;

        const int64_t* shape = gguf_get_tensor_ne(metadata, id);
        if (shape[0] != length || shape[1] != 1 || shape[2] != 1 || shape[3] != 1) {
            throw std::runtime_error("GGUF tensor shape mismatch: " + name);
        }

        const size_t data_offset = gguf_get_data_offset(metadata);
        const size_t tensor_offset = gguf_get_tensor_offset(metadata, id);
        const size_t bytes = gguf_get_tensor_size(metadata, id);
        if (tensor_offset > mapped_size || data_offset > mapped_size - tensor_offset ||
            bytes > mapped_size - data_offset - tensor_offset) {
            throw std::runtime_error("GGUF tensor data lies outside the model file: " + name);
        }

        auto* tensor = ggml_new_tensor_1d(context, gguf_get_tensor_type(metadata, id), length);
        if (ggml_nbytes(tensor) != bytes) {
            throw std::runtime_error("GGUF tensor byte size mismatch: " + name);
        }
        if (!backend_owned) {
            tensor->data = const_cast<uint8_t*>(mapped_data + data_offset + tensor_offset);
        }
        ggml_set_name(tensor, name.c_str());
        return tensor;
    }

    const void* mapped_tensor_data(const ggml_tensor* tensor) const {
        const int64_t id = gguf_find_tensor(metadata, ggml_get_name(tensor));
        if (id < 0) return nullptr;
        return mapped_data + gguf_get_data_offset(metadata) +
            gguf_get_tensor_offset(metadata, id);
    }

    std::string model_path;
    int file_descriptor = -1;
    const uint8_t* mapped_data = nullptr;
    size_t mapped_size = 0;
    gguf_context* metadata = nullptr;
    mutable std::mutex tokenizer_mutex;
    mutable llama_model* tokenizer_model = nullptr;
    Qwen2Config config {};
};

struct Qwen2GgufShard::Impl {
    struct LayerWeights {
        ggml_tensor* attention_norm;
        ggml_tensor* query;
        ggml_tensor* query_bias;
        ggml_tensor* key;
        ggml_tensor* key_bias;
        ggml_tensor* value;
        ggml_tensor* value_bias;
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

    struct LayerGraphResult {
        ggml_tensor* activation;
        ggml_tensor* rotated_key;
        ggml_tensor* value;
    };

    LayerGraphResult build_layer(
            ggml_context* ctx,
            const LayerWeights& weights,
            const LayerCache& cache,
            ggml_tensor* residual_input,
            int32_t position,
            const Qwen2Config& config) const {
        const int32_t head_dimension = config.embedding_size / config.attention_heads;
        const int32_t previous_tokens = static_cast<int32_t>(
            cache.keys.size() / static_cast<size_t>(head_dimension * config.kv_heads));

        auto* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        static_cast<int32_t*>(positions->data)[0] = position;

        auto* attention_norm = ggml_mul(ctx,
            ggml_rms_norm(ctx, residual_input, config.rms_norm_epsilon),
            weights.attention_norm);
        auto* query = ggml_mul_mat(ctx, weights.query, attention_norm);
        auto* key = ggml_mul_mat(ctx, weights.key, attention_norm);
        auto* value = ggml_mul_mat(ctx, weights.value, attention_norm);
        if (weights.query_bias != nullptr) query = ggml_add(ctx, query, weights.query_bias);
        if (weights.key_bias != nullptr) key = ggml_add(ctx, key, weights.key_bias);
        if (weights.value_bias != nullptr) value = ggml_add(ctx, value, weights.value_bias);
        query = ggml_reshape_3d(ctx, query, head_dimension, config.attention_heads, 1);
        key = ggml_reshape_3d(ctx, key, head_dimension, config.kv_heads, 1);
        value = ggml_reshape_3d(ctx, value, head_dimension, config.kv_heads, 1);

        constexpr float rope_scale = 1.0f;
        constexpr float rope_extension = 0.0f;
        constexpr float rope_attention = 1.0f;
        constexpr float rope_beta_fast = 32.0f;
        constexpr float rope_beta_slow = 1.0f;
        query = ggml_rope_ext(ctx, query, positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX,
            config.context_length, config.rope_frequency_base, rope_scale, rope_extension,
            rope_attention, rope_beta_fast, rope_beta_slow);
        key = ggml_rope_ext(ctx, key, positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX,
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
        if (k_for_attention->ne[0] != q_for_attention->ne[0] ||
            q_for_attention->ne[2] % k_for_attention->ne[2] != 0 ||
            q_for_attention->ne[3] % k_for_attention->ne[3] != 0) {
            throw std::runtime_error("Qwen2 attention score shapes are incompatible: k=[" +
                std::to_string(k_for_attention->ne[0]) + "," +
                std::to_string(k_for_attention->ne[1]) + "," +
                std::to_string(k_for_attention->ne[2]) + "," +
                std::to_string(k_for_attention->ne[3]) + "], q=[" +
                std::to_string(q_for_attention->ne[0]) + "," +
                std::to_string(q_for_attention->ne[1]) + "," +
                std::to_string(q_for_attention->ne[2]) + "," +
                std::to_string(q_for_attention->ne[3]) + "]");
        }
        auto* attention_scores = ggml_mul_mat(ctx, k_for_attention, q_for_attention);
        ggml_prec_set_acc(attention_scores, GGML_PREC_F32);
        auto* attention_probabilities = ggml_soft_max_ext(ctx, attention_scores, nullptr,
            1.0f / std::sqrt(static_cast<float>(head_dimension)), 0.0f);
        v_for_attention = ggml_cont(ctx, ggml_transpose(ctx, v_for_attention));
        if (v_for_attention->ne[0] != attention_probabilities->ne[0] ||
            attention_probabilities->ne[2] % v_for_attention->ne[2] != 0 ||
            attention_probabilities->ne[3] % v_for_attention->ne[3] != 0) {
            throw std::runtime_error("Qwen2 attention value shapes are incompatible");
        }
        auto* attention_values = ggml_mul_mat(ctx, v_for_attention, attention_probabilities);
        auto* attended = ggml_permute(ctx, attention_values, 0, 3, 1, 2);
        attended = ggml_cont_2d(ctx, attended, config.embedding_size, 1);

        auto* attention_output = ggml_mul_mat(ctx, weights.attention_output, attended);
        auto* attention_residual = ggml_add(ctx, residual_input, attention_output);
        auto* feed_forward_norm = ggml_mul(ctx,
            ggml_rms_norm(ctx, attention_residual, config.rms_norm_epsilon),
            weights.feed_forward_norm);
        auto* gate = ggml_silu(ctx, ggml_mul_mat(ctx, weights.feed_forward_gate, feed_forward_norm));
        auto* up = ggml_mul_mat(ctx, weights.feed_forward_up, feed_forward_norm);
        auto* down = ggml_mul_mat(ctx, weights.feed_forward_down, ggml_mul(ctx, gate, up));
        auto* output = ggml_add(ctx, attention_residual, down);

        return {output, key, value};
    }

    Impl(InferenceBackend backend, int32_t graph_threads)
        : graph_executor(backend, graph_threads) {}

    std::shared_ptr<const Qwen2GgufModel> model;
    GgmlContext weights_context {nullptr, ggml_free};
    GraphExecutor graph_executor;
    std::vector<LayerWeights> weights;
    std::vector<LayerCache> cache;
    int32_t first = 0;
    int32_t last = -1;
    int32_t graph_threads = 0;
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

size_t Qwen2GgufModel::layer_weight_bytes(int32_t first_layer, int32_t last_layer) const {
    if (first_layer < 0 || last_layer < first_layer || last_layer >= impl_->config.layer_count) {
        throw std::invalid_argument("invalid inclusive Qwen2 layer range");
    }
    static const char* const required_names[] = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
        "ffn_down.weight",
    };
    static const char* const optional_names[] = {
        "attn_q.bias", "attn_k.bias", "attn_v.bias",
    };
    size_t total = 0;
    for (int32_t layer = first_layer; layer <= last_layer; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const char* suffix : required_names) {
            const int64_t id = gguf_find_tensor(impl_->metadata, (prefix + suffix).c_str());
            if (id < 0) throw std::runtime_error("required GGUF layer tensor missing: " + prefix + suffix);
            total += gguf_get_tensor_size(impl_->metadata, id);
        }
        for (const char* suffix : optional_names) {
            const int64_t id = gguf_find_tensor(impl_->metadata, (prefix + suffix).c_str());
            if (id >= 0) total += gguf_get_tensor_size(impl_->metadata, id);
        }
    }
    return total;
}

const std::string& Qwen2GgufModel::path() const {
    return impl_->model_path;
}

std::vector<int32_t> Qwen2GgufModel::tokenize(const std::string& text) const {
    if (text.empty()) throw std::invalid_argument("prompt text must not be empty");
    const llama_vocab* vocab = impl_->tokenizer_vocab();
    std::vector<llama_token> tokens(std::max<size_t>(16, text.size() + 2));
    int32_t count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), true, false);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::overflow_error("tokenized prompt is too large");
    }
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(
            vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
            static_cast<int32_t>(tokens.size()), true, false);
    }
    if (count <= 0) throw std::runtime_error("GGUF tokenizer produced no tokens");
    tokens.resize(static_cast<size_t>(count));
    return {tokens.begin(), tokens.end()};
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

std::vector<float> Qwen2GgufModel::project_logits(
        const std::vector<float>& hidden_state,
        int32_t graph_threads) const {
    if (graph_threads < 0 || graph_threads > 8) {
        throw std::invalid_argument("Qwen2 graph thread count must be between 0 and 8");
    }
    const auto& config = impl_->config;
    if (hidden_state.size() != static_cast<size_t>(config.embedding_size)) {
        throw std::invalid_argument("final hidden state does not match Qwen2 embedding size");
    }
    for (float value : hidden_state) {
        if (!std::isfinite(value)) throw std::invalid_argument("final hidden state contains a non-finite value");
    }

    auto weight_context = create_context(2 * 1024 * 1024, true);
    auto* output_norm = impl_->map_tensor(weight_context.get(), "output_norm.weight",
        config.embedding_size, 1);
    const int64_t output_id = gguf_find_tensor(impl_->metadata, "output.weight");
    const std::string output_name = output_id >= 0 ? "output.weight" : "token_embd.weight";
    auto* output_weight = impl_->map_tensor(weight_context.get(), output_name,
        config.embedding_size, config.vocabulary_size);

    auto graph_context = create_context(8 * 1024 * 1024);
    auto* hidden = ggml_new_tensor_1d(graph_context.get(), GGML_TYPE_F32, config.embedding_size);
    std::memcpy(hidden->data, hidden_state.data(), hidden_state.size() * sizeof(float));
    auto* normalized = ggml_mul(graph_context.get(),
        ggml_rms_norm(graph_context.get(), hidden, config.rms_norm_epsilon), output_norm);
    auto* logits = ggml_mul_mat(graph_context.get(), output_weight, normalized);
    const int64_t output_bias_id = gguf_find_tensor(impl_->metadata, "output.bias");
    if (output_bias_id >= 0) {
        auto* output_bias = impl_->map_tensor(weight_context.get(), "output.bias", config.vocabulary_size, 1);
        logits = ggml_add(graph_context.get(), logits, output_bias);
    }
    run_graph(graph_context.get(), logits, graph_threads);
    const auto* data = static_cast<const float*>(logits->data);
    return {data, data + config.vocabulary_size};
}

Qwen2GgufShard::Qwen2GgufShard(
        std::shared_ptr<const Qwen2GgufModel> model,
        int32_t first_layer,
        int32_t last_layer,
        int32_t graph_threads,
        InferenceBackend backend,
        int64_t max_duration_ms) : impl_(std::make_unique<Impl>(backend, graph_threads)) {
    const auto deadline = execution_deadline(max_duration_ms);
    require_before_deadline(deadline);
    if (!model) throw std::invalid_argument("GGUF model is required");
    const auto& config = model->config();
    if (first_layer < 0 || last_layer < first_layer || last_layer >= config.layer_count) {
        throw std::invalid_argument("invalid inclusive Qwen2 layer range");
    }

    impl_->model = std::move(model);
    impl_->first = first_layer;
    impl_->last = last_layer;
    if (graph_threads < 0 || graph_threads > 8) {
        throw std::invalid_argument("Qwen2 graph thread count must be between 0 and 8");
    }
    impl_->weights_context = create_context(4 * 1024 * 1024, true);
    impl_->weights.reserve(static_cast<size_t>(last_layer - first_layer + 1));
    impl_->cache.resize(static_cast<size_t>(last_layer - first_layer + 1));
    const auto& model_impl = *impl_->model->impl_;

    const int32_t key_value_width =
        config.embedding_size / config.attention_heads * config.kv_heads;
    const bool backend_owned = backend == InferenceBackend::Vulkan;
    for (int32_t layer = first_layer; layer <= last_layer; ++layer) {
        require_before_deadline(deadline);
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        Impl::LayerWeights layer_weights {
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_norm.weight",
                config.embedding_size, 1, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_q.weight",
                config.embedding_size, config.embedding_size, backend_owned),
            model_impl.map_optional_vector(impl_->weights_context.get(), prefix + "attn_q.bias",
                config.embedding_size, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_k.weight",
                config.embedding_size, key_value_width, backend_owned),
            model_impl.map_optional_vector(impl_->weights_context.get(), prefix + "attn_k.bias",
                key_value_width, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_v.weight",
                config.embedding_size, key_value_width, backend_owned),
            model_impl.map_optional_vector(impl_->weights_context.get(), prefix + "attn_v.bias",
                key_value_width, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "attn_output.weight",
                config.embedding_size, config.embedding_size, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_norm.weight",
                config.embedding_size, 1, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_gate.weight",
                config.embedding_size, config.feed_forward_size, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_up.weight",
                config.embedding_size, config.feed_forward_size, backend_owned),
            model_impl.map_tensor(impl_->weights_context.get(), prefix + "ffn_down.weight",
                config.feed_forward_size, config.embedding_size, backend_owned),
        };
        const ggml_tensor* tensors[] = {
            layer_weights.attention_norm, layer_weights.query, layer_weights.key, layer_weights.value,
            layer_weights.attention_output, layer_weights.feed_forward_norm,
            layer_weights.feed_forward_gate, layer_weights.feed_forward_up, layer_weights.feed_forward_down,
            layer_weights.query_bias, layer_weights.key_bias, layer_weights.value_bias,
        };
        for (const auto* tensor : tensors) {
            if (tensor != nullptr) impl_->weight_bytes += ggml_nbytes(tensor);
        }
        impl_->weights.push_back(layer_weights);
    }
    impl_->graph_executor.allocate_weights(
        impl_->weights_context.get(),
        [&model_impl](const ggml_tensor* tensor) { return model_impl.mapped_tensor_data(tensor); },
        deadline);
}

Qwen2GgufShard::~Qwen2GgufShard() = default;
Qwen2GgufShard::Qwen2GgufShard(Qwen2GgufShard&&) noexcept = default;
Qwen2GgufShard& Qwen2GgufShard::operator=(Qwen2GgufShard&&) noexcept = default;

std::vector<float> Qwen2GgufShard::execute(
        const std::vector<float>& token_major_input,
        int32_t token_offset,
        int64_t max_duration_ms) {
    const auto deadline = execution_deadline(max_duration_ms);
    require_before_deadline(deadline);
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
        require_before_deadline(deadline);
        const auto begin = token_major_input.begin() + token * config.embedding_size;
        const size_t layer_count = impl_->weights.size();
        const size_t graph_memory_bytes = (layer_count + 1) * 32ULL * 1024ULL * 1024ULL;
        auto graph_context = create_context(graph_memory_bytes);
        auto* activation = ggml_new_tensor_1d(
            graph_context.get(), GGML_TYPE_F32, config.embedding_size);
        std::memcpy(activation->data, &*begin,
            static_cast<size_t>(config.embedding_size) * sizeof(float));
        std::vector<std::vector<float>> pending_keys(impl_->cache.size());
        std::vector<std::vector<float>> pending_values(impl_->cache.size());
        std::vector<ggml_tensor*> pending_key_tensors;
        std::vector<ggml_tensor*> pending_value_tensors;
        pending_key_tensors.reserve(layer_count);
        pending_value_tensors.reserve(layer_count);

        for (size_t index = 0; index < impl_->weights.size(); ++index) {
            require_before_deadline(deadline);
            const auto layer_result = impl_->build_layer(
                graph_context.get(), impl_->weights[index], impl_->cache[index],
                activation, impl_->token_count, config);
            activation = layer_result.activation;
            pending_key_tensors.push_back(layer_result.rotated_key);
            pending_value_tensors.push_back(layer_result.value);
        }

        impl_->graph_executor.compute(graph_context.get(), activation, deadline);
        std::vector<float> output_activation;
        try {
            output_activation = impl_->graph_executor.read_f32(activation, deadline);
            for (size_t index = 0; index < layer_count; ++index) {
                pending_keys[index] = impl_->graph_executor.read_f32(
                    pending_key_tensors[index], deadline);
                pending_values[index] = impl_->graph_executor.read_f32(
                    pending_value_tensors[index], deadline);
            }
            impl_->graph_executor.finish();
            require_before_deadline(deadline);
        } catch (...) {
            impl_->graph_executor.finish();
            throw;
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

        result.insert(result.end(), output_activation.begin(), output_activation.end());
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

bool Qwen2GgufShard::used_vulkan() const {
    return impl_->graph_executor.used_vulkan();
}

BackendExecutionStats Qwen2GgufShard::backend_stats() const {
    auto stats = impl_->graph_executor.stats();
    for (const auto& layer : impl_->cache) {
        stats.host_kv_bytes += static_cast<uint64_t>(
            (layer.keys.size() + layer.values.size()) * sizeof(float));
    }
    // KV tensors remain host-resident until the multi-layer persistent graph migration is complete.
    stats.gpu_kv_bytes = 0;
    return stats;
}

int32_t Qwen2GgufShard::first_layer() const {
    return impl_->first;
}

int32_t Qwen2GgufShard::last_layer() const {
    return impl_->last;
}

}
