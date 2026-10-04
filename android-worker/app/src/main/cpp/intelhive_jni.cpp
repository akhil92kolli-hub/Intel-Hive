#include <jni.h>
#include <vulkan/vulkan.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include "llama.h"
#include "qwen2_gguf.h"
#ifdef INTELHIVE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint8_t kActivationResult = 1;
constexpr uint8_t kTokenResult = 2;
constexpr int64_t kMaximumNativeDeadlineMs = 24LL * 60LL * 60LL * 1000LL;

struct ShardHandle {
    std::shared_ptr<const intelhive::Qwen2GgufModel> model;
    int32_t first_layer;
    int32_t last_layer;
    int32_t graph_threads;
    bool use_vulkan = false;
    bool used_vulkan = false;
    intelhive::BackendExecutionStats backend_stats {};
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<intelhive::Qwen2GgufShard>> sequences;
};

std::mutex handles_mutex;
jlong next_handle = 1;
std::unordered_map<jlong, std::shared_ptr<const intelhive::Qwen2GgufModel>> models;
std::unordered_map<jlong, std::shared_ptr<ShardHandle>> shards;

int64_t remaining_duration_ms(
        const std::chrono::steady_clock::time_point& started_at,
        int64_t max_duration_ms) {
    if (max_duration_ms == 0) return 0;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at).count();
    const int64_t remaining = max_duration_ms - elapsed;
    if (remaining <= 0) throw std::runtime_error("Native execution deadline exceeded");
    return remaining;
}

struct NativeCallDeadline {
    std::chrono::steady_clock::time_point started_at;
    int64_t max_duration_ms;
};

bool native_call_deadline_expired(void* data) {
    const auto& deadline = *static_cast<NativeCallDeadline*>(data);
    if (deadline.max_duration_ms == 0) return false;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - deadline.started_at).count() >=
        deadline.max_duration_ms;
}

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    if (env->ExceptionCheck()) return;
    jclass exception_class = env->FindClass(class_name);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
        env->DeleteLocalRef(exception_class);
    }
}

void throw_state(JNIEnv* env, const std::exception& error) {
    throw_java(env, "java/lang/IllegalStateException", error.what());
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    constexpr char digits[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += digits[character >> 4];
                    escaped += digits[character & 0x0f];
                } else {
                    escaped += static_cast<char>(character);
                }
        }
    }
    return escaped;
}

std::string vulkan_version(uint32_t version) {
    std::ostringstream value;
    value << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version)
          << "." << VK_VERSION_PATCH(version);
    return value.str();
}

const char* vulkan_vendor_name(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x1002: return "AMD";
        case 0x1010: return "Imagination Technologies";
        case 0x10de: return "NVIDIA";
        case 0x13b5: return "ARM";
        case 0x5143: return "Qualcomm";
        case 0x8086: return "Intel";
        case 0x1ae0: return "Google";
        default: return "Unknown";
    }
}

std::string vulkan_device_info() {
    uint32_t loader_version = VK_API_VERSION_1_0;
    const VkResult version_result = vkEnumerateInstanceVersion(&loader_version);
    if (version_result != VK_SUCCESS) {
        return "{\"available\":false,\"vulkan_available\":false,"
            "\"reason\":\"Vulkan loader version query failed\"}";
    }

    VkApplicationInfo application_info {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "IntelHive";
    application_info.applicationVersion = 1;
    application_info.pEngineName = "IntelHive";
    application_info.engineVersion = 1;
    application_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        return "{\"available\":false,\"vulkan_available\":false,"
            "\"reason\":\"Vulkan instance creation failed\"}";
    }

    std::string result;
    try {
        uint32_t device_count = 0;
        VkResult query = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (query != VK_SUCCESS || device_count == 0) {
            result = "{\"available\":false,\"vulkan_available\":false,"
                "\"reason\":\"No Vulkan physical device was enumerated\"}";
        } else {
            std::vector<VkPhysicalDevice> devices(device_count);
            query = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
            if (query != VK_SUCCESS || devices.empty()) {
                result = "{\"available\":false,\"vulkan_available\":false,"
                    "\"reason\":\"Vulkan physical-device query failed\"}";
            } else {
                VkPhysicalDeviceProperties properties {};
                vkGetPhysicalDeviceProperties(devices.front(), &properties);
                std::ostringstream json;
                json << "{\"available\":true"
                     << ",\"api_version\":\"" << vulkan_version(properties.apiVersion) << "\""
                     << ",\"loader_api_version\":\"" << vulkan_version(loader_version) << "\""
                     << ",\"gpu_name\":\"" << json_escape(properties.deviceName) << "\""
                     << ",\"vendor_id\":" << properties.vendorID
                     << ",\"vendor_name\":\"" << vulkan_vendor_name(properties.vendorID) << "\""
                     << ",\"device_id\":" << properties.deviceID
                     << ",\"driver_version\":" << properties.driverVersion
                     << ",\"device_type\":" << static_cast<int>(properties.deviceType);
#ifdef INTELHIVE_HAS_VULKAN
                const int backend_device_count = ggml_backend_vk_get_device_count();
                size_t free_memory = 0;
                size_t total_memory = 0;
                if (backend_device_count > 0) {
                    ggml_backend_vk_get_device_memory(0, &free_memory, &total_memory);
                }
                json << ",\"ggml_backend_compiled\":true"
                     << ",\"ggml_backend_device_count\":" << backend_device_count
                     << ",\"vulkan_available\":" << (backend_device_count > 0 ? "true" : "false")
                     << ",\"vulkan_free_memory_bytes\":" << free_memory
                     << ",\"vulkan_total_memory_bytes\":" << total_memory
                     << ",\"inference_backend_enabled\":"
                     << (backend_device_count > 0 ? "true" : "false")
                     << ",\"inference_status\":\""
                     << (backend_device_count > 0 ? "NOT_RUN" : "SKIPPED") << "\"";
                if (backend_device_count <= 0) {
                    json << ",\"inference_skip_reason\":\"GGML Vulkan found no usable Vulkan device.\"";
                }
#else
                json << ",\"ggml_backend_compiled\":false"
                     << ",\"vulkan_available\":false"
                     << ",\"inference_backend_enabled\":false"
                     << ",\"inference_status\":\"SKIPPED\""
                     << ",\"inference_skip_reason\":\"GGML Vulkan backend was not compiled into this APK.\"";
#endif
                json << "}";
                result = json.str();
            }
        }
    } catch (...) {
        vkDestroyInstance(instance, nullptr);
        throw;
    }
    vkDestroyInstance(instance, nullptr);
    return result;
}

std::string read_utf(JNIEnv* env, jstring value, const char* field_name) {
    if (value == nullptr) throw std::invalid_argument(std::string(field_name) + " is required");
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) throw std::runtime_error("could not read Java string");
    std::string result;
    try {
        result.assign(chars);
    } catch (...) {
        env->ReleaseStringUTFChars(value, chars);
        throw;
    }
    env->ReleaseStringUTFChars(value, chars);
    if (result.empty()) throw std::invalid_argument(std::string(field_name) + " must not be empty");
    return result;
}

std::shared_ptr<const intelhive::Qwen2GgufModel> get_model(jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    const auto found = models.find(handle);
    if (found == models.end()) throw std::invalid_argument("native model handle is invalid or closed");
    return found->second;
}

std::shared_ptr<ShardHandle> get_shard(jlong handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    const auto found = shards.find(handle);
    if (found == shards.end()) throw std::invalid_argument("native shard handle is invalid or closed");
    return found->second;
}

jlong allocate_handle() {
    std::lock_guard<std::mutex> lock(handles_mutex);
    if (next_handle <= 0 || next_handle == std::numeric_limits<jlong>::max()) {
        throw std::overflow_error("native handle space exhausted");
    }
    return next_handle++;
}

jbyteArray make_byte_array(JNIEnv* env, const std::vector<uint8_t>& bytes) {
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        throw std::overflow_error("native result exceeds the JNI byte-array limit");
    }
    auto result = env->NewByteArray(static_cast<jsize>(bytes.size()));
    if (result == nullptr) throw std::runtime_error("could not allocate JNI result array");
    if (!bytes.empty()) env->SetByteArrayRegion(result, 0, static_cast<jsize>(bytes.size()),
        reinterpret_cast<const jbyte*>(bytes.data()));
    if (env->ExceptionCheck()) throw std::runtime_error("could not marshal JNI result array");
    return result;
}

void append_u64_le(std::vector<uint8_t>& output, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
    }
}

void append_f32_le(std::vector<uint8_t>& output, float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "Qwen2 executor requires IEEE-754 float32");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
}

float read_f32_le(const uint8_t* data) {
    const uint32_t bits = static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8U) |
        (static_cast<uint32_t>(data[2]) << 16U) |
        (static_cast<uint32_t>(data[3]) << 24U);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeIsAvailable(JNIEnv*, jobject) {
    // JNI_OnLoad has already completed. Keep this startup probe
    // side-effect-free: llama_backend_init/system-info probing
    // can execute device-specific CPU detection and must not run while the UI
    // is merely checking whether the packaged bridge loaded successfully.
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeVulkanInfo(JNIEnv* env, jobject) {
    try {
        const std::string result = vulkan_device_info();
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& error) {
        throw_state(env, error);
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeRunLlamaCppBaseline(
        JNIEnv* env,
        jobject,
        jstring model_path,
        jint cpu_threads,
        jint prefill_tokens,
        jint generated_tokens,
        jlong max_duration_ms) {
    try {
        if (cpu_threads < 1 || cpu_threads > 8 ||
            prefill_tokens < 1 || prefill_tokens > 256 ||
            generated_tokens < 1 || generated_tokens > 128 ||
            max_duration_ms < 1) {
            throw std::invalid_argument("Invalid llama.cpp baseline benchmark configuration");
        }
        NativeCallDeadline deadline {std::chrono::steady_clock::now(), max_duration_ms};

        const std::string path = read_utf(env, model_path, "model path");
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), model_params),
            llama_model_free);
        if (!model) throw std::runtime_error("llama.cpp baseline model load failed");

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(prefill_tokens + generated_tokens + 1);
        context_params.n_batch = static_cast<uint32_t>(prefill_tokens);
        context_params.n_ubatch = static_cast<uint32_t>(prefill_tokens);
        context_params.n_threads = cpu_threads;
        context_params.n_threads_batch = cpu_threads;
        context_params.no_perf = true;
        context_params.abort_callback = native_call_deadline_expired;
        context_params.abort_callback_data = &deadline;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("llama.cpp baseline context creation failed");

        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        if (vocab == nullptr || llama_vocab_n_tokens(vocab) <= 1) {
            throw std::runtime_error("llama.cpp baseline vocabulary is unavailable");
        }
        std::vector<llama_token> prompt(static_cast<size_t>(prefill_tokens), 1);
        auto decode = [&](const llama_token* tokens, int32_t count) {
            const int32_t status = llama_decode(context.get(), llama_batch_get_one(
                const_cast<llama_token*>(tokens), count));
            if (status != 0) {
                if (status == 2 && native_call_deadline_expired(&deadline)) {
                    throw std::runtime_error("Native baseline execution deadline exceeded");
                }
                throw std::runtime_error("llama.cpp baseline decode failed with status " +
                    std::to_string(status));
            }
        };
        auto sample_greedy = [&]() -> llama_token {
            const float* logits = llama_get_logits_ith(context.get(), -1);
            if (logits == nullptr) throw std::runtime_error("llama.cpp baseline logits are unavailable");
            const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
            return static_cast<llama_token>(std::distance(
                logits, std::max_element(logits, logits + vocabulary_size)));
        };

        // Warm up the exact native path, then clear KV state before measuring.
        remaining_duration_ms(deadline.started_at, max_duration_ms);
        decode(prompt.data(), static_cast<int32_t>(prompt.size()));
        llama_token warmup_token = sample_greedy();
        decode(&warmup_token, 1);
        llama_memory_clear(llama_get_memory(context.get()), false);
        remaining_duration_ms(deadline.started_at, max_duration_ms);

        const auto benchmark_start = std::chrono::steady_clock::now();
        const auto prefill_start = benchmark_start;
        decode(prompt.data(), static_cast<int32_t>(prompt.size()));
        const auto prefill_end = std::chrono::steady_clock::now();
        llama_token input_token = sample_greedy();
        std::vector<double> decode_latencies_ms;
        decode_latencies_ms.reserve(static_cast<size_t>(generated_tokens));
        for (int32_t index = 0; index < generated_tokens; ++index) {
            const auto decode_start = std::chrono::steady_clock::now();
            decode(&input_token, 1);
            const auto decode_end = std::chrono::steady_clock::now();
            decode_latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(decode_end - decode_start).count());
            input_token = sample_greedy();
            if (native_call_deadline_expired(&deadline)) {
                break;
            }
        }
        const auto benchmark_end = std::chrono::steady_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(benchmark_end - prefill_end).count();
        std::ostringstream result;
        result << "{\"generated_tokens\":" << decode_latencies_ms.size()
               << ",\"requested_tokens\":" << generated_tokens
               << ",\"cpu_threads\":" << cpu_threads
               << ",\"gpu_layers\":0"
               << ",\"prefill_time_ms\":" << prefill_ms
               << ",\"decode_time_ms\":" << decode_ms
               << ",\"total_time_ms\":"
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                    benchmark_end - benchmark_start).count()
               << ",\"decode_latency_ms\":[";
        for (size_t index = 0; index < decode_latencies_ms.size(); ++index) {
            if (index != 0) result << ',';
            result << decode_latencies_ms[index];
        }
        result << "]}";
        return env->NewStringUTF(result.str().c_str());
    } catch (const std::exception& error) {
        throw_state(env, error);
        return nullptr;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeLoadModel(
        JNIEnv* env, jobject, jstring model_path) {
    try {
        auto model = std::make_shared<intelhive::Qwen2GgufModel>(read_utf(env, model_path, "model path"));
        const jlong handle = allocate_handle();
        std::lock_guard<std::mutex> lock(handles_mutex);
        models.emplace(handle, std::move(model));
        return handle;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeLayerCount(
        JNIEnv* env, jobject, jlong model_handle) {
    try {
        return get_model(model_handle)->config().layer_count;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeEmbeddingSize(
        JNIEnv* env, jobject, jlong model_handle) {
    try {
        return get_model(model_handle)->config().embedding_size;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeTokenize(
        JNIEnv* env, jobject, jlong model_handle, jbyteArray utf8_prompt) {
    try {
        if (utf8_prompt == nullptr) throw std::invalid_argument("UTF-8 prompt is required");
        const jsize byte_count = env->GetArrayLength(utf8_prompt);
        if (byte_count <= 0) throw std::invalid_argument("prompt must not be empty");
        std::string prompt(static_cast<size_t>(byte_count), '\0');
        env->GetByteArrayRegion(
            utf8_prompt, 0, byte_count, reinterpret_cast<jbyte*>(prompt.data()));
        if (env->ExceptionCheck()) throw std::runtime_error("could not read UTF-8 prompt");
        const auto tokens = get_model(model_handle)->tokenize(prompt);
        if (tokens.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
            throw std::overflow_error("tokenized prompt exceeds the JNI array limit");
        }
        auto result = env->NewLongArray(static_cast<jsize>(tokens.size()));
        if (result == nullptr) throw std::runtime_error("could not allocate token ID array");
        std::vector<jlong> values(tokens.begin(), tokens.end());
        env->SetLongArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
        if (env->ExceptionCheck()) throw std::runtime_error("could not marshal token IDs");
        return result;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return nullptr;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeCreateShard(
        JNIEnv* env, jobject, jlong model_handle, jint first_layer, jint last_layer,
        jint graph_threads, jboolean use_vulkan) {
    try {
        auto model = get_model(model_handle);
        if (first_layer < 0 || last_layer < first_layer || last_layer >= model->config().layer_count) {
            throw std::invalid_argument("requested Qwen2 layer range is invalid");
        }
        if (graph_threads < 0 || graph_threads > 8) {
            throw std::invalid_argument("Qwen2 graph thread count must be between 0 and 8");
        }
        auto shard = std::make_shared<ShardHandle>();
        shard->model = std::move(model);
        shard->first_layer = first_layer;
        shard->last_layer = last_layer;
        shard->graph_threads = graph_threads;
        shard->use_vulkan = use_vulkan == JNI_TRUE;
        const jlong handle = allocate_handle();
        std::lock_guard<std::mutex> lock(handles_mutex);
        shards.emplace(handle, std::move(shard));
        return handle;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeLayerWeightBytes(
        JNIEnv* env, jobject, jlong model_handle, jint first_layer, jint last_layer) {
    try {
        const size_t bytes = get_model(model_handle)->layer_weight_bytes(first_layer, last_layer);
        if (bytes > static_cast<size_t>(std::numeric_limits<jlong>::max())) {
            throw std::overflow_error("layer weight size exceeds JNI range");
        }
        return static_cast<jlong>(bytes);
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeShardUsedVulkan(
        JNIEnv* env, jobject, jlong shard_handle) {
    try {
        const auto shard = get_shard(shard_handle);
        std::lock_guard<std::mutex> lock(shard->mutex);
        if (shard->used_vulkan) return JNI_TRUE;
        for (const auto& entry : shard->sequences) {
            if (entry.second->used_vulkan()) return JNI_TRUE;
        }
        return JNI_FALSE;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeShardBackendStats(
        JNIEnv* env, jobject, jlong shard_handle) {
    try {
        const auto shard = get_shard(shard_handle);
        std::lock_guard<std::mutex> lock(shard->mutex);
        auto stats = shard->backend_stats;
        for (const auto& entry : shard->sequences) {
            const auto current = entry.second->backend_stats();
            stats.cpu_graph_nodes = std::max(stats.cpu_graph_nodes, current.cpu_graph_nodes);
            stats.vulkan_graph_nodes = std::max(stats.vulkan_graph_nodes, current.vulkan_graph_nodes);
            stats.gpu_weight_bytes = std::max(stats.gpu_weight_bytes, current.gpu_weight_bytes);
            stats.gpu_kv_bytes = std::max(stats.gpu_kv_bytes, current.gpu_kv_bytes);
            stats.host_kv_bytes = std::max(stats.host_kv_bytes, current.host_kv_bytes);
            stats.activation_transfer_bytes = std::max(
                stats.activation_transfer_bytes, current.activation_transfer_bytes);
            stats.fallback_used = stats.fallback_used || current.fallback_used;
        }
        const bool any_vulkan = stats.vulkan_graph_nodes > 0;
        const bool any_cpu = stats.cpu_graph_nodes > 0;
        const char* actual = any_vulkan && any_cpu ? "hybrid" : any_vulkan ? "vulkan" : "cpu";
        const bool full_gpu = shard->use_vulkan && any_vulkan && !any_cpu &&
            stats.gpu_kv_bytes > 0 && stats.host_kv_bytes == 0;
        std::ostringstream json;
        json << "{\"requested_backend\":\"" << (shard->use_vulkan ? "vulkan" : "cpu")
             << "\",\"actual_backend\":\"" << actual
             << "\",\"cpu_graph_nodes\":" << stats.cpu_graph_nodes
             << ",\"vulkan_graph_nodes\":" << stats.vulkan_graph_nodes
             << ",\"gpu_weight_bytes\":" << stats.gpu_weight_bytes
             << ",\"gpu_kv_bytes\":" << stats.gpu_kv_bytes
             << ",\"host_kv_bytes\":" << stats.host_kv_bytes
             << ",\"activation_transfer_bytes\":" << stats.activation_transfer_bytes
             << ",\"fallback_used\":" << (stats.fallback_used ? "true" : "false")
             << ",\"full_gpu_verified\":" << (full_gpu ? "true" : "false") << "}";
        return env->NewStringUTF(json.str().c_str());
    } catch (const std::exception& error) {
        throw_state(env, error);
        return nullptr;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeExecute(
        JNIEnv* env, jobject, jlong shard_handle, jstring sequence_id,
        jint token_offset, jint token_count, jlongArray token_ids, jbyteArray activation,
        jlong max_duration_ms) {
    try {
        if (max_duration_ms < 0 || max_duration_ms > kMaximumNativeDeadlineMs) {
            throw std::invalid_argument("native execution duration limit is invalid");
        }
        const auto native_started_at = std::chrono::steady_clock::now();
        const std::string sequence = read_utf(env, sequence_id, "sequence ID");
        if (token_offset < 0 || token_count <= 0) {
            throw std::invalid_argument("token offset and count are invalid");
        }
        const bool has_tokens = token_ids != nullptr;
        const bool has_activation = activation != nullptr;
        if (has_tokens == has_activation) {
            throw std::invalid_argument("exactly one of token IDs or activation bytes is required");
        }

        auto shard = get_shard(shard_handle);
        if (has_tokens && shard->first_layer != 0) {
            throw std::invalid_argument("token IDs can only be embedded by the first model shard");
        }
        if (has_activation && shard->first_layer == 0) {
            throw std::invalid_argument("the first model shard must receive token IDs");
        }
        const auto& config = shard->model->config();
        if (token_offset > config.context_length ||
            token_count > config.context_length - token_offset) {
            throw std::invalid_argument("requested tokens exceed the Qwen2 context capacity");
        }
        std::lock_guard<std::mutex> lock(shard->mutex);
        auto found = shard->sequences.find(sequence);
        if (found == shard->sequences.end()) {
            if (token_offset != 0) {
                throw std::invalid_argument("a shard sequence must start at KV token offset zero");
            }
            found = shard->sequences.emplace(sequence,
                std::make_unique<intelhive::Qwen2GgufShard>(
                    shard->model, shard->first_layer, shard->last_layer,
                    shard->graph_threads,
                    shard->use_vulkan ? intelhive::InferenceBackend::Vulkan
                                      : intelhive::InferenceBackend::Cpu,
                    remaining_duration_ms(native_started_at, max_duration_ms))).first;
        }
        auto& executor = *found->second;
        if (executor.cached_tokens() != token_offset) {
            shard->sequences.erase(found);
            throw std::invalid_argument("KV token offset does not match the native shard cache");
        }

        const int32_t embedding_size = shard->model->config().embedding_size;
        std::vector<float> input;
        if (has_tokens) {
            const jsize count = env->GetArrayLength(token_ids);
            if (count != token_count) {
                throw std::invalid_argument("token count does not match token ID input");
            }
            std::vector<jlong> ids(static_cast<size_t>(count));
            env->GetLongArrayRegion(token_ids, 0, count, ids.data());
            if (env->ExceptionCheck()) throw std::runtime_error("could not read token IDs");
            input.reserve(static_cast<size_t>(count) * static_cast<size_t>(embedding_size));
            for (jlong id : ids) {
                if (id < 0 || static_cast<uint64_t>(id) >
                    std::numeric_limits<uint32_t>::max()) {
                    throw std::invalid_argument("token ID is outside the supported range");
                }
                auto embedding = shard->model->embed_token(static_cast<uint32_t>(id));
                input.insert(input.end(), embedding.begin(), embedding.end());
            }
        } else {
            const size_t expected_size = static_cast<size_t>(token_count) *
                static_cast<size_t>(embedding_size) * sizeof(float);
            const jsize byte_count = env->GetArrayLength(activation);
            if (expected_size > static_cast<size_t>(std::numeric_limits<jsize>::max()) ||
                byte_count != static_cast<jsize>(expected_size)) {
                throw std::invalid_argument("activation byte length does not match F32 token-major shape");
            }
            std::vector<jbyte> bytes(static_cast<size_t>(byte_count));
            env->GetByteArrayRegion(activation, 0, byte_count, bytes.data());
            if (env->ExceptionCheck()) throw std::runtime_error("could not read activation bytes");
            input.resize(expected_size / sizeof(float));
            const auto* raw = reinterpret_cast<const uint8_t*>(bytes.data());
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = read_f32_le(raw + i * sizeof(float));
            }
        }

        std::vector<float> output;
        try {
            output = executor.execute(
                input,
                token_offset,
                remaining_duration_ms(native_started_at, max_duration_ms));
            shard->backend_stats = executor.backend_stats();
        } catch (...) {
            shard->sequences.erase(sequence);
            throw;
        }

        try {
            std::vector<uint8_t> result;
            if (shard->last_layer == shard->model->config().layer_count - 1) {
                const size_t last_hidden_offset =
                    (static_cast<size_t>(token_count) - 1) * static_cast<size_t>(embedding_size);
                std::vector<float> final_hidden(
                    output.begin() + last_hidden_offset,
                    output.begin() + last_hidden_offset + static_cast<size_t>(embedding_size));
                remaining_duration_ms(native_started_at, max_duration_ms);
                const auto logits = shard->model->project_logits(final_hidden, shard->graph_threads);
                remaining_duration_ms(native_started_at, max_duration_ms);
                const auto best = std::max_element(logits.begin(), logits.end());
                if (best == logits.end() || !std::isfinite(*best)) {
                    throw std::runtime_error("Qwen2 output logits contain no finite token");
                }
                result.reserve(9);
                result.push_back(kTokenResult);
                append_u64_le(result, static_cast<uint64_t>(std::distance(logits.begin(), best)));
            } else {
                result.reserve(1 + output.size() * sizeof(float));
                result.push_back(kActivationResult);
                for (float value : output) append_f32_le(result, value);
            }
            return make_byte_array(env, result);
        } catch (...) {
            shard->sequences.erase(sequence);
            throw;
        }
    } catch (const std::exception& error) {
        throw_state(env, error);
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeEndSequence(
        JNIEnv* env, jobject, jlong shard_handle, jstring sequence_id) {
    try {
        const std::string sequence = read_utf(env, sequence_id, "sequence ID");
        auto shard = get_shard(shard_handle);
        std::lock_guard<std::mutex> lock(shard->mutex);
        const auto found = shard->sequences.find(sequence);
        if (found != shard->sequences.end()) {
            shard->used_vulkan = shard->used_vulkan || found->second->used_vulkan();
            shard->backend_stats = found->second->backend_stats();
        }
        shard->sequences.erase(sequence);
    } catch (const std::exception& error) {
        throw_state(env, error);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeDestroyShard(
        JNIEnv*, jobject, jlong shard_handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    shards.erase(shard_handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeUnloadModel(
        JNIEnv*, jobject, jlong model_handle) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    models.erase(model_handle);
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || env == nullptr) {
        return JNI_ERR;
    }

    jclass bridge_class = env->FindClass("com/intellihive/worker/inference/NativeBridge");
    if (bridge_class == nullptr) {
        return JNI_ERR;
    }

    const JNINativeMethod methods[] = {
        {const_cast<char*>("nativeIsAvailable"), const_cast<char*>("()Z"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeIsAvailable)},
        {const_cast<char*>("nativeVulkanInfo"), const_cast<char*>("()Ljava/lang/String;"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeVulkanInfo)},
        {const_cast<char*>("nativeRunLlamaCppBaseline"),
            const_cast<char*>("(Ljava/lang/String;IIIJ)Ljava/lang/String;"),
            reinterpret_cast<void*>(
                Java_com_intelhive_worker_inference_NativeBridge_nativeRunLlamaCppBaseline)},
        {const_cast<char*>("nativeLoadModel"), const_cast<char*>("(Ljava/lang/String;)J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeLoadModel)},
        {const_cast<char*>("nativeLayerCount"), const_cast<char*>("(J)I"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeLayerCount)},
        {const_cast<char*>("nativeEmbeddingSize"), const_cast<char*>("(J)I"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeEmbeddingSize)},
        {const_cast<char*>("nativeTokenize"), const_cast<char*>("(J[B)[J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeTokenize)},
        {const_cast<char*>("nativeCreateShard"), const_cast<char*>("(JIIIZ)J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeCreateShard)},
        {const_cast<char*>("nativeLayerWeightBytes"), const_cast<char*>("(JII)J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeLayerWeightBytes)},
        {const_cast<char*>("nativeShardUsedVulkan"), const_cast<char*>("(J)Z"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeShardUsedVulkan)},
        {const_cast<char*>("nativeShardBackendStats"), const_cast<char*>("(J)Ljava/lang/String;"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeShardBackendStats)},
        {const_cast<char*>("nativeExecute"), const_cast<char*>("(JLjava/lang/String;II[J[BJ)[B"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeExecute)},
        {const_cast<char*>("nativeEndSequence"), const_cast<char*>("(JLjava/lang/String;)V"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeEndSequence)},
        {const_cast<char*>("nativeDestroyShard"), const_cast<char*>("(J)V"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeDestroyShard)},
        {const_cast<char*>("nativeUnloadModel"), const_cast<char*>("(J)V"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeUnloadModel)},
    };
    if (env->RegisterNatives(
            bridge_class,
            methods,
            static_cast<jint>(sizeof(methods) / sizeof(methods[0]))) != JNI_OK) {
        env->DeleteLocalRef(bridge_class);
        return JNI_ERR;
    }
    env->DeleteLocalRef(bridge_class);
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO,
        "IntelHiveJNI",
        "IntelHive Java JNI bridge loaded and native methods registered");
#endif
    return JNI_VERSION_1_6;
}
