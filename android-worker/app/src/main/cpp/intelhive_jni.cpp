#include <jni.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include "llama.h"
#include "qwen2_gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint8_t kActivationResult = 1;
constexpr uint8_t kTokenResult = 2;

struct ShardHandle {
    std::shared_ptr<const intelhive::Qwen2GgufModel> model;
    int32_t first_layer;
    int32_t last_layer;
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<intelhive::Qwen2GgufShard>> sequences;
};

std::mutex handles_mutex;
jlong next_handle = 1;
std::unordered_map<jlong, std::shared_ptr<const intelhive::Qwen2GgufModel>> models;
std::unordered_map<jlong, std::shared_ptr<ShardHandle>> shards;

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

extern "C" JNIEXPORT jlong JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeCreateShard(
        JNIEnv* env, jobject, jlong model_handle, jint first_layer, jint last_layer) {
    try {
        auto model = get_model(model_handle);
        if (first_layer < 0 || last_layer < first_layer || last_layer >= model->config().layer_count) {
            throw std::invalid_argument("requested Qwen2 layer range is invalid");
        }
        auto shard = std::make_shared<ShardHandle>();
        shard->model = std::move(model);
        shard->first_layer = first_layer;
        shard->last_layer = last_layer;
        const jlong handle = allocate_handle();
        std::lock_guard<std::mutex> lock(handles_mutex);
        shards.emplace(handle, std::move(shard));
        return handle;
    } catch (const std::exception& error) {
        throw_state(env, error);
        return 0;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_intelhive_worker_inference_NativeBridge_nativeExecute(
        JNIEnv* env, jobject, jlong shard_handle, jstring sequence_id,
        jint token_offset, jint token_count, jlongArray token_ids, jbyteArray activation) {
    try {
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
                    shard->model, shard->first_layer, shard->last_layer)).first;
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
            output = executor.execute(input, token_offset);
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
                const auto logits = shard->model->project_logits(final_hidden);
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
        {const_cast<char*>("nativeLoadModel"), const_cast<char*>("(Ljava/lang/String;)J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeLoadModel)},
        {const_cast<char*>("nativeLayerCount"), const_cast<char*>("(J)I"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeLayerCount)},
        {const_cast<char*>("nativeEmbeddingSize"), const_cast<char*>("(J)I"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeEmbeddingSize)},
        {const_cast<char*>("nativeCreateShard"), const_cast<char*>("(JII)J"),
            reinterpret_cast<void*>(Java_com_intelhive_worker_inference_NativeBridge_nativeCreateShard)},
        {const_cast<char*>("nativeExecute"), const_cast<char*>("(JLjava/lang/String;II[J[B)[B"),
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
