package com.intellihive.worker.model

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.net.URI
import java.util.Locale

data class ModelManifest(
    val modelId: String,
    val version: String,
    val architecture: String,
    val layerCount: Int,
    val hiddenSize: Int,
    val fileName: String,
    val downloadUrl: String,
    val sizeBytes: Long,
    val sha256: String
) {
    val displayName: String
        get() = "Qwen2.5 3B Instruct"

    val sizeLabel: String
        get() = String.format(Locale.US, "%.1f GB", sizeBytes / 1_000_000_000.0)

    fun validate(): String? = when {
        modelId !in SUPPORTED_MODEL_IDS -> "unsupported model_id"
        version != PINNED_MODEL_VERSION -> "unsupported model version"
        architecture != "qwen2" -> "unsupported model architecture"
        layerCount != 36 || hiddenSize != 2048 -> "model dimensions do not match the Android executor"
        fileName != EXPECTED_FILE_NAME -> "model filename does not match the Android executor"
        sizeBytes != PINNED_SIZE_BYTES -> "model size does not match the pinned artifact"
        !sha256.matches(SHA256_PATTERN) -> "artifact_digest must be a SHA-256 digest"
        sha256 != PINNED_ARTIFACT_DIGEST -> "artifact digest does not match the pinned native model"
        !downloadUrl.isHttpsUrl() -> "model download URL must use HTTPS"
        else -> null
    }

    companion object {
        const val ASSET_FILE_NAME = "qwen2.5-3b-instruct.json"
        const val EXPECTED_FILE_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
        const val PINNED_MODEL_ID = "qwen2.5-3b-instruct"
        const val PINNED_MODEL_VERSION = "1.0.0"
        const val PINNED_SIZE_BYTES = 2_104_932_768L
        const val PINNED_ARTIFACT_DIGEST =
            "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
        private val SHA256_PATTERN = Regex("sha256:[0-9a-f]{64}")
        private val SUPPORTED_MODEL_IDS = setOf("qwen2.5-3b", "qwen2.5-3b-instruct")

        fun fromJson(json: String): ModelManifest {
            val root = JsonParser.parseString(json).asJsonObject
            val shard = root.getAsJsonArray("shards")
                ?.firstOrNull()
                ?.asJsonObject
                ?: throw IllegalArgumentException("model manifest must contain a download shard")

            val manifest = ModelManifest(
                modelId = root.requiredString("model_id"),
                version = root.requiredString("version"),
                architecture = root.requiredString("architecture"),
                layerCount = root.requiredInt("layers"),
                hiddenSize = root.requiredInt("hidden_size"),
                fileName = EXPECTED_FILE_NAME,
                downloadUrl = shard.requiredString("download_url"),
                sizeBytes = root.requiredLong("total_size_bytes"),
                sha256 = root.requiredString("artifact_digest")
            )
            manifest.validate()?.let { error ->
                throw IllegalArgumentException("invalid model manifest: $error")
            }
            val shardDigest = shard.requiredString("sha256")
            val shardSize = shard.requiredLong("size_bytes")
            require(shardDigest == manifest.sha256 && shardSize == manifest.sizeBytes) {
                "model shard digest and size must match the artifact manifest"
            }
            return manifest
        }

        private fun JsonObject.requiredString(name: String): String {
            val value = get(name)
            require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isString) {
                "model manifest field '$name' must be a string"
            }
            return value.asString
        }

        private fun JsonObject.requiredInt(name: String): Int {
            val value = get(name)
            require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isNumber) {
                "model manifest field '$name' must be a number"
            }
            return value.asInt
        }

        private fun JsonObject.requiredLong(name: String): Long {
            val value = get(name)
            require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isNumber) {
                "model manifest field '$name' must be a number"
            }
            return value.asLong
        }

        private fun String.isHttpsUrl(): Boolean = runCatching {
            URI(this).let { it.scheme == "https" && !it.host.isNullOrBlank() }
        }.getOrDefault(false)
    }
}
