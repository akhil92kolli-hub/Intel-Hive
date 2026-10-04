package com.intellihive.worker.inference

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.intellihive.worker.model.ModelManifest
import com.intellihive.worker.model.RequiredModelManager

enum class InferenceBackend {
    CPU,
    VULKAN
}

internal class NativeLayerRangeBindings {
    fun nativeLoadModel(modelPath: String): Long = NativeBridge.nativeLoadModel(modelPath)
    fun nativeLayerCount(modelHandle: Long): Int = NativeBridge.nativeLayerCount(modelHandle)
    fun nativeEmbeddingSize(modelHandle: Long): Int = NativeBridge.nativeEmbeddingSize(modelHandle)
    fun nativeTokenize(modelHandle: Long, prompt: String): LongArray =
        NativeBridge.nativeTokenize(modelHandle, prompt.toByteArray(Charsets.UTF_8))
    fun nativeCreateShard(
        modelHandle: Long,
        firstLayer: Int,
        lastLayer: Int,
        graphThreads: Int,
        useVulkan: Boolean
    ): Long = NativeBridge.nativeCreateShard(
        modelHandle,
        firstLayer,
        lastLayer,
        graphThreads,
        useVulkan
    )
    fun nativeLayerWeightBytes(modelHandle: Long, firstLayer: Int, lastLayer: Int): Long =
        NativeBridge.nativeLayerWeightBytes(modelHandle, firstLayer, lastLayer)
    fun nativeShardUsedVulkan(shardHandle: Long): Boolean =
        NativeBridge.nativeShardUsedVulkan(shardHandle)
    fun nativeShardBackendStats(shardHandle: Long): String =
        NativeBridge.nativeShardBackendStats(shardHandle)
    fun nativeExecute(
        shardHandle: Long,
        sequenceId: String,
        tokenOffset: Int,
        tokenCount: Int,
        tokenIds: LongArray?,
        activation: ByteArray?,
        maxDurationMs: Long
    ): ByteArray = NativeBridge.nativeExecute(
        shardHandle,
        sequenceId,
        tokenOffset,
        tokenCount,
        tokenIds,
        activation,
        maxDurationMs
    )
    fun nativeEndSequence(shardHandle: Long, sequenceId: String) =
        NativeBridge.nativeEndSequence(shardHandle, sequenceId)
    fun nativeDestroyShard(shardHandle: Long) = NativeBridge.nativeDestroyShard(shardHandle)
    fun nativeUnloadModel(modelHandle: Long) = NativeBridge.nativeUnloadModel(modelHandle)
}

/**
 * Bridges the transport-neutral shard request to the host-validated Qwen2
 * layer-range implementation. Activation payloads are row-major F32 values
 * shaped [tokenCount, hiddenSize], encoded in little-endian byte order.
 */
class NativeLayerRangeShardExecutor(
    context: Context,
    private val modelFile: File = RequiredModelManager.defaultModelFile(context),
    graphThreads: Int = 0
) : NativeShardExecutor, NativePromptTokenizer, NativeExecutionDeadlineController, AutoCloseable {
    private val bindings = NativeLayerRangeBindings()
    private val gson = Gson()
    private val lock = Any()
    private val modelHandles = mutableMapOf<ModelKey, LoadedModel>()
    private val shardHandles = mutableMapOf<ShardKey, Long>()
    @Volatile
    private var graphThreads = graphThreads
    @Volatile
    private var backend = InferenceBackend.CPU
    @Volatile
    private var gpuLayers = 0
    @Volatile
    private var vulkanShardOverrides: Set<String>? = null
    @Volatile
    private var executionDeadlineNanos: Long? = null

    init {
        require(graphThreads in 0..8) { "graphThreads must be between 0 and 8" }
    }

    fun configureGraphThreads(threads: Int) {
        require(threads in 0..8) { "graphThreads must be between 0 and 8" }
        synchronized(lock) {
            if (graphThreads != threads) clearShardHandles()
            graphThreads = threads
        }
    }

    fun configureBackend(backend: InferenceBackend, gpuLayers: Int = 0) {
        require(gpuLayers in setOf(0, 10, 20, MODEL_LAYER_COUNT)) {
            "GPU offload must end at a supported shard boundary: 0, 10, 20, or $MODEL_LAYER_COUNT layers"
        }
        require((backend == InferenceBackend.CPU) == (gpuLayers == 0)) {
            "CPU execution requires 0 GPU layers and Vulkan execution requires at least 1 GPU layer"
        }
        synchronized(lock) {
            if (this.backend != backend || this.gpuLayers != gpuLayers ||
                vulkanShardOverrides != null
            ) clearShardHandles()
            this.backend = backend
            this.gpuLayers = gpuLayers
            vulkanShardOverrides = null
        }
    }

    fun configureShardBackends(vulkanShardIds: Set<String>) {
        val knownShardIds = QwenShardCatalog.ranges.map { it.shardId }.toSet()
        require(vulkanShardIds.all { it in knownShardIds }) {
            "Unknown Vulkan shard ID in backend profile"
        }
        synchronized(lock) {
            if (vulkanShardOverrides != vulkanShardIds) clearShardHandles()
            backend = if (vulkanShardIds.isEmpty()) InferenceBackend.CPU else InferenceBackend.VULKAN
            gpuLayers = 0
            vulkanShardOverrides = vulkanShardIds.toSet()
        }
    }

    override fun setExecutionDeadlineNanos(deadlineNanos: Long?) {
        require(deadlineNanos == null || deadlineNanos > 0L) {
            "execution deadline must be a positive monotonic timestamp"
        }
        executionDeadlineNanos = deadlineNanos
    }

    suspend fun layerWeightBytes(firstLayer: Int, lastLayer: Int): Long =
        withContext(Dispatchers.IO) {
            synchronized(lock) {
                val model = loadedModel()
                require(firstLayer >= 0 && lastLayer >= firstLayer && lastLayer < model.layerCount) {
                    "requested layer range $firstLayer-$lastLayer is invalid for this model"
                }
                bindings.nativeLayerWeightBytes(model.handle, firstLayer, lastLayer)
            }
        }

    fun didUseVulkan(): Boolean = synchronized(lock) {
        shardHandles.values.any(bindings::nativeShardUsedVulkan)
    }

    fun usedVulkanShardIds(): Set<String> = synchronized(lock) {
        shardHandles.filterValues(bindings::nativeShardUsedVulkan).keys.mapNotNull { key ->
            QwenShardCatalog.ranges.firstOrNull {
                it.layerStart == key.firstLayer && it.layerEnd == key.lastLayer
            }?.shardId
        }.toSet()
    }

    fun backendStatistics(): Map<String, Map<String, Any?>> = synchronized(lock) {
        val type = object : TypeToken<Map<String, Any?>>() {}.type
        shardHandles.mapNotNull { (key, handle) ->
            val shardId = QwenShardCatalog.ranges.firstOrNull {
                it.layerStart == key.firstLayer && it.layerEnd == key.lastLayer
            }?.shardId ?: return@mapNotNull null
            val parsed = gson.fromJson<Map<String, Any?>>(
                bindings.nativeShardBackendStats(handle),
                type
            ) ?: emptyMap()
            shardId to parsed
        }.toMap()
    }

    suspend fun prepareModel(): Unit = withContext(Dispatchers.IO) {
        synchronized(lock) {
            check(NativeRuntime.isAvailable()) {
                "IntelHive native runtime is unavailable: ${NativeRuntime.unavailableReason()}"
            }
            loadedModel()
            Unit
        }
    }

    suspend fun prepareShards(ranges: List<ModelShardRange>) = withContext(Dispatchers.IO) {
        synchronized(lock) {
            val model = loadedModel()
            ranges.forEach { range ->
                require(range.layerStart >= 0 && range.layerEnd >= range.layerStart &&
                    range.layerEnd < model.layerCount) {
                    "requested layer range ${range.layerStart}-${range.layerEnd} is invalid for this model"
                }
                shardHandle(model, range)
            }
        }
    }

    override suspend fun tokenize(prompt: String): List<Long> = withContext(Dispatchers.Default) {
        require(prompt.isNotBlank()) { "prompt must not be blank" }
        synchronized(lock) {
            bindings.nativeTokenize(loadedModel().handle, prompt).toList().also { tokens ->
                require(tokens.isNotEmpty()) { "GGUF tokenizer returned no tokens" }
            }
        }
    }

    override suspend fun execute(request: ShardExecutionRequest): NativeShardExecutionResult =
        withContext(Dispatchers.Default) {
            validateRequest(request)
            synchronized(lock) {
                check(NativeRuntime.isAvailable()) {
                    "IntelHive native runtime is unavailable: ${NativeRuntime.unavailableReason()}"
                }
                val model = loadedModel()
                require(request.shard.layerEnd < model.layerCount) {
                    "requested layer range exceeds the loaded model"
                }
                val shardHandle = shardHandle(
                    model,
                    ModelShardRange(request.shard.id, request.shard.layerStart, request.shard.layerEnd)
                )

                val tokenIds: LongArray?
                val activation: ByteArray?
                when (val input = request.input) {
                    is ExecutionInput.TokenIDs -> {
                        tokenIds = input.values.toLongArray()
                        activation = null
                    }
                    is ExecutionInput.Activation -> {
                        require(input.spec.dtype == TensorDType.F32 &&
                            input.spec.layout == TensorLayout.ROW_MAJOR_CONTIGUOUS &&
                            input.spec.byteOrder == TensorByteOrder.LITTLE_ENDIAN) {
                            "Android Qwen2 shards currently accept only row-major little-endian F32 activations"
                        }
                        val expectedShape = listOf(request.tokenCount, model.embeddingSize.toLong())
                        require(input.spec.shape == expectedShape) {
                            "activation shape must be [token_count, hidden_size]"
                        }
                        require(input.spec.validate() == null &&
                            input.spec.byteLength == input.payload.size.toLong()) {
                            "activation tensor metadata does not match its payload"
                        }
                        tokenIds = null
                        activation = input.payload
                    }
                }

                val tokenCount = Math.toIntExact(request.tokenCount)
                val nativeResult = bindings.nativeExecute(
                    shardHandle,
                    request.sequenceId,
                    Math.toIntExact(request.kvTokenOffset),
                    tokenCount,
                    tokenIds,
                    activation,
                    remainingExecutionTimeMs()
                )
                require(nativeResult.isNotEmpty()) { "Native shard returned an empty result envelope" }

                when (nativeResult[0].toInt() and 0xff) {
                    NATIVE_ACTIVATION -> {
                        val payload = nativeResult.copyOfRange(1, nativeResult.size)
                        val expectedBytes = Math.multiplyExact(
                            Math.multiplyExact(request.tokenCount, model.embeddingSize.toLong()),
                            Float.SIZE_BYTES.toLong()
                        )
                        require(payload.size.toLong() == expectedBytes) {
                            "Native activation result length does not match [token_count, hidden_size]"
                        }
                        val spec = CanonicalTensorSpec(
                            dtype = TensorDType.F32,
                            shape = listOf(request.tokenCount, model.embeddingSize.toLong()),
                            byteLength = payload.size.toLong()
                        )
                        NativeShardExecutionResult(
                            requestId = request.requestId,
                            sequenceId = request.sequenceId,
                            passOrdinal = request.passOrdinal,
                            kvTokenOffsetBefore = request.kvTokenOffset,
                            kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
                            outputType = ShardOutputType.ACTIVATION,
                            payload = payload,
                            tensorSpec = spec
                        )
                    }
                    NATIVE_TOKEN -> {
                        require(nativeResult.size == 1 + java.lang.Long.BYTES) {
                            "Native token result has an invalid envelope length"
                        }
                        val tokenId = ByteBuffer.wrap(nativeResult, 1, java.lang.Long.BYTES)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .long
                        require(tokenId in 0L..0xFFFFFFFFL) { "Native token ID is outside the supported range" }
                        NativeShardExecutionResult(
                            requestId = request.requestId,
                            sequenceId = request.sequenceId,
                            passOrdinal = request.passOrdinal,
                            kvTokenOffsetBefore = request.kvTokenOffset,
                            kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
                            outputType = ShardOutputType.TOKEN,
                            tokenId = tokenId
                        )
                    }
                    else -> error("Native shard returned an unsupported output type")
                }
            }
        }

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) {
        withContext(Dispatchers.Default) {
            synchronized(lock) {
                shardHandles.values.forEach { bindings.nativeEndSequence(it, sequenceId) }
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            executionDeadlineNanos = null
            clearShardHandles()
            modelHandles.values.forEach { bindings.nativeUnloadModel(it.handle) }
            modelHandles.clear()
        }
    }

    private fun remainingExecutionTimeMs(): Long {
        val deadline = executionDeadlineNanos ?: return 0L
        val remainingNanos = deadline - System.nanoTime()
        check(remainingNanos > 0L) { "Native execution deadline exceeded" }
        return ((remainingNanos + NANOS_PER_MILLISECOND - 1L) / NANOS_PER_MILLISECOND)
            .coerceAtLeast(1L)
    }

    private fun validateRequest(request: ShardExecutionRequest) {
        require(request.requestId.isNotBlank() && request.sequenceId.isNotBlank()) {
            "request and sequence IDs are required"
        }
        require(
            request.shard.modelId in SUPPORTED_MODEL_IDS &&
                request.shard.modelVersion in SUPPORTED_MODEL_VERSIONS
        ) {
            "Android native execution supports only the configured Qwen2.5-3B model versions"
        }
        require(request.shard.modelArtifactDigest == ModelManifest.PINNED_ARTIFACT_DIGEST) {
            "requested model digest is not the pinned Qwen2.5-3B Q4_K_M artifact"
        }
        require(request.shard.layerStart >= 0 && request.shard.layerEnd >= request.shard.layerStart) {
            "requested layer range is invalid"
        }
        require(request.passOrdinal >= 0 && request.kvTokenOffset >= 0 && request.tokenCount > 0) {
            "request execution counters are invalid"
        }
        require(request.tokenCount <= Int.MAX_VALUE && request.kvTokenOffset <= Int.MAX_VALUE) {
            "request exceeds the native executor's supported token range"
        }
        when (val input = request.input) {
            is ExecutionInput.TokenIDs -> {
                require(request.shard.layerStart == 0) {
                    "token IDs can only be embedded by the first model shard"
                }
                require(input.values.size.toLong() == request.tokenCount) {
                    "token count does not match token ID input"
                }
            }
            is ExecutionInput.Activation -> require(request.shard.layerStart > 0) {
                "the first model shard must receive token IDs"
            }
        }
    }

    private fun loadModel(key: ModelKey): LoadedModel {
        require(modelFile.isFile && modelFile.canRead()) {
            "Qwen2.5-3B model is not installed at ${modelFile.absolutePath}"
        }
        val actualDigest = sha256(modelFile)
        require(actualDigest == ModelManifest.PINNED_ARTIFACT_DIGEST.removePrefix("sha256:")) {
            "Installed Qwen2.5-3B model digest mismatch: expected ${ModelManifest.PINNED_ARTIFACT_DIGEST}, got sha256:$actualDigest"
        }
        val handle = bindings.nativeLoadModel(modelFile.absolutePath)
        check(handle > 0L) { "Native model loading returned an invalid handle" }
        try {
            val layerCount = bindings.nativeLayerCount(handle)
            val embeddingSize = bindings.nativeEmbeddingSize(handle)
            require(layerCount == MODEL_LAYER_COUNT && embeddingSize == MODEL_HIDDEN_SIZE) {
                "Loaded GGUF dimensions do not match the pinned Qwen2.5-3B model manifest"
            }
            return LoadedModel(handle, layerCount, embeddingSize).also { modelHandles[key] = it }
        } catch (error: Exception) {
            bindings.nativeUnloadModel(handle)
            throw error
        }
    }

    private fun loadedModel(): LoadedModel {
        val key = ModelKey(MODEL_ID, MODEL_VERSION, ModelManifest.PINNED_ARTIFACT_DIGEST)
        return modelHandles[key] ?: loadModel(key)
    }

    private fun shardHandle(model: LoadedModel, range: ModelShardRange): Long {
        val key = ModelKey(MODEL_ID, MODEL_VERSION, ModelManifest.PINNED_ARTIFACT_DIGEST)
        val useVulkan = vulkanShardOverrides?.let { selectedShardIds ->
            QwenShardCatalog.ranges.any {
                it.shardId in selectedShardIds &&
                    it.layerStart == range.layerStart &&
                    it.layerEnd == range.layerEnd
            }
        } ?: (backend == InferenceBackend.VULKAN && range.layerStart < gpuLayers)
        val shardKey = ShardKey(key, range.layerStart, range.layerEnd, graphThreads, useVulkan)
        return shardHandles[shardKey] ?: bindings.nativeCreateShard(
            model.handle,
            range.layerStart,
            range.layerEnd,
            graphThreads,
            useVulkan
        ).also { handle ->
            check(handle > 0L) { "Native shard creation returned an invalid handle" }
            shardHandles[shardKey] = handle
        }
    }

    private fun clearShardHandles() {
        shardHandles.values.forEach(bindings::nativeDestroyShard)
        shardHandles.clear()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val hexDigits = "0123456789abcdef"
        return buildString(64) {
            digest.digest().forEach { byte ->
                val value = byte.toInt() and 0xff
                append(hexDigits[value ushr 4])
                append(hexDigits[value and 0x0f])
            }
        }
    }

    private data class ModelKey(val modelId: String, val version: String, val digest: String)
    private data class ShardKey(
        val model: ModelKey,
        val firstLayer: Int,
        val lastLayer: Int,
        val graphThreads: Int,
        val useVulkan: Boolean
    )
    private data class LoadedModel(val handle: Long, val layerCount: Int, val embeddingSize: Int)

    companion object {
        private const val NATIVE_ACTIVATION = 1
        private const val NATIVE_TOKEN = 2
        private const val MODEL_ID = ModelManifest.PINNED_MODEL_ID
        private const val MODEL_LAYER_COUNT = 36
        private const val MODEL_HIDDEN_SIZE = 2048
        private const val MODEL_VERSION = ModelManifest.PINNED_MODEL_VERSION
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private val SUPPORTED_MODEL_IDS = setOf("qwen2.5-3b", "qwen2.5-3b-instruct")
        private val SUPPORTED_MODEL_VERSIONS = setOf("1", "v1", MODEL_VERSION)
    }
}
