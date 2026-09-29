package com.intellihive.worker.inference

import com.google.gson.annotations.SerializedName
import java.security.MessageDigest

data class ModelSpec(
    @SerializedName("model_id")
    val modelId: String,
    val version: String,
    val architecture: String,
    @SerializedName("layer_count")
    val layerCount: Int,
    @SerializedName("model_path")
    val modelPath: String
) {
    fun validate(): String? = when {
        modelId.isBlank() -> "modelId is required"
        version.isBlank() -> "model version is required"
        architecture.isBlank() -> "model architecture is required"
        layerCount <= 0 -> "model layer count must be positive"
        modelPath.isBlank() -> "model path is required"
        else -> null
    }
}

data class TensorSpec(
    val shape: List<Long>,
    val dtype: String
) {
    fun validate(): String? = when {
        shape.isEmpty() -> "tensor shape is required"
        shape.any { it <= 0 } -> "tensor dimensions must be positive"
        dtype.isBlank() -> "tensor dtype is required"
        else -> null
    }
}

/** Layer bounds are inclusive; a worker owns every layer in this range. */
data class ShardSpec(
    @SerializedName("model_id")
    val modelId: String,
    val version: String,
    @SerializedName("start_layer")
    val startLayer: Int,
    @SerializedName("end_layer")
    val endLayer: Int,
    @SerializedName("memory_bytes")
    val memoryBytes: Long,
    @SerializedName("input_spec")
    val inputSpec: TensorSpec,
    @SerializedName("output_spec")
    val outputSpec: TensorSpec
) {
    fun validate(model: ModelSpec): String? = when {
        modelId != model.modelId -> "shard modelId does not match loaded model"
        version != model.version -> "shard version does not match loaded model"
        startLayer < 0 || endLayer < startLayer || endLayer >= model.layerCount ->
            "shard layer range is outside the model"
        memoryBytes < 0 -> "shard memoryBytes cannot be negative"
        inputSpec.validate() != null -> inputSpec.validate()
        outputSpec.validate() != null -> outputSpec.validate()
        else -> null
    }
}

enum class InferencePhase {
    PREFILL,
    DECODE
}

sealed interface ShardInput {
    data class Prompt(val text: String) : ShardInput
    data class TokenIds(val values: List<Long>) : ShardInput
    data class PreviousActivation(val value: Activation) : ShardInput
}

data class InferenceStep(
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("request_id")
    val requestId: String,
    @SerializedName("sequence_id")
    val sequenceId: String,
    val phase: InferencePhase,
    val position: Int,
    val input: ShardInput
) {
    fun validate(): String? = when {
        jobId.isBlank() -> "jobId is required"
        requestId.isBlank() -> "requestId is required"
        sequenceId.isBlank() -> "sequenceId is required"
        position < 0 -> "position cannot be negative"
        phase == InferencePhase.PREFILL && position != 0 -> "prefill position must be zero"
        phase == InferencePhase.DECODE && position == 0 -> "decode position must be greater than zero"
        phase == InferencePhase.DECODE && input is ShardInput.Prompt -> "decode input cannot be prompt text"
        phase == InferencePhase.DECODE &&
            input is ShardInput.TokenIds && input.values.size != 1 ->
            "decode input must contain exactly one token"
        input is ShardInput.Prompt && input.text.isBlank() -> "prompt cannot be blank"
        input is ShardInput.TokenIds &&
            (input.values.isEmpty() || input.values.any { it !in 0L..MAX_TOKEN_ID }) ->
            "token IDs must be non-empty unsigned 32-bit values"
        input is ShardInput.PreviousActivation -> input.value.validate()
        else -> null
    }

    companion object {
        private const val MAX_TOKEN_ID = 0xFFFFFFFFL
    }
}

/** Tensor output from the inclusive `layer` boundary on the source worker. */
data class Activation(
 val position: Int = 0,
    @SerializedName("job_id")
    val jobId: String,
    @SerializedName("request_id")
    val requestId: String,
    @SerializedName("sequence_id")
    val sequenceId: String,
    @SerializedName("model_id")
    val modelId: String,
    @SerializedName("model_version")
    val modelVersion: String,
    @SerializedName("source_worker")
    val sourceWorker: String,
    @SerializedName("destination_worker")
    val destinationWorker: String,
    val layer: Int,
    val dtype: String,
    val shape: List<Long>,
    val payload: ByteArray,
    @SerializedName("checksum")
    val checksum: String
) {
    fun validate(): String? = when {
        position < 0 -> "activation position cannot be negative"
        jobId.isBlank() -> "activation jobId is required"
        requestId.isBlank() -> "activation requestId is required"
        sequenceId.isBlank() -> "activation sequenceId is required"
        modelId.isBlank() -> "activation modelId is required"
        modelVersion.isBlank() -> "activation modelVersion is required"
        sourceWorker.isBlank() -> "activation sourceWorker is required"
        destinationWorker.isBlank() -> "activation destinationWorker is required"
        layer < 0 -> "activation layer cannot be negative"
        dtype.isBlank() -> "activation dtype is required"
        shape.isEmpty() || shape.any { it <= 0 } -> "activation shape must contain positive dimensions"
        payload.isEmpty() -> "activation payload cannot be empty"
        payloadSizeError() != null -> payloadSizeError()
        checksum != checksumFor(payload) -> "activation checksum mismatch"
        else -> null
    }

    private fun payloadSizeError(): String? {
        var expected = when (dtype) { "float32" -> 4L; "float16" -> 2L; "uint8" -> 1L; else -> return "unsupported activation dtype" }
        for (dimension in shape) {
            if (dimension > payload.size.toLong() / expected) return "activation shape does not match payload length"
            expected *= dimension
        }
        return if (expected != payload.size.toLong()) "activation shape does not match payload length" else null
    }

    fun snapshot() = copy(shape = shape.toList(), payload = payload.copyOf())

    fun matches(a: com.intellihive.worker.service.WorkerJobAssignment, output: Boolean): Boolean =
        jobId == a.jobId && requestId == a.requestId && sequenceId == a.sequenceId &&
        modelId == a.modelId && modelVersion == a.modelVersion && position == a.position &&
        layer == (if (output) a.layerEnd else a.layerStart - 1) &&
        sourceWorker == (if (output) a.workerId else a.previousWorker) &&
        destinationWorker == (if (output) a.nextWorker else a.workerId)

    companion object {
        fun create(
            jobId: String,
            requestId: String,
            sequenceId: String,
            modelId: String,
            modelVersion: String,
            sourceWorker: String,
            destinationWorker: String,
            layer: Int,
            dtype: String,
            shape: List<Long>,
            payload: ByteArray
        ) = Activation(
            jobId = jobId,
            requestId = requestId,
            sequenceId = sequenceId,
            modelId = modelId,
            modelVersion = modelVersion,
            sourceWorker = sourceWorker,
            destinationWorker = destinationWorker,
            layer = layer,
            dtype = dtype,
            shape = shape.toList(),
            payload = payload.copyOf(),
            checksum = checksumFor(payload)
        )

        private fun checksumFor(data: ByteArray): String =
            "sha256:" + MessageDigest.getInstance("SHA-256")
                .digest(data)
                .joinToString(separator = "") { byte -> "%02x".format(byte) }
    }
}

data class ShardInferenceResult(
    val activation: Activation? = null,
    @SerializedName("sampled_token_id")
    val sampledTokenId: Long? = null,
    @SerializedName("end_of_sequence")
    val endOfSequence: Boolean = false,
    @SerializedName("generated_text")
    val generatedText: ByteArray? = null
)

/**
 * IntelHive-owned contract. A backend loads the local model and its assigned
 * shard; it does not own scheduling or communicate with other workers.
 */
interface InferenceEngine {
    suspend fun loadModel(model: ModelSpec)

    suspend fun loadShard(shard: ShardSpec)

    suspend fun executeShard(shard: ShardSpec, step: InferenceStep): ShardInferenceResult

    suspend fun releaseShard(shard: ShardSpec)
}
