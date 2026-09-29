package com.intellihive.worker.service

import com.intellihive.worker.inference.Activation
import com.google.gson.GsonBuilder
import com.google.gson.TypeAdapter
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonWriter
import java.util.Base64
import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName

data class WorkerEnvelope(
    val type: String,
    val payload: JsonElement
)

data class WorkerJobAssignment(
 @SerializedName("request_id") val requestId: String = "",
 @SerializedName("model_version") val modelVersion: String = "",
 @SerializedName("worker_id") val workerId: String = "",
 @SerializedName("previous_worker") val previousWorker: String? = null,
 @SerializedName("next_worker") val nextWorker: String? = null,
    @SerializedName("assignment_id") val assignmentId: String = "",
    @SerializedName("job_id") val jobId: String = "",
    @SerializedName("model_id") val modelId: String = "",
    @SerializedName("shard_id") val shardId: String = "",
    val phase: String = "",
    @SerializedName("sequence_id") val sequenceId: String = "",
    val position: Int = 0,
    val prompt: String? = null,
    @SerializedName("input_token_ids") val inputTokenIds: List<Long>? = null,
    val activation: Activation? = null,
    @SerializedName("final_shard") val finalShard: Boolean = false,
    @SerializedName("layer_start") val layerStart: Int = -1,
    @SerializedName("layer_end") val layerEnd: Int = -1,
    val sequence: Long = 0,
    val payload: ByteArray? = null
) {
    fun validate(): String? {
        if (requestId.isBlank() || modelVersion.isBlank() || workerId.isBlank()) return "request_id, model_version, and worker_id are required"
 if (!finalShard && nextWorker.isNullOrBlank()) return "non-final shard requires next_worker"
 if (assignmentId.isBlank() || jobId.isBlank() || modelId.isBlank() ||
            shardId.isBlank() || sequenceId.isBlank()
        ) {
            return "assignment_id, job_id, model_id, shard_id, and sequence_id are required"
        }
        if (layerStart < 0 || layerEnd < layerStart) {
            return "assignment layer range is invalid"
        }
        if (sequence < 0 || position < 0) return "assignment position is invalid"
        if (phase != PREFILL && phase != DECODE) return "unsupported inference phase"

        val hasPrompt = !prompt.isNullOrEmpty()
        val hasTokens = !inputTokenIds.isNullOrEmpty()
        val hasActivation = activation != null
        if (listOf(hasPrompt, hasTokens, hasActivation).count { it } != 1) {
            return "exactly one of prompt, input_token_ids, or activation is required"
        }
        if (phase == PREFILL && position != 0) return "prefill position must be zero"
        if (phase == DECODE && position == 0) return "decode position must be greater than zero"
        if (phase == DECODE && hasPrompt) return "decode steps cannot contain prompt text"
        if (phase == DECODE && hasTokens && inputTokenIds?.size != 1) {
            return "decode steps must contain exactly one input token"
        }
        if (inputTokenIds?.any { it < 0L || it > MAX_TOKEN_ID } == true) {
            return "input_token_ids contains an out-of-range token ID"
        }
        if ((layerStart == 0) == hasActivation) return "input does not match shard layer boundary"
        activation?.let {
            it.validate()?.let { error -> return error }
            if (!it.matches(this, output = false)) return "activation does not match assignment"
        }
        return null
    }

    companion object {
        const val PREFILL = "PREFILL"
        const val DECODE = "DECODE"
        private const val MAX_TOKEN_ID = 0xFFFFFFFFL
    }
}

data class WorkerJobAccepted(
    @SerializedName("assignment_id") val assignmentId: String,
    val accepted: Boolean,
    val reason: String? = null
)

data class WorkerJobComplete(
    @SerializedName("assignment_id") val assignmentId: String,
    @SerializedName("job_id") val jobId: String,
    @SerializedName("worker_id") val workerId: String,
    val status: String = "completed",
    @SerializedName("sequence_id") val sequenceId: String,
    val position: Int,
    val activation: Activation? = null,
    @SerializedName("sampled_token_id") val sampledTokenId: Long? = null,
    @SerializedName("end_of_sequence") val endOfSequence: Boolean = false,
    @SerializedName("generated_text") val generatedText: ByteArray? = null,
    @SerializedName("tokens_per_second") val tokensPerSecond: Double? = null
)

data class WorkerJobFailed(
    @SerializedName("assignment_id") val assignmentId: String,
    @SerializedName("job_id") val jobId: String,
    @SerializedName("worker_id") val workerId: String,
    val reason: String,
    val unavailable: Boolean = false
)

data class WorkerSequenceEnd(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("sequence_id") val sequenceId: String,
    val completed: Boolean
)

data class ShardExecutionResult(
    val activation: Activation? = null,
    val sampledTokenId: Long? = null,
    val endOfSequence: Boolean = false,
    val generatedText: ByteArray? = null,
    val tokensPerSecond: Double? = null
)

interface ShardExecutor {
    suspend fun execute(assignment: WorkerJobAssignment): ShardExecutionResult

    suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean)
}

/**
 * The native layer-range executor is not present yet. Reject assignments
 * rather than returning success-shaped mock inference results.
 */
internal val workerProtocolGson = GsonBuilder()
    .registerTypeAdapter(ByteArray::class.java, object : TypeAdapter<ByteArray>() {
        override fun write(out: JsonWriter, value: ByteArray) { out.value(Base64.getEncoder().encodeToString(value)) }
        override fun read(input: JsonReader): ByteArray = Base64.getDecoder().decode(input.nextString())
    }.nullSafe()).create()
