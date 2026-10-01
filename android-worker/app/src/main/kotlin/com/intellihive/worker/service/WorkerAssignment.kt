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
    @SerializedName("model_artifact_digest") val modelArtifactDigest: String = "",
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
    @SerializedName("pass_ordinal") val passOrdinal: Long = 0,
    @SerializedName("kv_token_offset") val kvTokenOffset: Long = 0,
    @SerializedName("token_count") val tokenCount: Long = 0,
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
		if (!modelArtifactDigest.matches(Regex("sha256:[0-9a-f]{64}"))) return "model_artifact_digest must be a SHA-256 digest"
 if (!finalShard && nextWorker.isNullOrBlank()) return "non-final shard requires next_worker"
 if (assignmentId.isBlank() || jobId.isBlank() || modelId.isBlank() ||
            shardId.isBlank() || sequenceId.isBlank()
        ) {
            return "assignment_id, job_id, model_id, shard_id, and sequence_id are required"
        }
        if (layerStart < 0 || layerEnd < layerStart) {
            return "assignment layer range is invalid"
        }
        if (sequence < 0 || position < 0 || passOrdinal < 0 || kvTokenOffset < 0 || tokenCount < 0) return "assignment execution metadata is invalid"
        if (phase != PREFILL && phase != DECODE) return "unsupported inference phase"

        val hasPrompt = !prompt.isNullOrEmpty()
        val hasTokens = !inputTokenIds.isNullOrEmpty()
        val hasActivation = activation != null
        if (listOf(hasPrompt, hasTokens, hasActivation).count { it } != 1) {
            return "exactly one of prompt, input_token_ids, or activation is required"
        }
        if (phase == PREFILL && passOrdinal != 0L) return "prefill pass_ordinal must be zero"
        if (phase == DECODE && passOrdinal == 0L) return "decode pass_ordinal must be greater than zero"
        if (hasPrompt && (phase != PREFILL || layerStart != 0 || passOrdinal != 0L ||
                kvTokenOffset != 0L || tokenCount != 0L)) {
            return "prompt prefill must start at layer zero with an unresolved token count"
        }
        if (!hasPrompt && tokenCount == 0L) return "token_count must be positive"
        if (phase == DECODE && hasTokens && inputTokenIds?.size != 1) {
            return "decode steps must contain exactly one input token"
        }
		if (hasTokens && inputTokenIds?.size?.toLong() != tokenCount) return "token_count does not match input_token_ids"
        if (inputTokenIds?.any { it < 0L || it > MAX_TOKEN_ID } == true) {
            return "input_token_ids contains an out-of-range token ID"
        }
        if ((layerStart == 0) == hasActivation) return "input does not match shard layer boundary"
        activation?.let {
            it.validate()?.let { error -> return error }
            if (!it.matches(this, output = false)) return "activation does not match assignment"
			if (it.canonicalSpecOrNull() == null || it.modelArtifactDigest != modelArtifactDigest ||
				it.passOrdinal != passOrdinal || it.kvTokenOffset != kvTokenOffset || it.tokenCount != tokenCount) {
				return "activation does not match canonical execution metadata"
			}
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
    @SerializedName("pass_ordinal") val passOrdinal: Long = 0,
    @SerializedName("kv_token_offset_before") val kvTokenOffsetBefore: Long = 0,
    @SerializedName("kv_token_offset_after") val kvTokenOffsetAfter: Long = 0,
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
    val passOrdinal: Long = 0,
    val kvTokenOffsetBefore: Long = 0,
    val kvTokenOffsetAfter: Long = 0,
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

internal val workerProtocolGson = GsonBuilder()
    .registerTypeAdapter(ByteArray::class.java, object : TypeAdapter<ByteArray>() {
        override fun write(out: JsonWriter, value: ByteArray) { out.value(Base64.getEncoder().encodeToString(value)) }
        override fun read(input: JsonReader): ByteArray = Base64.getDecoder().decode(input.nextString())
    }.nullSafe()).create()
