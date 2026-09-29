package com.intellihive.worker.inference

import com.intellihive.worker.service.WorkerJobAssignment

/**
 * Backend-facing contract. These types deliberately contain no WebSocket or
 * Gson DTOs: WorkerAssignmentAdapter is the sole transport boundary.
 */
enum class ExecutionMode { PREFILL, DECODE }
enum class TensorDType { F16, F32 }
enum class TensorLayout { ROW_MAJOR_CONTIGUOUS }
enum class TensorByteOrder { LITTLE_ENDIAN }
enum class ShardOutputType { ACTIVATION, LOGITS, TOKEN, EOS }

data class CanonicalTensorSpec(
    val dtype: TensorDType,
    val shape: List<Long>,
    val layout: TensorLayout = TensorLayout.ROW_MAJOR_CONTIGUOUS,
    val byteOrder: TensorByteOrder = TensorByteOrder.LITTLE_ENDIAN,
    val byteLength: Long
) {
    fun validate(): String? {
        if (shape.isEmpty() || shape.any { it <= 0 }) return "tensor shape must contain positive dimensions"
        val elementBytes = when (dtype) { TensorDType.F16 -> 2L; TensorDType.F32 -> 4L }
        var expected = elementBytes
        for (dimension in shape) {
            if (expected > Long.MAX_VALUE / dimension) return "tensor byte length overflows"
            expected *= dimension
        }
        return if (byteLength == expected) null else "tensor byte length does not match dtype and shape"
    }
}

data class ExecutionShardSpec(
    val id: String,
    val modelId: String,
    val modelVersion: String,
    val modelArtifactDigest: String,
    val layerStart: Int,
    val layerEnd: Int
)

sealed interface ExecutionInput {
    data class TokenIDs(val values: List<Long>) : ExecutionInput
    data class Activation(val payload: ByteArray, val spec: CanonicalTensorSpec) : ExecutionInput
}

data class ShardExecutionRequest(
    val requestId: String,
    val sequenceId: String,
    val shard: ExecutionShardSpec,
    val passOrdinal: Long,
    val mode: ExecutionMode,
    val kvTokenOffset: Long,
    val tokenCount: Long,
    val input: ExecutionInput
)

data class ShardExecutionResult(
    val requestId: String,
    val sequenceId: String,
    val passOrdinal: Long,
    val kvTokenOffsetBefore: Long,
    val kvTokenOffsetAfter: Long,
    val outputType: ShardOutputType,
    val payload: ByteArray? = null,
    val tokenId: Long? = null
)

/** Native implementations own only their model shard and local KV cache. */
interface NativeShardExecutor {
    suspend fun execute(request: ShardExecutionRequest): ShardExecutionResult
    suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean)
}

object WorkerAssignmentAdapter {
    fun toExecutionRequest(assignment: WorkerJobAssignment): ShardExecutionRequest {
        require(assignment.modelArtifactDigest.matches(Regex("sha256:[0-9a-f]{64}"))) { "model_artifact_digest must be a SHA-256 digest" }
        require(assignment.passOrdinal >= 0 && assignment.kvTokenOffset >= 0 && assignment.tokenCount > 0) { "pass_ordinal, kv_token_offset, and token_count are required" }
        val mode = when (assignment.phase) {
            WorkerJobAssignment.PREFILL -> ExecutionMode.PREFILL
            WorkerJobAssignment.DECODE -> ExecutionMode.DECODE
            else -> error("unsupported execution mode")
        }
        require(mode != ExecutionMode.DECODE || assignment.tokenCount == 1L) { "decode token_count must be one" }
        val input = assignment.inputTokenIds?.let { ExecutionInput.TokenIDs(it) }
            ?: assignment.activation?.let { error("activation envelope requires canonical tensor fields before native execution") }
            ?: error("native execution requires token IDs or a canonical activation")
        return ShardExecutionRequest(
            requestId = assignment.requestId, sequenceId = assignment.sequenceId,
            shard = ExecutionShardSpec(assignment.shardId, assignment.modelId, assignment.modelVersion,
                assignment.modelArtifactDigest, assignment.layerStart, assignment.layerEnd),
            passOrdinal = assignment.passOrdinal, mode = mode, kvTokenOffset = assignment.kvTokenOffset,
            tokenCount = assignment.tokenCount, input = input
        )
    }

    fun validateResult(request: ShardExecutionRequest, result: ShardExecutionResult, finalShard: Boolean): String? {
        if (result.requestId != request.requestId || result.sequenceId != request.sequenceId || result.passOrdinal != request.passOrdinal) return "execution result identity does not match request"
        if (result.kvTokenOffsetBefore != request.kvTokenOffset || result.kvTokenOffsetAfter != request.kvTokenOffset + request.tokenCount) return "execution result KV token range is not contiguous"
        return if (finalShard) {
            if (result.outputType !in setOf(ShardOutputType.LOGITS, ShardOutputType.TOKEN, ShardOutputType.EOS)) "final shard must return logits, token, or EOS" else null
        } else if (result.outputType != ShardOutputType.ACTIVATION) "intermediate shard must return activation" else null
    }
}
