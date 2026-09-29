package com.intellihive.worker.inference

import com.intellihive.worker.service.ShardExecutionResult
import com.intellihive.worker.service.ShardExecutor
import com.intellihive.worker.service.WorkerJobAssignment

data class LayerShardInvocation(
    val requestId: String,
    val modelVersion: String,
    val workerId: String,
    val nextWorker: String?,
    val jobId: String,
    val modelId: String,
    val shardId: String,
    val layerStart: Int,
    val layerEnd: Int,
    val phase: String,
    val sequenceId: String,
    val position: Int,
    val prompt: String?,
    val inputTokenIds: List<Long>?,
    val activation: Activation?,
    val finalShard: Boolean
)

/**
 * Adapter boundary around llama.cpp/ggml internals. An implementation owns
 * model tensors and KV-cache state for the requested shard layer range.
 */
interface LlamaCppLayerBackend {
    suspend fun execute(invocation: LayerShardInvocation): ShardExecutionResult

    suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean)
}

/**
 * Coordinates IntelHive sequence lifecycle and validates pass ordering while
 * delegating the actual partial-graph compute to a llama.cpp-backed adapter.
 */
class IntelHiveExecutionWrapper(
    private val backend: LlamaCppLayerBackend,
    private val maxActiveSequences: Int = DEFAULT_MAX_ACTIVE_SEQUENCES
) : ShardExecutor {
    private data class SequenceState(
        val jobId: String,
        val modelId: String,
        val modelVersion: String,
        val requestId: String,
        val workerId: String,
        val previousWorker: String?,
        val nextWorker: String?,
        val shardId: String,
        val layerStart: Int,
        val layerEnd: Int,
        val finalShard: Boolean,
        var nextPosition: Int,
        var inFlight: Boolean
    )

    private val stateLock = Any()
    private val sequences = mutableMapOf<String, SequenceState>()

    init {
        require(maxActiveSequences > 0) { "maxActiveSequences must be positive" }
    }

    override suspend fun execute(assignment: WorkerJobAssignment): ShardExecutionResult {
        assignment.validate()?.let { throw IllegalArgumentException(it) }
        val state = beginPass(assignment)
        try {
            val result = backend.execute(
                LayerShardInvocation(
                    requestId = assignment.requestId, modelVersion = assignment.modelVersion, workerId = assignment.workerId, nextWorker = assignment.nextWorker,
                    jobId = assignment.jobId,
                    modelId = assignment.modelId,
                    shardId = assignment.shardId,
                    layerStart = assignment.layerStart,
                    layerEnd = assignment.layerEnd,
                    phase = assignment.phase,
                    sequenceId = assignment.sequenceId,
                    position = assignment.position,
                    prompt = assignment.prompt,
                    inputTokenIds = assignment.inputTokenIds?.toList(),
                    activation = assignment.activation?.snapshot(),
                    finalShard = assignment.finalShard
                )
            )
            validateResult(assignment, result)?.let { throw IllegalStateException(it) }
            completePass(assignment)
            return result.copy(
                activation = result.activation?.snapshot(),
                generatedText = result.generatedText?.copyOf()
            )
        } catch (error: Exception) {
            abortPass(assignment, state)
            try {
                backend.endSequence(assignment.jobId, assignment.sequenceId, completed = false)
            } catch (cleanupError: Exception) {
                error.addSuppressed(cleanupError)
            }
            throw error
        }
    }

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) {
        val state = synchronized(stateLock) {
            val existing = sequences[sequenceId]
                ?: return
            require(existing.jobId == jobId) { "sequence_end job_id does not match active sequence" }
            require(!existing.inFlight) { "cannot end a sequence while a shard pass is running" }
            sequences.remove(sequenceId)
            existing
        }
        backend.endSequence(state.jobId, sequenceId, completed)
    }

    private fun beginPass(assignment: WorkerJobAssignment): SequenceState =
        synchronized(stateLock) {
            val existing = sequences[assignment.sequenceId]
            if (assignment.phase == WorkerJobAssignment.PREFILL) {
                require(existing == null) { "prefill sequence already exists" }
                require(sequences.size < maxActiveSequences) { "worker has reached its active sequence limit" }
                val state = SequenceState(
 modelVersion = assignment.modelVersion, requestId = assignment.requestId, workerId = assignment.workerId, previousWorker = assignment.previousWorker, nextWorker = assignment.nextWorker,
                    jobId = assignment.jobId,
                    modelId = assignment.modelId,
                    shardId = assignment.shardId,
                    layerStart = assignment.layerStart,
                    layerEnd = assignment.layerEnd,
                    finalShard = assignment.finalShard,
                    nextPosition = 1,
                    inFlight = true
                )
                sequences[assignment.sequenceId] = state
                state
            } else {
                val state = existing ?: throw IllegalStateException("decode sequence has no completed prefill")
                require(!state.inFlight) { "sequence already has an in-flight shard pass" }
                require(state.jobId == assignment.jobId &&
                    state.modelId == assignment.modelId &&
 state.modelVersion == assignment.modelVersion && state.requestId == assignment.requestId && state.workerId == assignment.workerId && state.previousWorker == assignment.previousWorker && state.nextWorker == assignment.nextWorker &&
                    state.shardId == assignment.shardId &&
                    state.layerStart == assignment.layerStart &&
                    state.layerEnd == assignment.layerEnd &&
                    state.finalShard == assignment.finalShard
                ) { "decode assignment does not match the active shard sequence" }
                require(assignment.position == state.nextPosition) {
                    "decode position ${assignment.position} does not match expected ${state.nextPosition}"
                }
                state.inFlight = true
                state
            }
        }

    private fun completePass(assignment: WorkerJobAssignment) {
        synchronized(stateLock) {
            val state = sequences[assignment.sequenceId]
                ?: throw IllegalStateException("sequence ended while shard pass was running")
            state.nextPosition = assignment.position + 1
            state.inFlight = false
        }
    }

    private fun abortPass(assignment: WorkerJobAssignment, state: SequenceState) {
        synchronized(stateLock) {
            val current = sequences[assignment.sequenceId]
            if (current !== state) return
            sequences.remove(assignment.sequenceId)
        }
    }

    private fun validateResult(
        assignment: WorkerJobAssignment,
        result: ShardExecutionResult
    ): String? {
        if (assignment.finalShard) {
            val hasToken = result.sampledTokenId != null
            if (hasToken == result.endOfSequence || result.activation != null) {
                return "final shard must return exactly one sampled token or EOS"
            }
        } else if (result.activation == null ||
            result.sampledTokenId != null || result.endOfSequence
        ) {
            return "non-final shard must return only a non-empty activation"
        }
        result.activation?.let {
            it.validate()?.let { error -> return error }
            if (!it.matches(assignment, output = true)) return "output activation does not match assignment"
        }
        return null
    }

    companion object {
        const val DEFAULT_MAX_ACTIVE_SEQUENCES = 8
    }
}

class UnavailableLlamaCppLayerBackend : LlamaCppLayerBackend {
    override suspend fun execute(invocation: LayerShardInvocation): ShardExecutionResult {
        throw UnsupportedOperationException(
            "IntelHive llama.cpp layer-range backend is not built into this Android app"
        )
    }

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) = Unit
}

class IntelHiveShardExecutor(backend: LlamaCppLayerBackend) : ShardExecutor {
    private val wrapper = IntelHiveExecutionWrapper(backend)

    override suspend fun execute(assignment: WorkerJobAssignment): ShardExecutionResult =
        wrapper.execute(assignment)

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) =
        wrapper.endSequence(jobId, sequenceId, completed)
}
