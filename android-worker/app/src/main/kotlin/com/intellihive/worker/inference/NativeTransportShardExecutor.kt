package com.intellihive.worker.inference

import com.intellihive.worker.service.ShardExecutionResult as TransportResult
import com.intellihive.worker.service.ShardExecutor as TransportExecutor
import com.intellihive.worker.service.WorkerJobAssignment

/**
 * The sole Android transport-to-execution adapter. NativeShardExecutor sees
 * only ShardExecutionRequest and never WebSocket/Gson assignment types.
 */
class NativeTransportShardExecutor(
    private val native: NativeShardExecutor
) : TransportExecutor, AutoCloseable {
    override suspend fun execute(assignment: WorkerJobAssignment): TransportResult {
        assignment.validate()?.let { throw IllegalArgumentException(it) }
        val request = WorkerAssignmentAdapter.toExecutionRequest(assignment)
        val result = native.execute(request)
        WorkerAssignmentAdapter.validateResult(request, result, assignment.finalShard)
            ?.let { throw IllegalStateException(it) }
        return when (result.outputType) {
            ShardOutputType.ACTIVATION -> activationResult(assignment, result)
            ShardOutputType.TOKEN -> TransportResult(sampledTokenId = result.tokenId
                ?: throw IllegalStateException("token output requires tokenId"))
            ShardOutputType.EOS -> TransportResult(endOfSequence = true)
            ShardOutputType.LOGITS -> throw UnsupportedOperationException(
                "central sampling for logits output is not implemented")
        }
    }

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) =
        native.endSequence(jobId, sequenceId, completed)

    override fun close() {
        (native as? AutoCloseable)?.close()
    }

    private fun activationResult(
        assignment: WorkerJobAssignment,
        result: NativeShardExecutionResult
    ): TransportResult {
        val spec = checkNotNull(result.tensorSpec)
        val payload = checkNotNull(result.payload)
        return TransportResult(activation = Activation(
            passOrdinal = result.passOrdinal,
            kvTokenOffset = result.kvTokenOffsetBefore,
            tokenCount = assignment.tokenCount,
            jobId = assignment.jobId,
            requestId = assignment.requestId,
            sequenceId = assignment.sequenceId,
            modelId = assignment.modelId,
            modelVersion = assignment.modelVersion,
            modelArtifactDigest = assignment.modelArtifactDigest,
            sourceWorker = assignment.workerId,
            destinationWorker = checkNotNull(assignment.nextWorker),
            layer = assignment.layerEnd,
            dtype = spec.dtype.name,
            shape = spec.shape,
            layout = spec.layout.name,
            byteOrder = spec.byteOrder.name,
            byteLength = spec.byteLength,
            payload = payload,
            checksum = Activation.checksum(payload)
        ))
    }
}

class UnavailableNativeShardExecutor(private val reason: String) : NativeShardExecutor {
    override suspend fun execute(request: ShardExecutionRequest): NativeShardExecutionResult =
        throw UnsupportedOperationException("IntelHive native shard executor unavailable: $reason")

    override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) = Unit
}
