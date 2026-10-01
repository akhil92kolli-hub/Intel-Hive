package com.intellihive.worker.inference

import com.intellihive.worker.service.WorkerJobAssignment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.coroutines.runBlocking

class ExecutionContractTest {
    @Test
    fun transportAdapterTokenizesPromptBeforeNativeExecution() = runBlocking {
        val native = RecordingPromptExecutor()
        val adapter = NativeTransportShardExecutor(native)
        val assignment = WorkerJobAssignment(
            requestId = "request-1",
            modelVersion = "1",
            modelArtifactDigest = MODEL_DIGEST,
            workerId = "worker-a",
            assignmentId = "assignment-1",
            jobId = "job-1",
            modelId = "qwen2.5-3b",
            shardId = "layers-0",
            phase = WorkerJobAssignment.PREFILL,
            sequenceId = "sequence-1",
            passOrdinal = 0,
            kvTokenOffset = 0,
            tokenCount = 0,
            prompt = "hello",
            finalShard = true,
            layerStart = 0,
            layerEnd = 35
        )

        val result = adapter.execute(assignment)

        assertEquals("hello", native.prompt)
        assertEquals(listOf(17L, 42L), (native.request!!.input as ExecutionInput.TokenIDs).values)
        assertEquals(2L, native.request!!.tokenCount)
        assertEquals(2L, result.kvTokenOffsetAfter)
        assertEquals(7L, result.sampledTokenId)
    }

    @Test
    fun mapsTokenPrefillAssignmentToNativeRequest() {
        val request = WorkerAssignmentAdapter.toExecutionRequest(assignment(
            layerStart = 0,
            phase = WorkerJobAssignment.PREFILL,
            tokenCount = 2,
            inputTokenIds = listOf(17, 42)
        ))

        assertEquals(ExecutionMode.PREFILL, request.mode)
        assertEquals(0L, request.passOrdinal)
        assertEquals(2L, request.tokenCount)
        assertEquals(listOf(17L, 42L), (request.input as ExecutionInput.TokenIDs).values)
    }

    @Test
    fun mapsActivationDecodeAndValidatesNativeActivationResult() {
        val payload = ByteArray(8)
        val assignment = assignment(
            layerStart = 12,
            phase = WorkerJobAssignment.DECODE,
            passOrdinal = 1,
            kvTokenOffset = 2,
            tokenCount = 1,
            activation = Activation.create(
                jobId = "job-1",
                requestId = "request-1",
                sequenceId = "sequence-1",
                modelId = "qwen2.5-3b",
                modelVersion = "1",
                sourceWorker = "worker-a",
                destinationWorker = "worker-b",
                layer = 11,
                dtype = "F32",
                shape = listOf(1, 2),
                payload = payload
            ).copy(
                position = 1,
                passOrdinal = 1,
                kvTokenOffset = 2,
                tokenCount = 1,
                modelArtifactDigest = MODEL_DIGEST,
                layout = "ROW_MAJOR_CONTIGUOUS",
                byteOrder = "LITTLE_ENDIAN",
                byteLength = payload.size.toLong()
            )
        )

        assertNull(assignment.validate())
        val request = WorkerAssignmentAdapter.toExecutionRequest(assignment)
        assertEquals(ExecutionMode.DECODE, request.mode)
        assertTrue(request.input is ExecutionInput.Activation)

        val result = NativeShardExecutionResult(
            requestId = request.requestId,
            sequenceId = request.sequenceId,
            passOrdinal = request.passOrdinal,
            kvTokenOffsetBefore = request.kvTokenOffset,
            kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
            outputType = ShardOutputType.ACTIVATION,
            payload = payload,
            tensorSpec = CanonicalTensorSpec(
                dtype = TensorDType.F32,
                shape = listOf(1, 2),
                byteLength = payload.size.toLong()
            )
        )
        assertNull(WorkerAssignmentAdapter.validateResult(request, result, finalShard = false))
        assertTrue(
            WorkerAssignmentAdapter.validateResult(
                request,
                result.copy(kvTokenOffsetAfter = 4),
                finalShard = false
            )!!.contains("KV token range")
        )
    }

    private fun assignment(
        layerStart: Int,
        phase: String,
        tokenCount: Long,
        passOrdinal: Long = 0,
        kvTokenOffset: Long = 0,
        inputTokenIds: List<Long>? = null,
        activation: Activation? = null
    ) = WorkerJobAssignment(
        requestId = "request-1",
        modelVersion = "1",
        modelArtifactDigest = MODEL_DIGEST,
        workerId = "worker-b",
        previousWorker = if (layerStart == 0) null else "worker-a",
        nextWorker = "worker-c",
        assignmentId = "assignment-1",
        jobId = "job-1",
        modelId = "qwen2.5-3b",
        shardId = "layers-$layerStart",
        phase = phase,
        sequenceId = "sequence-1",
        position = passOrdinal.toInt(),
        passOrdinal = passOrdinal,
        kvTokenOffset = kvTokenOffset,
        tokenCount = tokenCount,
        inputTokenIds = inputTokenIds,
        activation = activation,
        layerStart = layerStart,
        layerEnd = layerStart + 11,
        sequence = passOrdinal
    )

    companion object {
        private const val MODEL_DIGEST =
            "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
    }

    private class RecordingPromptExecutor : NativeShardExecutor, NativePromptTokenizer {
        var prompt: String? = null
        var request: ShardExecutionRequest? = null

        override suspend fun tokenize(prompt: String): List<Long> {
            this.prompt = prompt
            return listOf(17L, 42L)
        }

        override suspend fun execute(request: ShardExecutionRequest): NativeShardExecutionResult {
            this.request = request
            return NativeShardExecutionResult(
                requestId = request.requestId,
                sequenceId = request.sequenceId,
                passOrdinal = request.passOrdinal,
                kvTokenOffsetBefore = request.kvTokenOffset,
                kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
                outputType = ShardOutputType.TOKEN,
                tokenId = 7L
            )
        }

        override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) = Unit
    }
}
