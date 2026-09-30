package com.intellihive.worker.inference

import com.intellihive.worker.service.ShardExecutionResult
import com.intellihive.worker.service.WorkerJobAssignment
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class IntelHiveExecutionWrapperTest {
    @Test
    fun forwardsPrefillThenOrderedDecodeToShardBackend() = runBlocking {
        val backend = RecordingBackend()
        val wrapper = IntelHiveExecutionWrapper(backend)

        val prefill = wrapper.execute(assignment(
            phase = WorkerJobAssignment.PREFILL,
            position = 0,
            prompt = "hello",
            finalShard = false
        ))
        assertTrue(prefill.activation!!.payload.contentEquals(byteArrayOf(10)))

        val decode = wrapper.execute(assignment(
            phase = WorkerJobAssignment.DECODE,
            position = 1,
            inputTokenIds = listOf(42),
            finalShard = false
        ))
        assertTrue(decode.activation!!.payload.contentEquals(byteArrayOf(11)))

        assertEquals(listOf(0, 1), backend.invocations.map { it.position })
        assertNull(backend.invocations[0].prompt)
        assertEquals(listOf(1L), backend.invocations[0].inputTokenIds)
        assertEquals(listOf(42L), backend.invocations[1].inputTokenIds)
        assertNull(backend.invocations[0].activation)
        assertEquals("sequence-1", backend.invocations[1].sequenceId)
    }

    @Test
    fun rejectsDecodeBeforePrefillAndOutOfOrderPositions() = runBlocking {
        val wrapper = IntelHiveExecutionWrapper(RecordingBackend())

        val beforePrefill = runCatching {
            wrapper.execute(assignment(
                phase = WorkerJobAssignment.DECODE,
                position = 1,
                inputTokenIds = listOf(42),
                finalShard = false
            ))
        }.exceptionOrNull()
        assertTrue(beforePrefill is IllegalStateException)

        wrapper.execute(assignment(
            phase = WorkerJobAssignment.PREFILL,
            position = 0,
            prompt = "hello",
            finalShard = false
        ))
        val outOfOrder = runCatching {
            wrapper.execute(assignment(
                phase = WorkerJobAssignment.DECODE,
                position = 2,
                inputTokenIds = listOf(42),
                finalShard = false
            ))
        }.exceptionOrNull()
        assertTrue(outOfOrder is IllegalArgumentException)
    }

    @Test
    fun finalShardReturnsTokenAndSequenceEndReleasesBackendState() = runBlocking {
        val backend = RecordingBackend()
        val wrapper = IntelHiveExecutionWrapper(backend)

        val result = wrapper.execute(assignment(
            phase = WorkerJobAssignment.PREFILL,
            position = 0,
            prompt = "hello",
            finalShard = true
        ))
        assertEquals(7L, result.sampledTokenId)

        wrapper.endSequence("job-1", "sequence-1", completed = true)

        assertEquals(listOf(Triple("job-1", "sequence-1", true)), backend.endedSequences)
    }

    @Test
    fun prefillFailureDoesNotLeaveSequenceReserved() = runBlocking {
        val backend = RecordingBackend(failFirst = true)
        val wrapper = IntelHiveExecutionWrapper(backend)
        val prefill = assignment(
            phase = WorkerJobAssignment.PREFILL,
            position = 0,
            prompt = "hello",
            finalShard = false
        )

        assertTrue(runCatching { wrapper.execute(prefill) }.isFailure)
        assertEquals(listOf(Triple("job-1", "sequence-1", false)), backend.endedSequences)
        assertNull(wrapper.execute(prefill).sampledTokenId)
        assertEquals(2, backend.invocations.size)
    }

    private fun assignment(
        phase: String,
        position: Int,
        prompt: String? = null,
        inputTokenIds: List<Long>? = null,
        activation: Activation? = null,
        finalShard: Boolean
    ) = WorkerJobAssignment(
        requestId = "job-1", modelVersion = "v1",
        modelArtifactDigest = "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
        workerId = "worker-a", nextWorker = "worker-b",
        assignmentId = "assignment-$position",
        jobId = "job-1",
        modelId = "qwen2.5-3b-instruct",
        shardId = "shard-0-9",
        phase = phase,
        sequenceId = "sequence-1",
        position = position,
        passOrdinal = if (phase == WorkerJobAssignment.PREFILL) 0 else position.toLong(),
        kvTokenOffset = position.toLong(),
        tokenCount = inputTokenIds?.size?.toLong() ?: 1,
        inputTokenIds = inputTokenIds ?: if (prompt != null) listOf(1L) else null,
        activation = activation,
        finalShard = finalShard,
        layerStart = 0,
        layerEnd = 9,
        sequence = position.toLong()
    )

    private class RecordingBackend(
        private var failFirst: Boolean = false
    ) : LlamaCppLayerBackend {
        val invocations = mutableListOf<LayerShardInvocation>()
        val endedSequences = mutableListOf<Triple<String, String, Boolean>>()

        override suspend fun execute(invocation: LayerShardInvocation): ShardExecutionResult {
            invocations += invocation
            if (failFirst) {
                failFirst = false
                error("injected backend failure")
            }
            return if (invocation.finalShard) {
                ShardExecutionResult(sampledTokenId = 7)
            } else {
                ShardExecutionResult(activation = Activation.create(
                    invocation.jobId, invocation.requestId, invocation.sequenceId, invocation.modelId,
                    invocation.modelVersion, invocation.workerId, invocation.nextWorker!!,
                    invocation.layerEnd, "uint8", listOf(1),
                    byteArrayOf((invocation.position + 10).toByte())
                ).copy(position = invocation.position))
            }
        }

        override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) {
            endedSequences += Triple(jobId, sequenceId, completed)
        }
    }
}
