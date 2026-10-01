package com.intellihive.worker.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.coroutines.runBlocking

class NativeShardBenchmarkEngineTest {
    @Test
    fun prefillAndDecodeUseContiguousKvOffsets() {
        val passes = benchmarkExecutionPasses(prefillTokens = 37, generatedTokens = 2)

        assertEquals(
            listOf(
                BenchmarkExecutionPass(0, ExecutionMode.PREFILL, 0, 37),
                BenchmarkExecutionPass(1, ExecutionMode.DECODE, 37, 1),
                BenchmarkExecutionPass(2, ExecutionMode.DECODE, 38, 1)
            ),
            passes
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun benchmarkRejectsEmptyPrefill() {
        benchmarkExecutionPasses(prefillTokens = 0, generatedTokens = 1)
    }

    @Test
    fun benchmarkRunsEveryPassThroughAllThreeNativeShardContracts() = runBlocking {
        val executor = RecordingShardExecutor()
        val result = NativeShardBenchmarkEngine(executor).run(prefillTokens = 2, generatedTokens = 2)

        assertEquals(9, executor.requests.size)
        assertEquals(
            listOf(0, 10, 20, 0, 10, 20, 0, 10, 20),
            executor.requests.map { it.shard.layerStart }
        )
        assertEquals(
            listOf(0L, 0L, 0L, 2L, 2L, 2L, 3L, 3L, 3L),
            executor.requests.map { it.kvTokenOffset }
        )
        assertEquals(2, result.generatedTokens)
        assertTrue(executor.sequenceEnded)
        assertTrue(executor.completed)
    }

    @Test
    fun failedBenchmarkEndsNativeSequenceAsIncomplete() = runBlocking {
        val executor = RecordingShardExecutor(failAtLayerStart = 10)
        try {
            NativeShardBenchmarkEngine(executor).run(prefillTokens = 1, generatedTokens = 1)
        } catch (_: IllegalStateException) {
            assertTrue(executor.sequenceEnded)
            assertFalse(executor.completed)
            return@runBlocking
        }
        throw AssertionError("expected intermediate shard failure")
    }

    private class RecordingShardExecutor(
        private val failAtLayerStart: Int? = null
    ) : NativeShardExecutor {
        val requests = mutableListOf<ShardExecutionRequest>()
        var sequenceEnded = false
        var completed = false

        override suspend fun execute(request: ShardExecutionRequest): NativeShardExecutionResult {
            requests += request
            if (request.shard.layerStart == failAtLayerStart) {
                throw IllegalStateException("synthetic shard failure")
            }
            return if (request.shard.layerEnd == 35) {
                NativeShardExecutionResult(
                    requestId = request.requestId,
                    sequenceId = request.sequenceId,
                    passOrdinal = request.passOrdinal,
                    kvTokenOffsetBefore = request.kvTokenOffset,
                    kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
                    outputType = ShardOutputType.TOKEN,
                    tokenId = 1L
                )
            } else {
                val payload = ByteArray(Math.toIntExact(request.tokenCount * 2048L * Float.SIZE_BYTES))
                NativeShardExecutionResult(
                    requestId = request.requestId,
                    sequenceId = request.sequenceId,
                    passOrdinal = request.passOrdinal,
                    kvTokenOffsetBefore = request.kvTokenOffset,
                    kvTokenOffsetAfter = request.kvTokenOffset + request.tokenCount,
                    outputType = ShardOutputType.ACTIVATION,
                    payload = payload,
                    tensorSpec = CanonicalTensorSpec(
                        dtype = TensorDType.F32,
                        shape = listOf(request.tokenCount, 2048L),
                        byteLength = payload.size.toLong()
                    )
                )
            }
        }

        override suspend fun endSequence(jobId: String, sequenceId: String, completed: Boolean) {
            sequenceEnded = true
            this.completed = completed
        }
    }
}
