package com.intellihive.worker.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.coroutines.delay
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

    @Test
    fun prefillCanBeSplitIntoBoundedChunksWithoutChangingKvOffsets() {
        val passes = benchmarkExecutionPasses(
            prefillTokens = 10,
            generatedTokens = 2,
            prefillChunkTokens = 4
        )

        assertEquals(
            listOf(
                BenchmarkExecutionPass(0, ExecutionMode.PREFILL, 0, 4),
                BenchmarkExecutionPass(1, ExecutionMode.PREFILL, 4, 4),
                BenchmarkExecutionPass(2, ExecutionMode.PREFILL, 8, 2),
                BenchmarkExecutionPass(3, ExecutionMode.DECODE, 10, 1),
                BenchmarkExecutionPass(4, ExecutionMode.DECODE, 11, 1)
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

        assertEquals(12, executor.requests.size)
        assertEquals(
            listOf(0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20),
            executor.requests.map { it.shard.layerStart }
        )
        assertEquals(
            listOf(0L, 0L, 0L, 1L, 1L, 1L, 2L, 2L, 2L, 3L, 3L, 3L),
            executor.requests.map { it.kvTokenOffset }
        )
        assertEquals(2, result.generatedTokens)
        assertEquals(2, result.decodePassLatenciesNanos.size)
        assertTrue(result.decodePassLatenciesNanos.all { it > 0L })
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

    @Test
    fun sustainedRunStopsAfterCompletedDecodePasses() = runBlocking {
        val executor = RecordingShardExecutor(delayMillis = 10)

        val result = NativeShardBenchmarkEngine(executor).run(
            prefillTokens = 1,
            generatedTokens = 4,
            stopAfterMs = 45
        )

        assertEquals(1, result.generatedTokens)
        assertEquals(6, executor.requests.size)
        assertTrue(result.safetyLimitReached)
        assertTrue(executor.sequenceEnded)
        assertTrue(executor.completed)
    }

    @Test
    fun safetyLimitIncludesPrefillAndStopsBeforeDecode() = runBlocking {
        val executor = RecordingShardExecutor(delayMillis = 10)

        val result = NativeShardBenchmarkEngine(executor).run(
            prefillTokens = 8,
            generatedTokens = 4,
            stopAfterMs = 1
        )

        assertEquals(0, result.generatedTokens)
        assertEquals(3, executor.requests.size)
        assertTrue(result.safetyLimitReached)
        assertTrue(executor.sequenceEnded)
    }

    @Test
    fun nativeDeadlineIsInstalledForRunAndClearedAfterSequenceCleanup() = runBlocking {
        val executor = RecordingShardExecutor()

        NativeShardBenchmarkEngine(executor).run(
            prefillTokens = 1,
            generatedTokens = 1,
            stopAfterMs = 1_000
        )

        assertEquals(2, executor.deadlines.size)
        assertTrue(checkNotNull(executor.deadlines.first()) > 0L)
        assertEquals(null, executor.deadlines.last())
    }

    private class RecordingShardExecutor(
        private val failAtLayerStart: Int? = null,
        private val delayMillis: Long = 0L
    ) : NativeShardExecutor, NativeExecutionDeadlineController {
        val requests = mutableListOf<ShardExecutionRequest>()
        val deadlines = mutableListOf<Long?>()
        var sequenceEnded = false
        var completed = false

        override fun setExecutionDeadlineNanos(deadlineNanos: Long?) {
            deadlines += deadlineNanos
        }

        override suspend fun execute(request: ShardExecutionRequest): NativeShardExecutionResult {
            if (delayMillis > 0L) delay(delayMillis)
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
