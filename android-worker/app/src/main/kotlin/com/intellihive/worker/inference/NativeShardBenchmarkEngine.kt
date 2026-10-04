package com.intellihive.worker.inference

import java.util.UUID

private const val MAX_PREFILL_TOKENS = 4096
private const val MAX_GENERATED_TOKENS = 4096

data class BenchmarkExecutionPass(
    val passOrdinal: Int,
    val mode: ExecutionMode,
    val kvTokenOffset: Int,
    val tokenCount: Int
)

fun benchmarkExecutionPasses(
    prefillTokens: Int,
    generatedTokens: Int,
    prefillChunkTokens: Int = prefillTokens
): List<BenchmarkExecutionPass> {
    require(prefillTokens > 0) { "prefillTokens must be greater than zero" }
    require(generatedTokens > 0) { "generatedTokens must be greater than zero" }
    require(prefillChunkTokens > 0) { "prefillChunkTokens must be greater than zero" }
    require(prefillTokens <= MAX_PREFILL_TOKENS) {
        "prefillTokens must not exceed $MAX_PREFILL_TOKENS"
    }
    require(generatedTokens <= MAX_GENERATED_TOKENS) {
        "generatedTokens must not exceed $MAX_GENERATED_TOKENS"
    }
    Math.addExact(prefillTokens, generatedTokens)
    val prefillPassCount = (prefillTokens + prefillChunkTokens - 1) / prefillChunkTokens
    return buildList(generatedTokens + prefillPassCount) {
        var prefillOffset = 0
        while (prefillOffset < prefillTokens) {
            val chunkSize = minOf(prefillChunkTokens, prefillTokens - prefillOffset)
            add(BenchmarkExecutionPass(size, ExecutionMode.PREFILL, prefillOffset, chunkSize))
            prefillOffset += chunkSize
        }
        repeat(generatedTokens) { index ->
            add(
                BenchmarkExecutionPass(
                    passOrdinal = size,
                    mode = ExecutionMode.DECODE,
                    kvTokenOffset = Math.addExact(prefillTokens, index),
                    tokenCount = 1
                )
            )
        }
    }
}

data class NativeShardBenchmarkResult(
    val prefillTokens: Int,
    val generatedTokens: Int,
    val totalTimeMs: Long,
    val tokensPerSecond: Double,
    val prefillSpeedTokensPerSecond: Double,
    val generationSpeedTokensPerSecond: Double,
    val decodePassLatenciesNanos: List<Long>,
    val layerRanges: List<ModelShardRange>,
    val safetyLimitReached: Boolean
)

/** Optional native control used by benchmarks without leaking deadlines into the execution DTO. */
interface NativeExecutionDeadlineController {
    fun setExecutionDeadlineNanos(deadlineNanos: Long?)
}

class NativeShardBenchmarkEngine(
    private val executor: NativeShardExecutor,
    private val ranges: List<ModelShardRange> = QwenShardCatalog.ranges
) {
    suspend fun run(
        prefillTokens: Int,
        generatedTokens: Int,
        stopAfterMs: Long? = null,
        shouldStop: () -> Boolean = { false },
        onPassCompleted: (completed: Int, total: Int) -> Unit = { _, _ -> }
    ): NativeShardBenchmarkResult {
        require(stopAfterMs == null || stopAfterMs > 0L) {
            "stopAfterMs must be greater than zero"
        }
        require(ranges.isNotEmpty()) { "benchmark requires at least one model shard" }
        require(ranges.first().layerStart == 0) { "benchmark shard chain must start at layer zero" }
        ranges.zipWithNext().forEach { (previous, next) ->
            require(next.layerStart == previous.layerEnd + 1) {
                "benchmark shard ranges must be contiguous"
            }
        }
        require(ranges.last().layerEnd == 35) {
            "benchmark shard chain must cover all 36 Qwen2.5-3B layers"
        }

        val passes = benchmarkExecutionPasses(
            prefillTokens = prefillTokens,
            generatedTokens = generatedTokens,
            prefillChunkTokens = PREFILL_CHUNK_TOKENS
        )
        val sequenceId = "benchmark-${UUID.randomUUID()}"
        val allStart = System.nanoTime()
        val deadlineNanos = stopAfterMs?.let { durationMs ->
            Math.addExact(allStart, Math.multiplyExact(durationMs, NANOS_PER_MILLISECOND))
        }
        val deadlineController = executor as? NativeExecutionDeadlineController
        deadlineController?.setExecutionDeadlineNanos(deadlineNanos)
        var prefillElapsedNanos = 0L
        var generationElapsedNanos = 0L
        val decodePassLatenciesNanos = mutableListOf<Long>()
        var nextInputToken = BENCHMARK_TOKEN_ID
        var completedGeneratedTokens = 0
        var safetyLimitReached = false
        var completed = false
        try {
            for ((passIndex, pass) in passes.withIndex()) {
                if (stopAfterMs != null &&
                    (System.nanoTime() - allStart) / NANOS_PER_MILLISECOND >= stopAfterMs
                ) {
                    safetyLimitReached = true
                    break
                }
                val passStart = System.nanoTime()
                var input: ExecutionInput = if (pass.mode == ExecutionMode.PREFILL) {
                    ExecutionInput.TokenIDs(List(pass.tokenCount) { BENCHMARK_TOKEN_ID })
                } else {
                    ExecutionInput.TokenIDs(listOf(nextInputToken))
                }
                var finalResult: NativeShardExecutionResult? = null

                ranges.forEachIndexed { rangeIndex, range ->
                    val request = ShardExecutionRequest(
                        requestId = "$sequenceId-${pass.passOrdinal}-${range.shardId}",
                        sequenceId = sequenceId,
                        shard = range.executionSpec(),
                        passOrdinal = pass.passOrdinal.toLong(),
                        mode = pass.mode,
                        kvTokenOffset = pass.kvTokenOffset.toLong(),
                        tokenCount = pass.tokenCount.toLong(),
                        input = input
                    )
                    val result = executor.execute(request)
                    WorkerAssignmentAdapter.validateResult(
                        request,
                        result,
                        finalShard = rangeIndex == ranges.lastIndex
                    )?.let { error ->
                        throw IllegalStateException("Benchmark shard ${range.shardId}: $error")
                    }
                    if (rangeIndex == ranges.lastIndex) {
                        require(result.outputType == ShardOutputType.TOKEN && result.tokenId != null) {
                            "final benchmark shard must return a sampled token"
                        }
                        finalResult = result
                    } else {
                        require(result.outputType == ShardOutputType.ACTIVATION) {
                            "intermediate benchmark shard must return an activation"
                        }
                        input = ExecutionInput.Activation(
                            payload = checkNotNull(result.payload),
                            spec = checkNotNull(result.tensorSpec)
                        )
                    }
                }
                nextInputToken = checkNotNull(finalResult?.tokenId) {
                    "final benchmark shard did not return a token"
                }
                val passElapsed = System.nanoTime() - passStart
                if (pass.mode == ExecutionMode.PREFILL) {
                    prefillElapsedNanos += passElapsed
                } else {
                    generationElapsedNanos += passElapsed
                    completedGeneratedTokens++
                    decodePassLatenciesNanos += passElapsed
                }
                onPassCompleted(completedGeneratedTokens, passes.size)
                if (shouldStop()) break
                if (stopAfterMs != null &&
                    (System.nanoTime() - allStart) / NANOS_PER_MILLISECOND >= stopAfterMs &&
                    passIndex < passes.lastIndex
                ) {
                    safetyLimitReached = true
                    break
                }
            }
            completed = true
        } finally {
            try {
                executor.endSequence(sequenceId, sequenceId, completed)
            } finally {
                deadlineController?.setExecutionDeadlineNanos(null)
            }
        }

        val totalElapsedNanos = System.nanoTime() - allStart
        val totalElapsedMs = (totalElapsedNanos / NANOS_PER_MILLISECOND).coerceAtLeast(1)
        return NativeShardBenchmarkResult(
            prefillTokens = prefillTokens,
            generatedTokens = completedGeneratedTokens,
            totalTimeMs = totalElapsedMs,
            tokensPerSecond = rate(prefillTokens + completedGeneratedTokens, totalElapsedNanos),
            prefillSpeedTokensPerSecond = rate(prefillTokens, prefillElapsedNanos),
            generationSpeedTokensPerSecond = rate(completedGeneratedTokens, generationElapsedNanos),
            decodePassLatenciesNanos = decodePassLatenciesNanos,
            layerRanges = ranges,
            safetyLimitReached = safetyLimitReached
        )
    }

    private fun rate(tokens: Int, elapsedNanos: Long): Double =
        if (elapsedNanos > 0L) tokens * NANOS_PER_SECOND / elapsedNanos else 0.0

    companion object {
        private const val PREFILL_CHUNK_TOKENS = 1
        private const val BENCHMARK_TOKEN_ID = 1L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
