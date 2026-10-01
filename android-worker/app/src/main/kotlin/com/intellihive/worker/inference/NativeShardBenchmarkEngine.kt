package com.intellihive.worker.inference

import java.util.UUID

private const val MAX_PREFILL_TOKENS = 4096
private const val MAX_GENERATED_TOKENS = 256

data class BenchmarkExecutionPass(
    val passOrdinal: Int,
    val mode: ExecutionMode,
    val kvTokenOffset: Int,
    val tokenCount: Int
)

fun benchmarkExecutionPasses(
    prefillTokens: Int,
    generatedTokens: Int
): List<BenchmarkExecutionPass> {
    require(prefillTokens > 0) { "prefillTokens must be greater than zero" }
    require(generatedTokens > 0) { "generatedTokens must be greater than zero" }
    require(prefillTokens <= MAX_PREFILL_TOKENS) {
        "prefillTokens must not exceed $MAX_PREFILL_TOKENS"
    }
    require(generatedTokens <= MAX_GENERATED_TOKENS) {
        "generatedTokens must not exceed $MAX_GENERATED_TOKENS"
    }
    Math.addExact(prefillTokens, generatedTokens)
    return buildList(generatedTokens + 1) {
        add(BenchmarkExecutionPass(0, ExecutionMode.PREFILL, 0, prefillTokens))
        repeat(generatedTokens) { index ->
            add(
                BenchmarkExecutionPass(
                    passOrdinal = index + 1,
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
    val layerRanges: List<ModelShardRange>
)

class NativeShardBenchmarkEngine(
    private val executor: NativeShardExecutor,
    private val ranges: List<ModelShardRange> = QwenShardCatalog.ranges
) {
    suspend fun run(
        prefillTokens: Int,
        generatedTokens: Int,
        onPassCompleted: (completed: Int, total: Int) -> Unit = { _, _ -> }
    ): NativeShardBenchmarkResult {
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

        val passes = benchmarkExecutionPasses(prefillTokens, generatedTokens)
        val sequenceId = "benchmark-${UUID.randomUUID()}"
        val allStart = System.nanoTime()
        var prefillElapsedNanos = 0L
        var generationElapsedNanos = 0L
        var nextInputToken = BENCHMARK_TOKEN_ID
        var completed = false
        try {
            passes.forEachIndexed { passIndex, pass ->
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
                    prefillElapsedNanos = passElapsed
                } else {
                    generationElapsedNanos += passElapsed
                }
                onPassCompleted(passIndex + 1, passes.size)
            }
            completed = true
        } finally {
            executor.endSequence(sequenceId, sequenceId, completed)
        }

        val totalElapsedNanos = System.nanoTime() - allStart
        val totalElapsedMs = (totalElapsedNanos / NANOS_PER_MILLISECOND).coerceAtLeast(1)
        return NativeShardBenchmarkResult(
            prefillTokens = prefillTokens,
            generatedTokens = generatedTokens,
            totalTimeMs = totalElapsedMs,
            tokensPerSecond = rate(prefillTokens + generatedTokens, totalElapsedNanos),
            prefillSpeedTokensPerSecond = rate(prefillTokens, prefillElapsedNanos),
            generationSpeedTokensPerSecond = rate(generatedTokens, generationElapsedNanos),
            layerRanges = ranges
        )
    }

    private fun rate(tokens: Int, elapsedNanos: Long): Double =
        if (elapsedNanos > 0L) tokens * NANOS_PER_SECOND / elapsedNanos else 0.0

    companion object {
        private const val BENCHMARK_TOKEN_ID = 1L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
