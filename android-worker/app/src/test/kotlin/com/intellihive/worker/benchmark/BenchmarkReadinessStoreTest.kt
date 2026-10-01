package com.intellihive.worker.benchmark

import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkReadinessStoreTest {
    @Test
    fun completedBenchmarkIsPersistedAndReadBack() {
        val directory = Files.createTempDirectory("intelhive-benchmark").toFile()
        try {
            val store = BenchmarkReadinessStore(directory.resolve(BenchmarkReadinessStore.FILE_NAME))
            val completed = BenchmarkReadiness.completed(
                prefillTokens = 32,
                generatedTokens = 8,
                totalTimeMs = 1200,
                tokensPerSecond = 33.3,
                prefillSpeedTokensPerSecond = 40.0,
                generationSpeedTokensPerSecond = 25.0,
                layerRanges = com.intellihive.worker.inference.QwenShardCatalog.ranges
            )

            store.write(completed)

            assertEquals(BenchmarkStatus.COMPLETED, store.read().status)
            assertEquals(
                BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS,
                store.read().executionMode
            )
            assertEquals(completed.tokensPerSecond, store.read().tokensPerSecond)
            assertTrue(directory.resolve(BenchmarkReadinessStore.FILE_NAME).isFile)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun legacyCompletedBenchmarkIsMigratedToSingleDeviceExecutionMode() {
        val directory = Files.createTempDirectory("intelhive-benchmark-legacy").toFile()
        try {
            val resultFile = directory.resolve(BenchmarkReadinessStore.FILE_NAME)
            resultFile.writeText(
                """{"status":"COMPLETED","modelId":"qwen2.5-3b-instruct","modelVersion":"1.0.0","modelArtifactDigest":"sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d","tokensPerSecond":1.0,"layerRanges":[{"shardId":"s0","layerStart":0,"layerEnd":9},{"shardId":"s1","layerStart":10,"layerEnd":19},{"shardId":"s2","layerStart":20,"layerEnd":35}]}"""
            )

            val migrated = BenchmarkReadinessStore(resultFile).read()

            assertEquals(
                BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS,
                migrated.executionMode
            )
            assertTrue(resultFile.readText().contains("\"execution_mode\":\"single_device_all_shards\""))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun missingBenchmarkReturnsNotRun() {
        val directory = Files.createTempDirectory("intelhive-benchmark").toFile()
        try {
            assertEquals(
                BenchmarkStatus.NOT_RUN,
                BenchmarkReadinessStore(directory.resolve(BenchmarkReadinessStore.FILE_NAME))
                    .read().status
            )
        } finally {
            directory.deleteRecursively()
        }
    }
}
