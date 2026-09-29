package com.intellihive.worker.inference

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeLayerRangeInstrumentedTest {
    @Test
    fun loadsPinnedArm64JniAndLlamaRuntime() {
        assertTrue(
            "JNI load probe failed: ${NativeRuntime.unavailableReason()}",
            NativeRuntime.isAvailable()
        )
    }

    @Test
    fun executesQwenPrefillAndDecodeAcrossThreeNativeShards() = runBlocking {
        assertTrue(
            "JNI load probe failed: ${NativeRuntime.unavailableReason()}",
            NativeRuntime.isAvailable()
        )
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelName = "qwen2.5-3b-instruct-q4_k_m.gguf"
        val cacheModel = File(context.cacheDir, "models/$modelName")
        val externalModel = context.getExternalFilesDir("Documents")
            ?.let { File(it, "models/$modelName") }
        val model = externalModel?.takeIf { it.isFile } ?: cacheModel
        assumeTrue("Install the verified Q4_K_M GGUF at ${model.absolutePath}", model.isFile)

        val executor = NativeLayerRangeShardExecutor(context, model)
        try {
            val first = shardRequest(0, 11, 0, ExecutionInput.TokenIDs(listOf(17, 42, 1337)))
            val firstResult = executor.execute(first)
            assertEquals(ShardOutputType.ACTIVATION, firstResult.outputType)
            assertNotNull(firstResult.payload)

            val second = shardRequest(
                12,
                23,
                0,
                ExecutionInput.Activation(
                    firstResult.payload!!,
                    checkNotNull(firstResult.tensorSpec)
                )
            )
            val secondResult = executor.execute(second)
            assertEquals(ShardOutputType.ACTIVATION, secondResult.outputType)

            val third = shardRequest(
                24,
                35,
                0,
                ExecutionInput.Activation(
                    checkNotNull(secondResult.payload),
                    checkNotNull(secondResult.tensorSpec)
                )
            )
            val prefillResult = executor.execute(third)
            assertEquals(ShardOutputType.TOKEN, prefillResult.outputType)
            assertNotNull(prefillResult.tokenId)

            val nextToken = checkNotNull(prefillResult.tokenId)
            assertTrue(nextToken in 0L until 151_936L)
            val decodeFirst = shardRequest(
                0,
                11,
                3,
                ExecutionInput.TokenIDs(listOf(nextToken)),
                passOrdinal = 1
            )
            val decodeFirstResult = executor.execute(decodeFirst)
            val decodeSecond = shardRequest(
                12,
                23,
                3,
                ExecutionInput.Activation(
                    checkNotNull(decodeFirstResult.payload),
                    checkNotNull(decodeFirstResult.tensorSpec)
                ),
                passOrdinal = 1
            )
            val decodeSecondResult = executor.execute(decodeSecond)
            val decodeThird = shardRequest(
                24,
                35,
                3,
                ExecutionInput.Activation(
                    checkNotNull(decodeSecondResult.payload),
                    checkNotNull(decodeSecondResult.tensorSpec)
                ),
                passOrdinal = 1
            )
            val decodeResult = executor.execute(decodeThird)
            assertEquals(ShardOutputType.TOKEN, decodeResult.outputType)
            assertTrue(decodeResult.tokenId!! in 0L until 151_936L)
            assertEquals(4L, decodeResult.kvTokenOffsetAfter)
        } finally {
            executor.endSequence("android-instrumentation", "android-native-sequence", completed = true)
            executor.close()
        }
    }

    private fun shardRequest(
        firstLayer: Int,
        lastLayer: Int,
        offset: Long,
        input: ExecutionInput,
        passOrdinal: Long = 0
    ) = ShardExecutionRequest(
        requestId = "android-instrumentation",
        sequenceId = "android-native-sequence",
        shard = ExecutionShardSpec(
            id = "layers-$firstLayer-$lastLayer",
            modelId = "qwen2.5-3b",
            modelVersion = "1.0.0",
            modelArtifactDigest =
                "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
            layerStart = firstLayer,
            layerEnd = lastLayer
        ),
        passOrdinal = passOrdinal,
        mode = if (passOrdinal == 0L) ExecutionMode.PREFILL else ExecutionMode.DECODE,
        kvTokenOffset = offset,
        tokenCount = when (input) {
            is ExecutionInput.TokenIDs -> input.values.size.toLong()
            is ExecutionInput.Activation -> input.spec.shape.first()
        },
        input = input
    )
}
