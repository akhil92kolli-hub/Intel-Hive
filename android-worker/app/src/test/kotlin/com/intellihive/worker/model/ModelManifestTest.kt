package com.intellihive.worker.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelManifestTest {
    @Test
    fun parsesAndValidatesPinnedQwenManifest() {
        val manifest = ModelManifest.fromJson(manifestJson())

        assertEquals("qwen2.5-3b-instruct", manifest.modelId)
        assertEquals("1.0.0", manifest.version)
        assertEquals(2_104_932_768L, manifest.sizeBytes)
        assertEquals(
            "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
            manifest.sha256
        )
        assertEquals(null, manifest.validate())
    }

    @Test
    fun rejectsManifestWithNonTlsDownloadUrl() {
        val error = runCatching {
            ModelManifest.fromJson(manifestJson().replace("https://", "http://"))
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
        assertTrue(error?.message.orEmpty().contains("HTTPS"))
    }

    @Test
    fun rejectsShardMetadataThatDoesNotMatchArtifact() {
        val error = runCatching {
            ModelManifest.fromJson(manifestJson().replace(
                "\"size_bytes\":2104932768",
                "\"size_bytes\":2104932767"
            ))
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
        assertTrue(error?.message.orEmpty().contains("digest and size"))
    }

    @Test
    fun rejectsArtifactDigestNotApprovedForNativeExecutor() {
        val differentDigest = "sha256:" + "0".repeat(64)
        val error = runCatching {
            ModelManifest.fromJson(manifestJson().replace(MODEL_DIGEST, differentDigest))
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
        assertTrue(error?.message.orEmpty().contains("pinned native model"))
    }

    private fun manifestJson(): String = """
        {
          "model_id":"qwen2.5-3b-instruct",
          "version":"1.0.0",
          "architecture":"qwen2",
          "layers":36,
          "hidden_size":2048,
          "artifact_digest":"$MODEL_DIGEST",
          "total_size_bytes":2104932768,
          "shards":[{
            "download_url":"https://models.example.test/qwen.gguf",
            "sha256":"$MODEL_DIGEST",
            "size_bytes":2104932768
          }]
        }
    """.trimIndent()

    companion object {
        private const val MODEL_DIGEST =
            "sha256:626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
    }
}
