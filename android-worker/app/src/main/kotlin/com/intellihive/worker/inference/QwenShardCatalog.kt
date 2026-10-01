package com.intellihive.worker.inference

import com.intellihive.worker.model.ModelManifest

data class ModelShardRange(
    val shardId: String,
    val layerStart: Int,
    val layerEnd: Int
) {
    fun executionSpec() = ExecutionShardSpec(
        id = shardId,
        modelId = ModelManifest.PINNED_MODEL_ID,
        modelVersion = ModelManifest.PINNED_MODEL_VERSION,
        modelArtifactDigest = ModelManifest.PINNED_ARTIFACT_DIGEST,
        layerStart = layerStart,
        layerEnd = layerEnd
    )
}

object QwenShardCatalog {
    val ranges = listOf(
        ModelShardRange("s0", 0, 9),
        ModelShardRange("s1", 10, 19),
        ModelShardRange("s2", 20, 35)
    )
}
