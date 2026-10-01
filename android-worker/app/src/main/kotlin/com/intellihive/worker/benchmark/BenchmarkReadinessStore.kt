package com.intellihive.worker.benchmark

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.intellihive.worker.inference.ModelShardRange
import com.intellihive.worker.model.ModelManifest
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.time.Instant

enum class BenchmarkStatus { NOT_RUN, RUNNING, COMPLETED, FAILED }

enum class BenchmarkExecutionMode(val wireValue: String) {
    @SerializedName("single_device_all_shards")
    SINGLE_DEVICE_ALL_SHARDS("single_device_all_shards"),

    @SerializedName("distributed_pipeline")
    DISTRIBUTED_PIPELINE("distributed_pipeline")
}

data class BenchmarkReadiness(
    val status: BenchmarkStatus,
    @SerializedName("execution_mode")
    val executionMode: BenchmarkExecutionMode? = null,
    val modelId: String = ModelManifest.PINNED_MODEL_ID,
    val modelVersion: String = ModelManifest.PINNED_MODEL_VERSION,
    val modelArtifactDigest: String = ModelManifest.PINNED_ARTIFACT_DIGEST,
    val prefillTokens: Int? = null,
    val generatedTokens: Int? = null,
    val totalTimeMs: Long? = null,
    val tokensPerSecond: Double? = null,
    val prefillSpeedTokensPerSecond: Double? = null,
    val generationSpeedTokensPerSecond: Double? = null,
    val layerRanges: List<ModelShardRange> = emptyList(),
    val completedAt: String? = null,
    val errorMessage: String? = null
) {
    fun validate(): String? = when {
        modelId != ModelManifest.PINNED_MODEL_ID ||
            modelVersion != ModelManifest.PINNED_MODEL_VERSION ||
            modelArtifactDigest != ModelManifest.PINNED_ARTIFACT_DIGEST ->
            "benchmark result model metadata does not match the pinned model"
        status == BenchmarkStatus.COMPLETED && executionMode == null ->
            "completed benchmark must identify its execution mode"
        status == BenchmarkStatus.COMPLETED &&
            (tokensPerSecond == null || !tokensPerSecond.isFinite() || tokensPerSecond <= 0.0) ->
            "completed benchmark must contain a positive finite tokens_per_second value"
        status == BenchmarkStatus.COMPLETED && layerRanges != com.intellihive.worker.inference.QwenShardCatalog.ranges ->
            "completed benchmark must cover the configured three-shard model topology"
        status == BenchmarkStatus.FAILED && errorMessage.isNullOrBlank() ->
            "failed benchmark must contain an error message"
        else -> null
    }

    companion object {
        fun running() = BenchmarkReadiness(
            status = BenchmarkStatus.RUNNING,
            executionMode = BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS
        )

        fun completed(
            prefillTokens: Int,
            generatedTokens: Int,
            totalTimeMs: Long,
            tokensPerSecond: Double,
            prefillSpeedTokensPerSecond: Double,
            generationSpeedTokensPerSecond: Double,
            layerRanges: List<ModelShardRange>
        ) = BenchmarkReadiness(
            status = BenchmarkStatus.COMPLETED,
            executionMode = BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS,
            prefillTokens = prefillTokens,
            generatedTokens = generatedTokens,
            totalTimeMs = totalTimeMs,
            tokensPerSecond = tokensPerSecond,
            prefillSpeedTokensPerSecond = prefillSpeedTokensPerSecond,
            generationSpeedTokensPerSecond = generationSpeedTokensPerSecond,
            layerRanges = layerRanges,
            completedAt = Instant.now().toString()
        )

        fun failed(errorMessage: String) = BenchmarkReadiness(
            status = BenchmarkStatus.FAILED,
            executionMode = BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS,
            errorMessage = errorMessage
        )
    }
}

class BenchmarkReadinessStore(private val resultFile: File) {
    private val gson = Gson()

    fun read(): BenchmarkReadiness {
        if (!resultFile.exists()) return BenchmarkReadiness(BenchmarkStatus.NOT_RUN)
        val record = try {
            gson.fromJson(resultFile.readText(), BenchmarkReadiness::class.java)
                ?: throw IllegalStateException("benchmark result file is empty")
        } catch (error: Exception) {
            throw IllegalStateException("Could not read persisted benchmark result: ${error.message}", error)
        }
        val migrated = if (record.status == BenchmarkStatus.COMPLETED && record.executionMode == null) {
            record.copy(executionMode = BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS)
        } else {
            record
        }
        migrated.validate()?.let { throw IllegalStateException("Invalid persisted benchmark result: $it") }
        if (migrated != record) write(migrated)
        return migrated
    }

    fun write(record: BenchmarkReadiness) {
        record.validate()?.let { throw IllegalArgumentException(it) }
        val parent = resultFile.parentFile
            ?: throw IllegalArgumentException("benchmark result file must have a parent directory")
        check(parent.isDirectory || parent.mkdirs()) {
            "Could not create benchmark result directory ${parent.absolutePath}"
        }
        val temporary = File(parent, "${resultFile.name}.tmp")
        try {
            FileOutputStream(temporary).use { output ->
                output.write(gson.toJson(record).toByteArray(Charsets.UTF_8))
                output.fd.sync()
            }
            Files.move(
                temporary.toPath(),
                resultFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            )
        } catch (error: Exception) {
            temporary.delete()
            throw IllegalStateException("Could not persist benchmark result: ${error.message}", error)
        }
    }

    companion object {
        const val FILE_NAME = "benchmark_result.json"
    }
}
