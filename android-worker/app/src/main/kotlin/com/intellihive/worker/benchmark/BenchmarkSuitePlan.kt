package com.intellihive.worker.benchmark

import com.google.gson.Gson
import com.intellihive.worker.inference.ShardPlacementProfile
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.time.Instant
import java.util.UUID

enum class BenchmarkStepKind {
    LLAMA_CPP_BASELINE,
    CPU_ONLY,
    VULKAN_ONLY,
    HYBRID
}

data class BenchmarkStepDefinition(
    val id: String,
    val kind: BenchmarkStepKind,
    val profile: ShardPlacementProfile,
    val measurementPhase: String,
    val durationMs: Long,
    val generatedTokens: Int = MAX_GENERATED_TOKENS,
    val prefillTokens: Int = DEFAULT_PREFILL_TOKENS,
    val graphThreads: Int = 0
) {
    init {
        require(id.isNotBlank())
        require(measurementPhase in setOf("comparison", "screening", "sustained"))
        require(durationMs > 0)
        require(generatedTokens > 0)
        require(prefillTokens > 0)
        require(graphThreads in 0..8)
    }
}

fun ShardPlacementProfile.displayName(): String = when (this) {
    ShardPlacementProfile.CPU_ONLY -> "CPU only"
    ShardPlacementProfile.VULKAN_S0 -> "Vulkan s0 + CPU s1/s2"
    ShardPlacementProfile.VULKAN_S0_S1 -> "Vulkan s0/s1 + CPU s2"
    ShardPlacementProfile.VULKAN_ALL -> "Vulkan all shards"
    ShardPlacementProfile.VULKAN_S2 -> "CPU s0/s1 + Vulkan s2"
    ShardPlacementProfile.VULKAN_S1_S2 -> "CPU s0 + Vulkan s1/s2"
}

val ShardPlacementProfile.configuredBackend: String
    get() = when {
        vulkanShardIds.isEmpty() -> "llama.cpp-cpu-layer-range"
        vulkanShardIds.size == 3 -> "llama.cpp-vulkan-layer-range"
        else -> "llama.cpp-hybrid-layer-range"
    }

enum class BenchmarkStepStatus { PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, SKIPPED }
enum class BenchmarkUploadStatus { PENDING, UPLOADED, NOT_REQUIRED }

data class BenchmarkStepAttempt(
    val id: String = UUID.randomUUID().toString(),
    val status: BenchmarkStepStatus,
    val startedAt: String = Instant.now().toString(),
    val finishedAt: String? = null,
    val generatedTokens: Int = 0,
    val result: Map<String, Any?> = emptyMap(),
    val errorReason: String? = null,
    val uploadStatus: BenchmarkUploadStatus = BenchmarkUploadStatus.NOT_REQUIRED,
    val uploadError: String? = null
)

data class BenchmarkSuiteStep(
    val definition: BenchmarkStepDefinition,
    val attempts: List<BenchmarkStepAttempt> = emptyList()
) {
    val latestAttempt: BenchmarkStepAttempt? get() = attempts.lastOrNull()
}

data class BenchmarkSuiteCheckpoint(
    val suiteId: String = UUID.randomUUID().toString(),
    val mode: CharacterizationMode = CharacterizationMode.FULL,
    val benchmarkVersion: String = com.intellihive.worker.inference.HYBRID_BENCHMARK_VERSION,
    val createdAt: String = Instant.now().toString(),
    val updatedAt: String = Instant.now().toString(),
    val status: String = "PENDING",
    val device: Map<String, Any?> = emptyMap(),
    val steps: List<BenchmarkSuiteStep> = BenchmarkSuitePlan.stepsFor(mode).map(::BenchmarkSuiteStep)
)

object BenchmarkSuitePlan {
    val generatedTokenTargets = listOf(32, 64, 128)

    private val categories = listOf(
        Triple("llama_cpp_baseline", BenchmarkStepKind.LLAMA_CPP_BASELINE, ShardPlacementProfile.CPU_ONLY),
        Triple("cpu_only", BenchmarkStepKind.CPU_ONLY, ShardPlacementProfile.CPU_ONLY),
        Triple("vulkan_only", BenchmarkStepKind.VULKAN_ONLY, ShardPlacementProfile.VULKAN_ALL),
        Triple("hybrid", BenchmarkStepKind.HYBRID, ShardPlacementProfile.VULKAN_S0_S1)
    )

    fun stepsFor(mode: CharacterizationMode): List<BenchmarkStepDefinition> {
        val targets = if (mode == CharacterizationMode.QUICK) {
            generatedTokenTargets.take(1)
        } else {
            generatedTokenTargets
        }
        return categories.flatMap { (category, kind, profile) ->
            targets.map { tokenTarget ->
                BenchmarkStepDefinition(
                    id = "${category}_${tokenTarget}_tokens",
                    kind = kind,
                    profile = profile,
                    measurementPhase = "comparison",
                    durationMs = STEP_SAFETY_LIMIT_MS,
                    generatedTokens = tokenTarget,
                    prefillTokens = DEFAULT_PREFILL_TOKENS,
                    graphThreads = DEFAULT_GRAPH_THREADS
                )
            }
        }
    }

    fun step(id: String, mode: CharacterizationMode): BenchmarkStepDefinition? =
        stepsFor(mode).firstOrNull { it.id == id }

    fun resumableSteps(checkpoint: BenchmarkSuiteCheckpoint): List<BenchmarkSuiteStep> =
        checkpoint.steps.filter { step ->
            step.latestAttempt?.status == null ||
                step.latestAttempt?.status in setOf(BenchmarkStepStatus.PENDING, BenchmarkStepStatus.CANCELLED)
        }
}

class BenchmarkSuiteCheckpointStore(private val checkpointFile: File) {
    private val gson = Gson()

    fun readOrCreate(mode: CharacterizationMode = CharacterizationMode.FULL): BenchmarkSuiteCheckpoint {
        if (!checkpointFile.exists()) return BenchmarkSuiteCheckpoint(mode = mode).also(::write)
        val checkpoint = try {
            gson.fromJson(checkpointFile.readText(), BenchmarkSuiteCheckpoint::class.java)
                ?: throw IllegalStateException("Benchmark checkpoint is empty")
        } catch (error: Exception) {
            throw IllegalStateException("Could not read benchmark checkpoint: ${error.message}", error)
        }
        val expectedIds = BenchmarkSuitePlan.stepsFor(checkpoint.mode).map { it.id }
        if (checkpoint.steps.map { it.definition.id } != expectedIds) {
            val archive = File(
                checkpointFile.parentFile,
                "${checkpointFile.name}.legacy-${checkpoint.suiteId}.json"
            )
            try {
                Files.copy(checkpointFile.toPath(), archive.toPath())
            } catch (error: Exception) {
                throw IllegalStateException(
                    "Could not preserve the previous benchmark checkpoint at ${archive.absolutePath}",
                    error
                )
            }
            return BenchmarkSuiteCheckpoint(mode = mode).also(::write)
        }
        validate(checkpoint)
        return checkpoint
    }

    fun recoverInterrupted(mode: CharacterizationMode = CharacterizationMode.FULL): BenchmarkSuiteCheckpoint {
        val checkpoint = readOrCreate(mode)
        val recovered = checkpoint.steps.map { step ->
            val latest = step.latestAttempt
            if (latest?.status == BenchmarkStepStatus.RUNNING) step.copy(
                attempts = step.attempts.dropLast(1) + latest.copy(
                    status = BenchmarkStepStatus.CANCELLED,
                    finishedAt = Instant.now().toString(),
                    errorReason = "Interrupted when the app process stopped before this step completed."
                )
            ) else step
        }
        val updated = checkpoint.copy(
            steps = recovered,
            status = statusFor(recovered),
            updatedAt = Instant.now().toString()
        )
        if (updated != checkpoint) write(updated)
        return updated
    }

    fun write(checkpoint: BenchmarkSuiteCheckpoint) {
        validate(checkpoint)
        val parent = checkpointFile.parentFile
            ?: throw IllegalArgumentException("Benchmark checkpoint needs a parent directory")
        check(parent.isDirectory || parent.mkdirs()) {
            "Could not create benchmark checkpoint directory ${parent.absolutePath}"
        }
        val temporary = File(parent, "${checkpointFile.name}.tmp")
        try {
            FileOutputStream(temporary).use { output ->
                output.write(gson.toJson(checkpoint.copy(updatedAt = Instant.now().toString()))
                    .toByteArray(Charsets.UTF_8))
                output.fd.sync()
            }
            Files.move(temporary.toPath(), checkpointFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (error: Exception) {
            temporary.delete()
            throw IllegalStateException("Could not persist benchmark checkpoint: ${error.message}", error)
        }
    }

    fun reset(mode: CharacterizationMode = CharacterizationMode.FULL): BenchmarkSuiteCheckpoint =
        BenchmarkSuiteCheckpoint(mode = mode).also(::write)

    private fun validate(checkpoint: BenchmarkSuiteCheckpoint) {
        require(checkpoint.steps.map { it.definition.id } ==
            BenchmarkSuitePlan.stepsFor(checkpoint.mode).map { it.id }) {
            "Benchmark checkpoint does not match the ${checkpoint.mode.wireValue} plan"
        }
    }

    companion object {
        const val FILE_NAME = "benchmark_suite_checkpoint.json"

        fun statusFor(steps: List<BenchmarkSuiteStep>): String {
            val states = steps.map { it.latestAttempt?.status ?: BenchmarkStepStatus.PENDING }
            return when {
                states.all { it in setOf(BenchmarkStepStatus.COMPLETED, BenchmarkStepStatus.SKIPPED) } ->
                    if (states.any { it == BenchmarkStepStatus.SKIPPED }) "COMPLETED_WITH_FAILURES" else "COMPLETED"
                states.any { it == BenchmarkStepStatus.RUNNING } -> "RUNNING"
                states.any { it == BenchmarkStepStatus.CANCELLED } -> "CANCELLED"
                states.any { it == BenchmarkStepStatus.PENDING } -> "PENDING"
                else -> "COMPLETED_WITH_FAILURES"
            }
        }
    }
}

private const val DEFAULT_PREFILL_TOKENS = 32
private const val DEFAULT_GRAPH_THREADS = 4
private const val MAX_GENERATED_TOKENS = 128
private const val STEP_SAFETY_LIMIT_MS = 5 * 60_000L
