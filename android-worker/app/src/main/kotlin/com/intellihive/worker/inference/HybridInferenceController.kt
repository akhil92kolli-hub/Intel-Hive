package com.intellihive.worker.inference

import com.intellihive.worker.benchmark.BenchmarkReadiness
import com.intellihive.worker.model.ModelManifest

const val HYBRID_BENCHMARK_VERSION = "0.6.0"

enum class ShardPlacementProfile(
    val wireValue: String,
    val vulkanShardIds: Set<String>,
    val backendTransitions: Int
) {
    CPU_ONLY("cpu_only", emptySet(), 0),
    VULKAN_S0("vulkan_s0", setOf("s0"), 1),
    VULKAN_S0_S1("vulkan_s0_s1", setOf("s0", "s1"), 1),
    VULKAN_ALL("vulkan_all", setOf("s0", "s1", "s2"), 0),
    VULKAN_S2("vulkan_s2", setOf("s2"), 1),
    VULKAN_S1_S2("vulkan_s1_s2", setOf("s1", "s2"), 1);

    val isHybrid: Boolean
        get() = vulkanShardIds.isNotEmpty() && vulkanShardIds.size < QwenShardCatalog.ranges.size

    companion object {
        fun fromWireValue(value: String): ShardPlacementProfile? =
            entries.firstOrNull { it.wireValue == value }
    }
}

data class ShardPlacementPlan(
    val profileId: String,
    val cpuThreads: Int,
    val benchmarkVersion: String = HYBRID_BENCHMARK_VERSION,
    val modelArtifactDigest: String = ModelManifest.PINNED_ARTIFACT_DIGEST,
    val sustainedGenerationTokensPerSecond: Double = 0.0,
    val additionalMemoryMb: Long = 0,
    val gpuWeightBytes: Long = 0,
    val thermalRiseC: Double? = null,
    val thermalStatusPeak: String? = null,
    val measurementPhase: String = "screening"
) {
    val profile: ShardPlacementProfile
        get() = requireNotNull(ShardPlacementProfile.fromWireValue(profileId)) {
            "Unknown shard placement profile: $profileId"
        }

    fun validate(): String? = when {
        ShardPlacementProfile.fromWireValue(profileId) == null ->
            "unknown placement profile"
        cpuThreads !in 1..8 -> "cpuThreads must be between 1 and 8"
        benchmarkVersion != HYBRID_BENCHMARK_VERSION -> "benchmark version is incompatible"
        modelArtifactDigest != ModelManifest.PINNED_ARTIFACT_DIGEST ->
            "model artifact digest is incompatible"
        !sustainedGenerationTokensPerSecond.isFinite() || sustainedGenerationTokensPerSecond < 0.0 ->
            "sustained generation rate is invalid"
        additionalMemoryMb < 0 || gpuWeightBytes < 0 -> "placement memory metrics are invalid"
        measurementPhase !in setOf("screening", "sustained") -> "measurement phase is invalid"
        else -> null
    }

    companion object {
        fun cpuFallback(cpuThreads: Int = 4) = ShardPlacementPlan(
            profileId = ShardPlacementProfile.CPU_ONLY.wireValue,
            cpuThreads = cpuThreads,
            measurementPhase = "screening"
        )
    }
}

data class RuntimeSafetySnapshot(
    val availableRamMb: Long,
    val thermalStatus: String?,
    val vulkanFreeBytes: Long? = null
)

class ThermalSafetyException(message: String) : IllegalStateException(message)

/**
 * Selects one worker-wide placement between sequences. A placement is frozen
 * while any sequence owns native KV state.
 */
class HybridInferenceController(
    private val readiness: BenchmarkReadiness,
    private val applyPlan: (ShardPlacementPlan) -> Unit
) {
    private val liveSequences = linkedSetOf<String>()
    private val blacklistedProfiles = linkedSetOf<String>()
    private var currentPlan: ShardPlacementPlan? = null

    @Synchronized
    fun initialize(snapshot: RuntimeSafetySnapshot): ShardPlacementPlan =
        chooseAndApply(snapshot)

    @Synchronized
    fun beginSequence(sequenceId: String, snapshot: RuntimeSafetySnapshot): ShardPlacementPlan {
        require(sequenceId.isNotBlank()) { "sequenceId is required" }
        currentPlan?.let { plan ->
            if (sequenceId in liveSequences) return plan
        }
        if (thermalSeverity(snapshot.thermalStatus) >= THERMAL_SEVERE) {
            throw ThermalSafetyException(
                "Device thermal status ${snapshot.thermalStatus} is too high to start another sequence"
            )
        }
        val selected = if (liveSequences.isEmpty()) chooseAndApply(snapshot) else checkNotNull(currentPlan)
        liveSequences += sequenceId
        return selected
    }

    @Synchronized
    fun endSequence(sequenceId: String) {
        liveSequences.remove(sequenceId)
    }

    @Synchronized
    fun failSequence(sequenceId: String) {
        currentPlan?.takeIf { it.profile.vulkanShardIds.isNotEmpty() }?.let {
            blacklistedProfiles += it.profileId
        }
        liveSequences.remove(sequenceId)
    }

    @Synchronized
    fun selectedPlan(): ShardPlacementPlan = currentPlan ?: ShardPlacementPlan.cpuFallback()

    @Synchronized
    fun verifiedPlans(): List<ShardPlacementPlan> = eligibleBenchmarkPlans()

    @Synchronized
    fun blacklistedProfileIds(): Set<String> = blacklistedProfiles.toSet()

    private fun chooseAndApply(snapshot: RuntimeSafetySnapshot): ShardPlacementPlan {
        check(liveSequences.isEmpty()) { "placement cannot change while sequences own KV state" }
        val eligible = eligibleBenchmarkPlans()
            .filterNot { it.profileId in blacklistedProfiles }
            .filter { isSafe(it, snapshot) }
        val best = eligible.sortedWith(
            compareByDescending<ShardPlacementPlan> { it.sustainedGenerationTokensPerSecond }
                .thenBy { it.thermalRiseC ?: Double.POSITIVE_INFINITY }
                .thenBy { it.profile.backendTransitions }
        ).firstOrNull() ?: ShardPlacementPlan.cpuFallback(currentPlan?.cpuThreads ?: 4)

        val retained = currentPlan?.takeIf { current ->
            isSafe(current, snapshot) && current.profileId !in blacklistedProfiles &&
                best.sustainedGenerationTokensPerSecond <
                current.sustainedGenerationTokensPerSecond * SWITCH_THRESHOLD
        }
        val selected = retained ?: best
        if (currentPlan != selected) {
            try {
                applyPlan(selected)
                currentPlan = selected
            } catch (error: Exception) {
                if (selected.profile.vulkanShardIds.isEmpty()) throw error
                blacklistedProfiles += selected.profileId
                val fallback = ShardPlacementPlan.cpuFallback(selected.cpuThreads)
                applyPlan(fallback)
                currentPlan = fallback
            }
        }
        return checkNotNull(currentPlan)
    }

    private fun eligibleBenchmarkPlans(): List<ShardPlacementPlan> {
        if (readiness.benchmarkMode != "full" ||
            readiness.benchmarkVersion != HYBRID_BENCHMARK_VERSION ||
            readiness.modelArtifactDigest != ModelManifest.PINNED_ARTIFACT_DIGEST
        ) return emptyList()
        return readiness.verifiedPlacements.filter { it.validate() == null && it.measurementPhase == "sustained" }
    }

    private fun isSafe(plan: ShardPlacementPlan, snapshot: RuntimeSafetySnapshot): Boolean {
        if (thermalSeverity(snapshot.thermalStatus) >= THERMAL_SEVERE) return false
        if (snapshot.availableRamMb < plan.additionalMemoryMb + MIN_RAM_HEADROOM_MB) return false
        if (plan.profile.vulkanShardIds.isNotEmpty()) {
            val free = snapshot.vulkanFreeBytes ?: return false
            if (plan.gpuWeightBytes > 0 && plan.gpuWeightBytes * GPU_HEADROOM_DENOMINATOR >
                free * GPU_HEADROOM_NUMERATOR
            ) return false
        }
        return true
    }

    companion object {
        private const val SWITCH_THRESHOLD = 1.05
        private const val MIN_RAM_HEADROOM_MB = 512L
        private const val GPU_HEADROOM_NUMERATOR = 3L
        private const val GPU_HEADROOM_DENOMINATOR = 4L
        private const val THERMAL_SEVERE = 3

        fun thermalSeverity(status: String?): Int = when (status) {
            "NONE" -> 0
            "LIGHT" -> 1
            "MODERATE" -> 2
            "SEVERE" -> 3
            "CRITICAL" -> 4
            "EMERGENCY" -> 5
            "SHUTDOWN" -> 6
            else -> 0
        }
    }
}
