package com.intellihive.worker.inference

import com.intellihive.worker.benchmark.BenchmarkExecutionMode
import com.intellihive.worker.benchmark.BenchmarkReadiness
import com.intellihive.worker.benchmark.BenchmarkStatus
import com.intellihive.worker.model.ModelManifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class HybridInferenceControllerTest {
    private val safe = RuntimeSafetySnapshot(
        availableRamMb = 4096,
        thermalStatus = "NONE",
        vulkanFreeBytes = 4L * 1024 * 1024 * 1024
    )

    @Test
    fun ranksThroughputThenThermalThenBoundaries() {
        val applied = mutableListOf<ShardPlacementPlan>()
        val controller = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.CPU_ONLY, 1.0, thermal = 0.0),
            plan(ShardPlacementProfile.VULKAN_S0, 2.0, thermal = 2.0),
            plan(ShardPlacementProfile.VULKAN_ALL, 2.0, thermal = 1.0)
        ), applied::add)

        assertEquals(ShardPlacementProfile.VULKAN_ALL, controller.initialize(safe).profile)
        assertEquals(ShardPlacementProfile.VULKAN_ALL, applied.single().profile)
    }

    @Test
    fun requiresFivePercentImprovementBeforeSwitching() {
        val applied = mutableListOf<ShardPlacementPlan>()
        val controller = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.CPU_ONLY, 1.0),
            plan(ShardPlacementProfile.VULKAN_ALL, 1.04)
        ), applied::add)
        assertEquals(
            ShardPlacementProfile.CPU_ONLY,
            controller.initialize(safe.copy(vulkanFreeBytes = null)).profile
        )
        controller.beginSequence("a", safe)
        assertEquals(ShardPlacementProfile.CPU_ONLY, controller.selectedPlan().profile)
        assertEquals(1, applied.size)
    }

    @Test
    fun freezesPlacementUntilEverySequenceEnds() {
        val applied = mutableListOf<ShardPlacementPlan>()
        val controller = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.VULKAN_ALL, 2.0)
        ), applied::add)

        controller.beginSequence("a", safe)
        controller.beginSequence("b", safe.copy(availableRamMb = 1))
        assertEquals(1, applied.size)
        controller.endSequence("a")
        assertEquals(ShardPlacementProfile.VULKAN_ALL, controller.selectedPlan().profile)
        controller.endSequence("b")
        controller.beginSequence("c", safe.copy(availableRamMb = 1))
        assertEquals(ShardPlacementProfile.CPU_ONLY, controller.selectedPlan().profile)
    }

    @Test
    fun rejectsNewSequenceAtSevereThermalStatus() {
        val controller = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.CPU_ONLY, 1.0)
        )) { }
        assertThrows(ThermalSafetyException::class.java) {
            controller.beginSequence("hot", safe.copy(thermalStatus = "SEVERE"))
        }
    }

    @Test
    fun rejectsMemoryUnsafeProfileAndBlacklistsFailedVulkanProfile() {
        val controller = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.CPU_ONLY, 1.0),
            plan(ShardPlacementProfile.VULKAN_ALL, 2.0, memoryMb = 1000)
        )) { }
        assertEquals(
            ShardPlacementProfile.CPU_ONLY,
            controller.initialize(safe.copy(availableRamMb = 1400)).profile
        )

        val retryController = HybridInferenceController(readiness(
            plan(ShardPlacementProfile.CPU_ONLY, 1.0),
            plan(ShardPlacementProfile.VULKAN_ALL, 2.0)
        )) { }
        retryController.beginSequence("failed", safe)
        retryController.failSequence("failed")
        assertTrue(ShardPlacementProfile.VULKAN_ALL.wireValue in retryController.blacklistedProfileIds())
        assertEquals(ShardPlacementProfile.CPU_ONLY, retryController.beginSequence("retry", safe).profile)
    }

    @Test
    fun incompatibleBenchmarkDefaultsToCpu() {
        val incompatible = readiness(plan(ShardPlacementProfile.VULKAN_ALL, 2.0)).copy(
            benchmarkVersion = "old"
        )
        val controller = HybridInferenceController(incompatible) { }
        assertEquals(ShardPlacementProfile.CPU_ONLY, controller.initialize(safe).profile)
    }

    private fun plan(
        profile: ShardPlacementProfile,
        throughput: Double,
        thermal: Double? = null,
        memoryMb: Long = 0
    ) = ShardPlacementPlan(
        profileId = profile.wireValue,
        cpuThreads = 4,
        sustainedGenerationTokensPerSecond = throughput,
        additionalMemoryMb = memoryMb,
        gpuWeightBytes = if (profile.vulkanShardIds.isEmpty()) 0 else 256L * 1024 * 1024,
        thermalRiseC = thermal,
        measurementPhase = "sustained"
    )

    private fun readiness(vararg plans: ShardPlacementPlan) = BenchmarkReadiness(
        status = BenchmarkStatus.COMPLETED,
        executionMode = BenchmarkExecutionMode.SINGLE_DEVICE_ALL_SHARDS,
        modelId = ModelManifest.PINNED_MODEL_ID,
        modelVersion = ModelManifest.PINNED_MODEL_VERSION,
        modelArtifactDigest = ModelManifest.PINNED_ARTIFACT_DIGEST,
        prefillTokens = 32,
        generatedTokens = 8,
        totalTimeMs = 1000,
        tokensPerSecond = 1.0,
        prefillSpeedTokensPerSecond = 1.0,
        generationSpeedTokensPerSecond = 1.0,
        layerRanges = QwenShardCatalog.ranges,
        benchmarkMode = "full",
        benchmarkVersion = HYBRID_BENCHMARK_VERSION,
        selectedPlacement = plans.firstOrNull(),
        verifiedPlacements = plans.toList()
    )
}
