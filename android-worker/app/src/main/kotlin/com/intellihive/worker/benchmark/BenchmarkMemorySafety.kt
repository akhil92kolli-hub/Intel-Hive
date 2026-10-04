package com.intellihive.worker.benchmark

import kotlin.math.ceil

object BenchmarkMemorySafety {
    private const val BYTES_PER_MIB = 1024L * 1024L
    private const val RAM_HEADROOM_MB = 512L
    private const val MODEL_MEMORY_MULTIPLIER = 1.25

    fun requiredAvailableRamMb(modelSizeBytes: Long): Long {
        require(modelSizeBytes > 0L) { "modelSizeBytes must be greater than zero" }
        val modelAndRuntimeMb = ceil(
            modelSizeBytes.toDouble() * MODEL_MEMORY_MULTIPLIER / BYTES_PER_MIB
        ).toLong()
        return modelAndRuntimeMb + RAM_HEADROOM_MB
    }

    fun vulkanSkipReason(
        availableRamMb: Long?,
        modelSizeBytes: Long
    ): String? {
        val requiredRamMb = requiredAvailableRamMb(modelSizeBytes)
        if (availableRamMb == null) {
            return "Vulkan benchmark skipped for device safety: available system RAM could not be measured."
        }
        if (availableRamMb >= requiredRamMb) return null
        return "Vulkan benchmark skipped for device safety: available RAM ${availableRamMb} MB is below " +
            "the estimated ${requiredRamMb} MB requirement."
    }
}
