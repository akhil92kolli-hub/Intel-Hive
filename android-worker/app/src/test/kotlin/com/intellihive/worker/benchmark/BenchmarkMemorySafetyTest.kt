package com.intellihive.worker.benchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BenchmarkMemorySafetyTest {
    @Test
    fun requiredRamIncludesModelRuntimeMultiplierAndHeadroom() {
        assertEquals(
            3_022L,
            BenchmarkMemorySafety.requiredAvailableRamMb(2_104_932_768L)
        )
    }

    @Test
    fun vulkanIsSkippedWhenAvailableRamIsBelowTheSafeThreshold() {
        assertEquals(
            "Vulkan benchmark skipped for device safety: available RAM 2365 MB is below " +
                "the estimated 3022 MB requirement.",
            BenchmarkMemorySafety.vulkanSkipReason(2_365L, 2_104_932_768L)
        )
    }

    @Test
    fun vulkanIsEligibleAtTheSafeThreshold() {
        assertNull(BenchmarkMemorySafety.vulkanSkipReason(3_022L, 2_104_932_768L))
    }

    @Test
    fun vulkanIsSkippedWhenAvailableRamCannotBeMeasured() {
        assertEquals(
            "Vulkan benchmark skipped for device safety: available system RAM could not be measured.",
            BenchmarkMemorySafety.vulkanSkipReason(null, 2_104_932_768L)
        )
    }
}
