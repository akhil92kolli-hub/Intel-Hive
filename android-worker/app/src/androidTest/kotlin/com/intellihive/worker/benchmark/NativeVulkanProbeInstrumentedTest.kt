package com.intellihive.worker.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.intellihive.worker.inference.NativeBridge
import com.intellihive.worker.inference.NativeRuntime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject

@RunWith(AndroidJUnit4::class)
class NativeVulkanProbeInstrumentedTest {
    @Test
    fun reportsVulkanDeviceAndInferenceBackendAvailabilitySeparately() {
        assertTrue("IntelHive JNI library should load on supported arm64 devices", NativeRuntime.isAvailable())

        val result = JSONObject(NativeBridge.nativeVulkanInfo())
        assertTrue(result.has("available"))
        assertTrue(result.has("vulkan_available"))
        if (!result.getBoolean("available")) {
            assertFalse(result.getBoolean("vulkan_available"))
            assertTrue(result.getString("reason").isNotBlank())
            return
        }
        assertTrue(result.getString("api_version").isNotBlank())
        assertTrue(result.getString("gpu_name").isNotBlank())
        assertTrue(result.has("vendor_id"))
        assertTrue(result.has("driver_version"))
        assertEquals(
            result.getBoolean("inference_backend_enabled"),
            result.getBoolean("vulkan_available")
        )
        if (result.getBoolean("vulkan_available")) {
            assertTrue(result.getBoolean("ggml_backend_compiled"))
            assertTrue(result.getInt("ggml_backend_device_count") > 0)
            assertEquals("NOT_RUN", result.getString("inference_status"))
        } else {
            assertEquals("SKIPPED", result.getString("inference_status"))
            assertTrue(result.getString("inference_skip_reason").isNotBlank())
        }
    }
}
