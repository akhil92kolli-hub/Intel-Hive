package com.intellihive.worker.benchmark

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.intellihive.worker.inference.NativeBridge
import com.intellihive.worker.inference.NativeRuntime
import com.intellihive.worker.inference.RuntimeSafetySnapshot
import java.io.File
import java.util.Locale

private const val BYTES_PER_MB = 1024L * 1024L

data class BenchmarkTelemetrySample(
    val elapsedRealtimeMs: Long,
    val batteryTemperatureC: Double?,
    val thermalStatus: String?,
    val cpuFrequencyKHz: Map<String, Long>,
    val memory: Map<String, Long?>,
    val unavailable: Map<String, String>
)

class AndroidDeviceTelemetry(private val context: Context) {
    private val gson = Gson()

    fun deviceProfile(): Map<String, Any?> {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val memoryInfo = try {
            activityManager?.let { manager ->
                ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
            }
        } catch (_: Exception) {
            null
        }
        val cpuFrequencies = cpuFrequenciesKHz("cpuinfo_max_freq")

        return mapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "android_version" to Build.VERSION.RELEASE,
            "sdk_int" to Build.VERSION.SDK_INT,
            "soc" to if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MODEL.takeIf(String::isNotBlank) ?: Build.HARDWARE
            } else {
                Build.HARDWARE
            },
            "abi" to Build.SUPPORTED_ABIS.firstOrNull(),
            "cpu_cores" to Runtime.getRuntime().availableProcessors(),
            "cpu_max_frequency_khz" to cpuFrequencies,
            "cpu_frequency_status" to if (cpuFrequencies.isEmpty()) "SKIPPED" else "AVAILABLE",
            "ram_total_mb" to memoryInfo?.totalMem?.div(BYTES_PER_MB),
            "ram_available_mb" to memoryInfo?.availMem?.div(BYTES_PER_MB),
            "system_memory_status" to if (memoryInfo == null) "SKIPPED" else "AVAILABLE",
            "gpu" to gpuProfile()
        )
    }

    fun snapshot(): BenchmarkTelemetrySample {
        val unavailable = mutableMapOf<String, String>()
        val memory = try {
            memoryProfile()
        } catch (error: Exception) {
            unavailable["memory"] = error.message ?: error.javaClass.simpleName
            emptyMap()
        }
        val temperature = try {
            batteryTemperatureC()
        } catch (error: Exception) {
            unavailable["battery_temperature"] = error.message ?: error.javaClass.simpleName
            null
        }
        val thermalStatus = try {
            thermalStatus()
        } catch (error: Exception) {
            unavailable["thermal_status"] = error.message ?: error.javaClass.simpleName
            null
        }
        return BenchmarkTelemetrySample(
            elapsedRealtimeMs = SystemClock.elapsedRealtime(),
            batteryTemperatureC = temperature,
            thermalStatus = thermalStatus,
            cpuFrequencyKHz = try {
                cpuFrequenciesKHz("scaling_cur_freq")
            } catch (error: Exception) {
                unavailable["cpu_frequency"] =
                    error.message ?: error.javaClass.simpleName
                emptyMap()
            },
            memory = memory,
            unavailable = unavailable
        )
    }

    fun runtimeSafetySnapshot(): RuntimeSafetySnapshot {
        val sample = snapshot()
        val gpu = gpuProfile()
        return RuntimeSafetySnapshot(
            availableRamMb = sample.memory["ram_available_mb"] ?: 0L,
            thermalStatus = sample.thermalStatus,
            vulkanFreeBytes = (gpu["vulkan_free_memory_bytes"] as? Number)?.toLong()
        )
    }

    private fun gpuProfile(): Map<String, Any?> {
        if (!NativeRuntime.isAvailable()) {
            return mapOf(
                "vulkan_available" to false,
                "status" to "SKIPPED",
                "skip_reason" to "Native Vulkan capability probe unavailable: ${NativeRuntime.unavailableReason()}"
            )
        }
        return try {
            val type = object : TypeToken<Map<String, Any?>>() {}.type
            gson.fromJson<Map<String, Any?>>(NativeBridge.nativeVulkanInfo(), type)
                ?: mapOf("vulkan_available" to false, "status" to "SKIPPED")
        } catch (error: Exception) {
            mapOf(
                "vulkan_available" to false,
                "status" to "SKIPPED",
                "skip_reason" to (error.message ?: error.javaClass.simpleName)
            )
        }
    }

    private fun cpuFrequenciesKHz(attribute: String): Map<String, Long> =
        (0 until Runtime.getRuntime().availableProcessors().coerceAtMost(MAX_CPU_FREQUENCY_PROBES))
            .mapNotNull { core ->
                try {
                    File("/sys/devices/system/cpu/cpu$core/cpufreq/$attribute")
                        .takeIf(File::canRead)
                        ?.readText()
                        ?.trim()
                        ?.toLongOrNull()
                        ?.let { "cpu$core" to it }
                } catch (_: Exception) {
                    null
                }
            }
            .toMap()

    private fun memoryProfile(): Map<String, Long?> {
        val activityManager = context.getSystemService(ActivityManager::class.java)
            ?: throw IllegalStateException("ActivityManager is unavailable")
        val systemMemory = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(systemMemory)
        val processMemory = Debug.MemoryInfo()
        Debug.getMemoryInfo(processMemory)
        val statusValues = File("/proc/self/status").useLines { lines ->
            lines.mapNotNull { line ->
                val name = line.substringBefore(':', "")
                val value = line.substringAfter(':', "").trim().substringBefore(' ').toLongOrNull()
                if (name.isNotEmpty() && value != null) name to value else null
            }.toMap()
        }
        val majorPageFaults = File("/proc/self/stat").useLines { lines ->
            val stat = lines.firstOrNull().orEmpty()
            val fields = stat.substringAfterLast(')', "").trim().split(WHITESPACE)
            fields.getOrNull(PROC_STAT_MAJOR_FAULTS_FIELD_INDEX)?.toLongOrNull()
        }
        val runtime = Runtime.getRuntime()
        return mapOf(
            "ram_available_mb" to systemMemory.availMem / BYTES_PER_MB,
            "app_pss_mb" to processMemory.totalPss.toLong() / 1024L,
            "app_rss_mb" to statusValues["VmRSS"]?.div(1024L),
            "native_heap_allocated_mb" to Debug.getNativeHeapAllocatedSize() / BYTES_PER_MB,
            "java_heap_used_mb" to (runtime.totalMemory() - runtime.freeMemory()) / BYTES_PER_MB,
            "major_page_faults" to majorPageFaults
        )
    }

    private fun batteryTemperatureC(): Double? {
        val batteryIntent: Intent = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        ) ?: return null
        val temperature = batteryIntent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        return if (temperature == Int.MIN_VALUE) null else temperature / 10.0
    }

    private fun thermalStatus(): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return null
        val powerManager = context.getSystemService(PowerManager::class.java)
            ?: return null
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> String.format(Locale.US, "UNKNOWN_%d", powerManager.currentThermalStatus)
        }
    }

    companion object {
        private const val MAX_CPU_FREQUENCY_PROBES = 128
        private const val PROC_STAT_MAJOR_FAULTS_FIELD_INDEX = 9
        private val WHITESPACE = Regex("\\s+")
    }
}
