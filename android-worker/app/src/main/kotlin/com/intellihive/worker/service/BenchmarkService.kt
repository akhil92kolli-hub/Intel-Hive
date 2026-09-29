package com.intellihive.worker.service

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.intellihive.worker.BuildConfig
import com.intellihive.worker.benchmark.SupabaseBenchmarkUploader
import com.intellihive.worker.inference.LlamaCppBenchmarkEngine
import com.intellihive.worker.model.ModelManifest
import com.intellihive.worker.model.RequiredModelManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class BenchmarkService : Service() {

    private val engine by lazy { LlamaCppBenchmarkEngine(this) }
    private val benchmarkUploader by lazy { SupabaseBenchmarkUploader() }
    private val gson: Gson = GsonBuilder().setPrettyPrinting().create()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d("BenchmarkService", "Starting benchmark service")

        val prefillTokens = intent?.getIntExtra("prefill_tokens", 128) ?: 128
        val generatedTokens = intent?.getIntExtra("generated_tokens", 100) ?: 100

        serviceScope.launch {
            try {
                runBenchmark(prefillTokens, generatedTokens)
            } catch (e: Exception) {
                Log.e("BenchmarkService", "Benchmark failed", e)
            } finally {
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        serviceScope.cancel()
        engine.unload()
        super.onDestroy()
    }

    private suspend fun runBenchmark(prefillTokens: Int, generatedTokens: Int) {
        Log.d("BenchmarkService", "Loading model...")

        val modelPath = RequiredModelManager.defaultModelFile(this@BenchmarkService).absolutePath

        if (!engine.loadModel(modelPath, gpuLayers = 36)) {
            throw IllegalStateException("Failed to load model at $modelPath")
        }

        Log.d("BenchmarkService", "Model loaded. Running benchmark...")

        val prompt = "The future of AI is"
        val result = engine.runBenchmark(prompt, prefillTokens, generatedTokens)

        Log.d("BenchmarkService", "Benchmark complete: ${result.tokensPerSecond} tok/s")

        val output = mapOf(
            "timestamp" to SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).format(Date()),
            "model" to result.modelName,
            "quantization" to result.quantization,
            "device" to mapOf(
                "model" to getDeviceModel(),
                "android_version" to android.os.Build.VERSION.RELEASE,
                "ram_mb" to getTotalRAM(),
                "gpu" to getGPUInfo()
            ),
            "benchmark" to mapOf(
                "prefill_tokens" to result.prefillTokens,
                "generated_tokens" to result.generatedTokens,
                "total_time_ms" to result.totalTimeMs,
                "tokens_per_second" to result.tokensPerSecond,
                "prefill_speed_tokens_per_second" to result.prefillSpeedTokensPerSecond,
                "generation_speed_tokens_per_second" to result.generationSpeedTokensPerSecond
            ),
            "memory" to mapOf(
                "peak_rss_mb" to result.peakRssMb,
                "peak_gpu_mb" to result.peakGpuMb
            ),
            "thermal" to mapOf(
                "initial_temp_c" to result.initialTempC,
                "peak_temp_c" to result.peakTempC,
                "final_temp_c" to result.finalTempC
            ),
            "activations" to mapOf(
                "layer_output_size_bytes" to result.activationSizeBytes,
                "dtype" to result.activationDtype,
                "shape" to result.activationShape
            )
        )

        val json = gson.toJson(output)
        Log.d("BenchmarkService", "Result:\n$json")

        // Write to file
        val outputFile = File(cacheDir, "benchmark_result.json")
        outputFile.writeText(json)
        Log.d("BenchmarkService", "Saved to: ${outputFile.absolutePath}")

        val preferences = getSharedPreferences("worker_runtime", MODE_PRIVATE)
        val workerId = preferences.getString("worker_id", null)
            ?.takeIf(String::isNotBlank)
            ?: UUID.randomUUID().toString().also {
                preferences.edit().putString("worker_id", it).apply()
            }
        val databaseRow = mapOf(
            "client_run_id" to UUID.randomUUID().toString(),
            "platform" to "android",
            "worker_id" to workerId,
            "app_version" to BuildConfig.VERSION_NAME,
            "model_id" to ModelManifest.PINNED_MODEL_ID,
            "model_version" to ModelManifest.PINNED_MODEL_VERSION,
            "model_artifact_digest" to ModelManifest.PINNED_ARTIFACT_DIGEST,
            "device_model" to getDeviceModel(),
            "os_version" to android.os.Build.VERSION.RELEASE,
            "backend" to "llama.cpp-cpu",
            "prefill_tokens" to result.prefillTokens,
            "generated_tokens" to result.generatedTokens,
            "total_time_ms" to result.totalTimeMs,
            "tokens_per_second" to result.tokensPerSecond,
            "prefill_tokens_per_second" to result.prefillSpeedTokensPerSecond,
            "generation_tokens_per_second" to result.generationSpeedTokensPerSecond,
            "peak_rss_mb" to result.peakRssMb,
            "peak_gpu_mb" to result.peakGpuMb,
            "initial_temperature_c" to result.initialTempC,
            "peak_temperature_c" to result.peakTempC,
            "final_temperature_c" to result.finalTempC,
            "activation_size_bytes" to result.activationSizeBytes,
            "activation_dtype" to result.activationDtype,
            "activation_shape" to result.activationShape,
            "raw_result" to output
        )
        try {
            benchmarkUploader.upload(databaseRow)
            Log.d("BenchmarkService", "Benchmark uploaded to Supabase")
        } catch (error: Exception) {
            Log.e(
                "BenchmarkService",
                "Benchmark saved locally but Supabase upload failed",
                error
            )
        }

    }

    private fun getDeviceModel(): String = android.os.Build.MODEL
    private fun getTotalRAM(): Long = Runtime.getRuntime().maxMemory() / (1024 * 1024)
    private fun getGPUInfo(): String {
        // Read GPU info from system properties
        return try {
            val process = Runtime.getRuntime().exec("getprop ro.hardware.keystore")
            process.inputStream.bufferedReader().readText().trim()
        } catch (e: Exception) {
            "Unknown"
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
