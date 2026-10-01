package com.intellihive.worker.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.gson.GsonBuilder
import com.intellihive.worker.BuildConfig
import com.intellihive.worker.MainActivity
import com.intellihive.worker.benchmark.BenchmarkReadiness
import com.intellihive.worker.benchmark.BenchmarkReadinessStore
import com.intellihive.worker.benchmark.SupabaseBenchmarkUploader
import com.intellihive.worker.inference.NativeLayerRangeShardExecutor
import com.intellihive.worker.inference.NativeRuntime
import com.intellihive.worker.inference.NativeShardBenchmarkEngine
import com.intellihive.worker.model.ModelManifest
import com.intellihive.worker.model.RequiredModelManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

class BenchmarkService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = GsonBuilder().setPrettyPrinting().create()
    private var benchmarkJob: Job? = null
    private var nativeExecutor: NativeLayerRangeShardExecutor? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (benchmarkJob?.isActive == true) return START_NOT_STICKY

        startForeground(NOTIFICATION_ID, notification("Preparing native shard benchmark"))
        val prefillTokens = intent?.getIntExtra(EXTRA_PREFILL_TOKENS, DEFAULT_PREFILL_TOKENS)
            ?: DEFAULT_PREFILL_TOKENS
        val generatedTokens = intent?.getIntExtra(EXTRA_GENERATED_TOKENS, DEFAULT_GENERATED_TOKENS)
            ?: DEFAULT_GENERATED_TOKENS

        benchmarkJob = serviceScope.launch {
            var benchmarkStarted = false
            val store = BenchmarkReadinessStore(File(filesDir, BenchmarkReadinessStore.FILE_NAME))
            try {
                ensureWorkerDisconnected()
                store.read()
                store.write(BenchmarkReadiness.running())
                benchmarkStarted = true
                notifyStatus(
                    STATUS_RUNNING,
                    "Verifying the downloaded model and preparing the native shard executor"
                )

                val models = RequiredModelManager(this@BenchmarkService)
                val manifest = models.loadManifest()
                check(models.isInstalled(manifest)) {
                    "The verified Qwen2.5-3B model is not installed; download it before benchmarking."
                }
                check(NativeRuntime.isAvailable()) {
                    "Native engine unavailable: ${NativeRuntime.unavailableReason()}"
                }

                nativeExecutor = NativeLayerRangeShardExecutor(this@BenchmarkService)
                nativeExecutor?.prepareModel()
                val result = NativeShardBenchmarkEngine(checkNotNull(nativeExecutor)).run(
                    prefillTokens = prefillTokens,
                    generatedTokens = generatedTokens
                ) { completed, total ->
                    notifyStatus(
                        STATUS_RUNNING,
                        "Executing native shard benchmark pass $completed of $total",
                        progress = completed * 100 / total
                    )
                }
                val completed = BenchmarkReadiness.completed(
                    prefillTokens = result.prefillTokens,
                    generatedTokens = result.generatedTokens,
                    totalTimeMs = result.totalTimeMs,
                    tokensPerSecond = result.tokensPerSecond,
                    prefillSpeedTokensPerSecond = result.prefillSpeedTokensPerSecond,
                    generationSpeedTokensPerSecond = result.generationSpeedTokensPerSecond,
                    layerRanges = result.layerRanges
                )
                store.write(completed)
                val metrics = gson.toJson(
                    mapOf(
                        "status" to completed.status.name,
                        "execution_mode" to completed.executionMode?.wireValue,
                        "model_id" to completed.modelId,
                        "model_version" to completed.modelVersion,
                        "model_artifact_digest" to completed.modelArtifactDigest,
                        "prefill_tokens" to completed.prefillTokens,
                        "generated_tokens" to completed.generatedTokens,
                        "total_time_ms" to completed.totalTimeMs,
                        "tokens_per_second" to completed.tokensPerSecond,
                        "prefill_tokens_per_second" to completed.prefillSpeedTokensPerSecond,
                        "generation_tokens_per_second" to completed.generationSpeedTokensPerSecond,
                        "layer_ranges" to completed.layerRanges,
                        "completed_at" to completed.completedAt
                    )
                )
                notifyStatus(
                    STATUS_COMPLETED,
                    "Benchmark completed: ${"%.2f".format(java.util.Locale.US, result.tokensPerSecond)} tokens/s. Reconnect to report readiness.",
                    metrics = metrics,
                    progress = 100
                )
                uploadResult(completed, metrics)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                Log.e(TAG, "Native shard benchmark failed", error)
                if (benchmarkStarted) {
                    val failure = BenchmarkReadiness.failed(
                        error.message ?: error.javaClass.simpleName
                    )
                    try {
                        store.write(failure)
                    } catch (persistError: Exception) {
                        Log.e(TAG, "Could not persist benchmark failure state", persistError)
                    }
                    notifyStatus(STATUS_FAILED, failure.errorMessage.orEmpty())
                } else {
                    notifyStatus(STATUS_FAILED, error.message ?: "Benchmark could not start")
                }
            } finally {
                nativeExecutor?.close()
                nativeExecutor = null
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        benchmarkJob?.cancel()
        benchmarkJob = null
        nativeExecutor?.close()
        nativeExecutor = null
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureWorkerDisconnected() {
        val preferences = getSharedPreferences(WorkerService.PREFERENCES, Context.MODE_PRIVATE)
        check(!isAppServiceRunning(this, WorkerService::class.java) &&
            !preferences.getBoolean(WorkerService.KEY_CONNECTED, false) &&
            !preferences.getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)) {
            "Disconnect the worker and wait for it to stop before running the benchmark."
        }
    }

    private suspend fun uploadResult(record: BenchmarkReadiness, rawResult: String) {
        val workerId = getSharedPreferences(WorkerService.PREFERENCES, Context.MODE_PRIVATE)
            .getString(WorkerService.KEY_WORKER_ID, null)
            ?.takeIf(String::isNotBlank)
            ?: UUID.randomUUID().toString()
        val row = mapOf(
            "client_run_id" to UUID.randomUUID().toString(),
            "platform" to "android",
            "worker_id" to workerId,
            "app_version" to BuildConfig.VERSION_NAME,
            "model_id" to record.modelId,
            "model_version" to record.modelVersion,
            "model_artifact_digest" to record.modelArtifactDigest,
            "device_model" to Build.MODEL,
            "os_version" to Build.VERSION.RELEASE,
            "backend" to "llama.cpp-cpu-layer-range",
            "execution_mode" to record.executionMode?.wireValue,
            "prefill_tokens" to record.prefillTokens,
            "generated_tokens" to record.generatedTokens,
            "total_time_ms" to record.totalTimeMs,
            "tokens_per_second" to record.tokensPerSecond,
            "prefill_tokens_per_second" to record.prefillSpeedTokensPerSecond,
            "generation_tokens_per_second" to record.generationSpeedTokensPerSecond,
            "raw_result" to gson.fromJson(rawResult, Any::class.java)
        )
        try {
            SupabaseBenchmarkUploader().upload(row)
            Log.i(TAG, "Completed benchmark uploaded")
        } catch (error: Exception) {
            Log.e(TAG, "Benchmark completed and persisted locally; upload failed", error)
        }
    }

    private fun notifyStatus(
        status: String,
        message: String,
        metrics: String? = null,
        progress: Int? = null
    ) {
        sendBroadcast(
            Intent(ACTION_STATUS)
                .setPackage(packageName)
                .putExtra(EXTRA_STATUS, status)
                .putExtra(EXTRA_MESSAGE, message)
                .apply {
                    metrics?.let { putExtra(EXTRA_METRICS, it) }
                    progress?.let { putExtra(EXTRA_PROGRESS, it) }
                }
        )
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, notification(message))
    }

    private fun notification(message: String): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "IntelHive benchmark",
                    NotificationManager.IMPORTANCE_LOW
                )
            )
        }
        val openActivity = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("IntelHive native benchmark")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(openActivity)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val ACTION_STATUS = "com.intellihive.worker.BENCHMARK_STATUS"
        const val EXTRA_STATUS = "benchmark_status"
        const val EXTRA_MESSAGE = "benchmark_message"
        const val EXTRA_METRICS = "benchmark_metrics"
        const val EXTRA_PROGRESS = "benchmark_progress"
        const val EXTRA_PREFILL_TOKENS = "prefill_tokens"
        const val EXTRA_GENERATED_TOKENS = "generated_tokens"
        const val STATUS_RUNNING = "RUNNING"
        const val STATUS_COMPLETED = "COMPLETED"
        const val STATUS_FAILED = "FAILED"
        private const val TAG = "BenchmarkService"
        private const val CHANNEL_ID = "native_benchmark"
        private const val NOTIFICATION_ID = 1102
        private const val DEFAULT_PREFILL_TOKENS = 32
        private const val DEFAULT_GENERATED_TOKENS = 8
    }
}
