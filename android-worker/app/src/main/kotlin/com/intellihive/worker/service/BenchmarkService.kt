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
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.gson.Gson
import com.google.gson.JsonParser
import com.intellihive.worker.BuildConfig
import com.intellihive.worker.MainActivity
import com.intellihive.worker.benchmark.AndroidDeviceTelemetry
import com.intellihive.worker.benchmark.BenchmarkReadiness
import com.intellihive.worker.benchmark.BenchmarkReadinessStore
import com.intellihive.worker.benchmark.BenchmarkMemorySafety
import com.intellihive.worker.benchmark.BenchmarkStepAttempt
import com.intellihive.worker.benchmark.BenchmarkStepKind
import com.intellihive.worker.benchmark.BenchmarkStepStatus
import com.intellihive.worker.benchmark.BenchmarkSuiteCheckpoint
import com.intellihive.worker.benchmark.BenchmarkSuiteCheckpointStore
import com.intellihive.worker.benchmark.BenchmarkSuitePlan
import com.intellihive.worker.benchmark.BenchmarkSuiteStep
import com.intellihive.worker.benchmark.BenchmarkUploadStatus
import com.intellihive.worker.benchmark.CharacterizationMode
import com.intellihive.worker.benchmark.SupabaseBenchmarkUploader
import com.intellihive.worker.benchmark.configuredBackend
import com.intellihive.worker.benchmark.displayName
import com.intellihive.worker.inference.HYBRID_BENCHMARK_VERSION
import com.intellihive.worker.inference.NativeLayerRangeShardExecutor
import com.intellihive.worker.inference.NativeBridge
import com.intellihive.worker.inference.NativeRuntime
import com.intellihive.worker.inference.NativeShardBenchmarkEngine
import com.intellihive.worker.inference.QwenShardCatalog
import com.intellihive.worker.inference.ShardPlacementPlan
import com.intellihive.worker.inference.ShardPlacementProfile
import com.intellihive.worker.model.ModelManifest
import com.intellihive.worker.model.RequiredModelManager
import java.io.File
import java.time.Instant
import java.util.Collections
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class BenchmarkService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    private var benchmarkJob: Job? = null
    private var nativeExecutor: NativeLayerRangeShardExecutor? = null
    private var benchmarkWakeLock: PowerManager.WakeLock? = null
    private val progressLock = Any()
    @Volatile
    private var cancelRequested = false

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            cancelRequested = true
            notifyStatus(STATUS_RUNNING, "Stopping after the current native inference pass")
            return START_NOT_STICKY
        }
        if (benchmarkJob?.isActive == true) return START_NOT_STICKY
        cancelRequested = false
        startForeground(NOTIFICATION_ID, notification("Preparing resumable benchmark"))
        try {
            acquireBenchmarkWakeLock()
        } catch (error: Exception) {
            Log.e(TAG, "Could not keep the benchmark CPU active while the screen is off", error)
            return failToStart(
                startId,
                "Could not acquire the benchmark wake lock: ${exceptionReason(error)}"
            )
        }
        val retryStepId = if (intent?.action == ACTION_RETRY_STEP) {
            intent.getStringExtra(EXTRA_STEP_ID)
                ?: return failToStart(startId, "Retry request did not include a step ID")
        } else {
            null
        }
        val requestedMode = intent?.getStringExtra(EXTRA_MODE)?.let(CharacterizationMode::fromWireValue)
        benchmarkJob = serviceScope.launch {
            executeSuite(startId, retryStepId, requestedMode)
        }
        return START_NOT_STICKY
    }

    private fun failToStart(startId: Int, message: String): Int {
        notifyStatus(STATUS_FAILED, message)
        releaseBenchmarkWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf(startId)
        return START_NOT_STICKY
    }

    private suspend fun executeSuite(
        startId: Int,
        retryStepId: String?,
        requestedMode: CharacterizationMode?
    ) {
        val store = BenchmarkSuiteCheckpointStore(File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME))
        val telemetry = AndroidDeviceTelemetry(this)
        var checkpoint: BenchmarkSuiteCheckpoint? = null
        try {
            ensureWorkerDisconnected()
            var current = if (retryStepId != null) {
                store.recoverInterrupted()
            } else {
                val existing = store.readOrCreate(requestedMode ?: CharacterizationMode.QUICK)
                if (requestedMode != null && existing.mode != requestedMode) {
                    store.reset(requestedMode)
                } else {
                    store.recoverInterrupted(existing.mode)
                }
            }
            checkpoint = current
            if (retryStepId != null) {
                val id = retryStepId
                check(BenchmarkSuitePlan.step(id, current.mode) != null) { "Unknown benchmark step: $id" }
                val step = current.steps.first { it.definition.id == id }
                check(step.latestAttempt?.status in setOf(
                    BenchmarkStepStatus.FAILED,
                    BenchmarkStepStatus.CANCELLED,
                    BenchmarkStepStatus.SKIPPED
                )) { "Step $id is not in a retryable state" }
            }

            val modelManager = RequiredModelManager(this)
            val manifest = modelManager.loadManifest()
            check(manifest.modelId == ModelManifest.PINNED_MODEL_ID) {
                "Installed model manifest does not match the pinned benchmark model"
            }
            check(modelManager.isInstalled(manifest)) {
                "The verified Qwen2.5-3B model is not installed; download it before benchmarking."
            }
            check(NativeRuntime.isAvailable()) {
                "Native engine unavailable: ${NativeRuntime.unavailableReason()}"
            }

            current = current.copy(
                status = "RUNNING",
                device = telemetry.deviceProfile()
            )
            checkpoint = current
            store.write(current)
            notifyStatus(STATUS_RUNNING, "Preparing native model and device capabilities", current)

            nativeExecutor = NativeLayerRangeShardExecutor(this)
            nativeExecutor?.prepareModel()
            current = uploadPendingAttempts(store, current)
            checkpoint = current

            val steps = if (retryStepId != null) {
                listOf(checkNotNull(current.steps.firstOrNull {
                    it.definition.id == retryStepId
                }))
            } else {
                BenchmarkSuitePlan.resumableSteps(current)
            }

            for (candidate in steps) {
                if (cancelRequested) break
                val step = current.steps.first { it.definition.id == candidate.definition.id }
                current = beginAttempt(current, step)
                checkpoint = current
                store.write(current)
                notifyStatus(
                    STATUS_RUNNING,
                    "${step.definition.profile.displayName()}: ${step.definition.durationMs / 1000}s ${step.definition.measurementPhase}",
                    current
                )

                val attemptId = checkNotNull(
                    current.steps.first { it.definition.id == step.definition.id }
                        .latestAttempt?.id
                )
                val stepOutcome = try {
                    runStep(step, attemptId, current, store, telemetry, manifest)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    Log.e(TAG, "Benchmark step ${step.definition.id} failed", error)
                    StepOutcome(
                        status = BenchmarkStepStatus.FAILED,
                        generatedTokens = 0,
                        result = failureResult(step, error),
                        errorReason = exceptionReason(error)
                    )
                }

                current = finishAttempt(current, step.definition.id, stepOutcome)
                checkpoint = current
                store.write(current)
                current = uploadAttempt(store, current, step.definition.id, attemptId)
                checkpoint = current
                store.write(current)
                notifyStatus(
                    STATUS_RUNNING,
                    "${step.definition.id}: ${stepOutcome.status.name.lowercase()}",
                    current
                )
            }

            current = current.copy(
                status = if (cancelRequested) "CANCELLED" else BenchmarkSuiteCheckpointStore.statusFor(current.steps)
            )
            checkpoint = current
            store.write(current)
            persistReadiness(current)
            current = uploadSuiteAndTests(store, current)
            checkpoint = current
            val terminalMessage = when (current.status) {
                "COMPLETED" -> "${current.mode.wireValue.replaceFirstChar(Char::uppercase)} benchmark completed."
                "COMPLETED_WITH_FAILURES" -> "Benchmark stages finished; failed or skipped steps can be retried individually."
                "CANCELLED" -> "Benchmark paused. Completed results are saved; resume to continue."
                else -> "Benchmark paused with pending steps."
            }
            notifyStatus(
                if (current.status == "CANCELLED" || current.status == "PENDING") STATUS_FAILED else STATUS_COMPLETED,
                terminalMessage,
                current,
                if (current.status == "COMPLETED" || current.status == "COMPLETED_WITH_FAILURES") 100 else null
            )
        } catch (cancelled: CancellationException) {
            checkpoint?.let { current ->
                val stopped = current.copy(
                    status = "CANCELLED",
                    updatedAt = Instant.now().toString()
                )
                runCatching { store.write(stopped) }
                    .onFailure { Log.e(TAG, "Could not persist interrupted benchmark state", it) }
                notifyStatus(STATUS_FAILED, "Benchmark interrupted; resume to continue.", stopped)
            }
            throw cancelled
        } catch (error: Exception) {
            Log.e(TAG, "Resumable benchmark could not run", error)
            val reason = exceptionReason(error)
            if (checkpoint != null) {
                try {
                    val failedStep = checkpoint.steps.firstOrNull {
                        it.definition.id == retryStepId &&
                            it.latestAttempt?.status in setOf(
                                BenchmarkStepStatus.FAILED,
                                BenchmarkStepStatus.CANCELLED,
                                BenchmarkStepStatus.SKIPPED,
                                BenchmarkStepStatus.RUNNING
                            )
                    } ?: BenchmarkSuitePlan.resumableSteps(checkpoint).firstOrNull()
                    if (failedStep != null) {
                        val existingAttempt = failedStep.latestAttempt
                        val runningAttemptId = existingAttempt
                            ?.takeIf { it.status == BenchmarkStepStatus.RUNNING }
                            ?.id
                        var failedCheckpoint = checkpoint
                        val attemptId = if (runningAttemptId != null) {
                            runningAttemptId
                        } else {
                            failedCheckpoint = beginAttempt(failedCheckpoint, failedStep)
                            failedCheckpoint.steps.first {
                                it.definition.id == failedStep.definition.id
                            }.latestAttempt!!.id
                        }
                        val outcome = StepOutcome(
                            BenchmarkStepStatus.FAILED,
                            0,
                            mapOf(
                                "step_id" to failedStep.definition.id,
                                "profile" to failedStep.definition.profile.wireValue,
                                "configured_backend" to failedStep.definition.profile.configuredBackend,
                                "requested_tokens" to failedStep.definition.generatedTokens,
                                "failure_reason" to reason
                            ),
                            reason
                        )
                        failedCheckpoint = finishAttempt(
                            failedCheckpoint,
                            failedStep.definition.id,
                            outcome
                        )
                        store.write(failedCheckpoint)
                        failedCheckpoint = uploadAttempt(
                            store,
                            failedCheckpoint,
                            failedStep.definition.id,
                            attemptId
                        )
                        checkpoint = failedCheckpoint.copy(
                            status = BenchmarkSuiteCheckpointStore.statusFor(failedCheckpoint.steps)
                        )
                        store.write(checkpoint)
                    }
                } catch (persistError: Exception) {
                    Log.e(TAG, "Could not persist or upload benchmark setup failure", persistError)
                }
            }
            notifyStatus(STATUS_FAILED, reason, checkpoint)
        } finally {
            nativeExecutor?.close()
            nativeExecutor = null
            releaseBenchmarkWakeLock()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
        }
    }

    private suspend fun runStep(
        step: BenchmarkSuiteStep,
        attemptId: String,
        initialCheckpoint: BenchmarkSuiteCheckpoint,
        store: BenchmarkSuiteCheckpointStore,
        telemetry: AndroidDeviceTelemetry,
        manifest: ModelManifest
    ): StepOutcome {
        val definition = step.definition
        val executor = checkNotNull(nativeExecutor)
        val resolvedProfile = definition.profile
        val resolvedThreads = definition.graphThreads
        if (definition.kind == BenchmarkStepKind.LLAMA_CPP_BASELINE) {
            return runLlamaCppBaseline(step, attemptId, initialCheckpoint, store, telemetry, manifest)
        }
        val gpuInfo = initialCheckpoint.device["gpu"] as? Map<*, *>
        val vulkanAvailable = gpuInfo?.get("vulkan_available") == true &&
            gpuInfo["ggml_backend_compiled"] == true &&
            gpuInfo["inference_backend_enabled"] == true
        if (resolvedProfile.vulkanShardIds.isNotEmpty() && !vulkanAvailable) {
            val skipReason = gpuInfo?.get("inference_skip_reason") as? String
                ?: gpuInfo?.get("reason") as? String
                ?: "Vulkan is not available in this device/native build."
            return StepOutcome(
                BenchmarkStepStatus.SKIPPED,
                0,
                mapOf(
                    "step_id" to definition.id,
                    "profile" to resolvedProfile.wireValue,
                    "benchmark_category" to definition.kind.name.lowercase(),
                    "configured_backend" to resolvedProfile.configuredBackend,
                    "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                    "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                    "graph_threads" to resolvedThreads,
                    "measurement_phase" to definition.measurementPhase,
                    "requested_tokens" to definition.generatedTokens,
                    "observed_vulkan_shards" to emptySet<String>(),
                    "skip_reason" to skipReason
                ),
                skipReason
            )
        }

        val gpuWeightBytes = resolvedProfile.vulkanShardIds.sumOf { shardId ->
            val range = QwenShardCatalog.ranges.first { it.shardId == shardId }
            executor.layerWeightBytes(range.layerStart, range.layerEnd)
        }
        if (resolvedProfile.vulkanShardIds.isNotEmpty()) {
            val availableRamMb = telemetry.snapshot().memory["ram_available_mb"]
            val memorySkipReason = BenchmarkMemorySafety.vulkanSkipReason(
                availableRamMb,
                manifest.sizeBytes
            )
            if (memorySkipReason != null) {
                return StepOutcome(
                    BenchmarkStepStatus.SKIPPED,
                    0,
                    mapOf(
                        "step_id" to definition.id,
                        "profile" to resolvedProfile.wireValue,
                        "benchmark_category" to definition.kind.name.lowercase(),
                        "configured_backend" to resolvedProfile.configuredBackend,
                        "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                        "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                        "graph_threads" to resolvedThreads,
                        "measurement_phase" to definition.measurementPhase,
                        "requested_tokens" to definition.generatedTokens,
                        "gpu_weight_bytes" to gpuWeightBytes,
                        "available_ram_mb" to availableRamMb,
                        "skip_reason" to memorySkipReason
                    ),
                    memorySkipReason
                )
            }
        }
        val freeVulkanBytes = (gpuInfo?.get("vulkan_free_memory_bytes") as? Number)?.toLong()
        if (resolvedProfile.vulkanShardIds.isNotEmpty() &&
            (freeVulkanBytes == null || gpuWeightBytes * 4L > freeVulkanBytes * 3L)
        ) {
            val reason = if (freeVulkanBytes == null) {
                "Vulkan free memory is unavailable; cannot enforce 25% headroom"
            } else {
                "Vulkan placement would leave less than 25% GPU memory headroom"
            }
            return StepOutcome(
                BenchmarkStepStatus.SKIPPED,
                0,
                mapOf(
                    "step_id" to definition.id,
                    "profile" to resolvedProfile.wireValue,
                    "benchmark_category" to definition.kind.name.lowercase(),
                    "configured_backend" to resolvedProfile.configuredBackend,
                    "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                    "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                    "graph_threads" to resolvedThreads,
                    "measurement_phase" to definition.measurementPhase,
                    "requested_tokens" to definition.generatedTokens,
                    "gpu_weight_bytes" to gpuWeightBytes,
                    "skip_reason" to reason
                ),
                reason
            )
        }

        executor.configureGraphThreads(resolvedThreads)
        executor.configureShardBackends(resolvedProfile.vulkanShardIds)
        val benchmarkStartedAt = SystemClock.elapsedRealtime()
        val samples = Collections.synchronizedList(mutableListOf(telemetry.snapshot()))
        val progressPhase = AtomicReference("warm-up prefill")
        val sampler = serviceScope.launch {
            while (isActive) {
                delay(TELEMETRY_SAMPLE_INTERVAL_MS)
                samples += telemetry.snapshot()
                updateRunningProgress(
                    store,
                    step.definition.id,
                    attemptId,
                    currentGeneratedTokens(store, step.definition.id, attemptId),
                    SystemClock.elapsedRealtime() - benchmarkStartedAt,
                    ((SystemClock.elapsedRealtime() - benchmarkStartedAt) * 100 /
                        definition.durationMs).toInt().coerceIn(0, 100),
                    phase = progressPhase.get()
                )
            }
        }
        var lastPersistedTokens = 0
        try {
            val engine = NativeShardBenchmarkEngine(executor)
            val warmup = engine.run(
                prefillTokens = definition.prefillTokens,
                generatedTokens = 1,
                stopAfterMs = definition.durationMs,
                shouldStop = { cancelRequested }
            )
            if (cancelRequested) {
                return StepOutcome(BenchmarkStepStatus.CANCELLED, 0, emptyMap(), "Cancelled after warm-up")
            }
            if (warmup.safetyLimitReached) {
                val reason = "Safety time limit reached during Vulkan/CPU warm-up prefill."
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    0,
                    mapOf(
                        "step_id" to definition.id,
                        "profile" to resolvedProfile.wireValue,
                        "benchmark_category" to definition.kind.name.lowercase(),
                        "configured_backend" to resolvedProfile.configuredBackend,
                        "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                        "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                        "graph_threads" to resolvedThreads,
                        "measurement_phase" to definition.measurementPhase,
                        "requested_tokens" to definition.generatedTokens,
                        "elapsed_ms" to (SystemClock.elapsedRealtime() - benchmarkStartedAt),
                        "failure_reason" to reason
                    ),
                    reason
                )
            }
            val remainingStepTimeMs =
                definition.durationMs - (SystemClock.elapsedRealtime() - benchmarkStartedAt)
            if (remainingStepTimeMs <= 0L) {
                val reason = "Safety time limit reached during warm-up; measurement did not start."
                return StepOutcome(BenchmarkStepStatus.FAILED, 0, mapOf(
                    "step_id" to definition.id,
                    "profile" to resolvedProfile.wireValue,
                    "benchmark_category" to definition.kind.name.lowercase(),
                    "configured_backend" to resolvedProfile.configuredBackend,
                    "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                    "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                    "graph_threads" to resolvedThreads,
                    "measurement_phase" to definition.measurementPhase,
                    "requested_tokens" to definition.generatedTokens,
                    "elapsed_ms" to (SystemClock.elapsedRealtime() - benchmarkStartedAt),
                    "failure_reason" to reason
                ), reason)
            }
            progressPhase.set("measurement")
            val result = engine.run(
                prefillTokens = definition.prefillTokens,
                generatedTokens = definition.generatedTokens,
                stopAfterMs = remainingStepTimeMs,
                shouldStop = { cancelRequested },
                onPassCompleted = { generated, _ ->
                    val completedTokens = generated.coerceIn(
                        0,
                        definition.generatedTokens
                    )
                    if (completedTokens - lastPersistedTokens >= PROGRESS_SAVE_INTERVAL_TOKENS
                    ) {
                        lastPersistedTokens = completedTokens
                        val elapsed = SystemClock.elapsedRealtime() - benchmarkStartedAt
                        updateRunningProgress(
                            store,
                            definition.id,
                            attemptId,
                            completedTokens,
                            elapsed,
                            (elapsed * 100 / definition.durationMs).toInt().coerceIn(0, 100),
                            progressPhase.get()
                        )
                    }
                }
            )
            samples += telemetry.snapshot()
            val observedVulkan = executor.usedVulkanShardIds()
            val backendStatistics = executor.backendStatistics()
            val missingVulkan = resolvedProfile.vulkanShardIds - observedVulkan
            val unexpectedVulkan = observedVulkan - resolvedProfile.vulkanShardIds
            val unverifiedFullGpuShards = if (resolvedProfile == ShardPlacementProfile.VULKAN_ALL) {
                resolvedProfile.vulkanShardIds.filter { shardId ->
                    backendStatistics[shardId]?.get("full_gpu_verified") != true
                }
            } else emptyList()
            val telemetryResult = telemetrySummary(samples.toList())
            val baseResult = mapOf(
                "step_id" to definition.id,
                "profile" to resolvedProfile.wireValue,
                "benchmark_category" to definition.kind.name.lowercase(),
                "configured_backend" to resolvedProfile.configuredBackend,
                "requested_vulkan_shards" to resolvedProfile.vulkanShardIds,
                "requested_gpu_layers" to gpuLayerCount(resolvedProfile),
                "graph_threads" to resolvedThreads,
                "measurement_phase" to definition.measurementPhase,
                "target_duration_ms" to definition.durationMs,
                "gpu_weight_bytes" to gpuWeightBytes,
                "observed_vulkan_shards" to observedVulkan,
                "missing_vulkan_shards" to missingVulkan,
                "unexpected_vulkan_shards" to unexpectedVulkan,
                "backend_statistics" to backendStatistics,
                "prefill_tokens" to result.prefillTokens,
                "generated_tokens" to result.generatedTokens,
                "total_time_ms" to result.totalTimeMs,
                "tokens_per_second" to result.tokensPerSecond,
                "prefill_tokens_per_second" to result.prefillSpeedTokensPerSecond,
                "generation_tokens_per_second" to result.generationSpeedTokensPerSecond,
                "decode_latency_ms" to result.decodePassLatenciesNanos.map { it / NANOS_PER_MILLISECOND },
                "thermal" to telemetryResult.thermal,
                "memory" to telemetryResult.memory,
                "cpu_frequency" to telemetryResult.cpuFrequencies,
                "device" to initialCheckpoint.device,
                "actual_inference_backend" to when {
                    observedVulkan.isEmpty() -> "CPU"
                    observedVulkan.size == QwenShardCatalog.ranges.size -> "VULKAN"
                    else -> "HYBRID"
                },
                "observed_gpu_layers" to observedVulkan.sumOf { shardId ->
                    val range = QwenShardCatalog.ranges.first { it.shardId == shardId }
                    range.layerEnd - range.layerStart + 1
                }
            )
            if (cancelRequested) {
                return StepOutcome(
                    BenchmarkStepStatus.CANCELLED,
                    result.generatedTokens,
                    baseResult,
                    "Cancelled after ${result.generatedTokens} measured decode tokens."
                )
            }
            if (result.generatedTokens == 0) {
                val reason = if (result.safetyLimitReached) {
                    "Safety time limit reached during prefill before any decode token completed."
                } else {
                    "No decode token completed during the measurement window."
                }
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    0,
                    baseResult,
                    reason
                )
            }
            if (result.safetyLimitReached || result.generatedTokens != definition.generatedTokens) {
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    result.generatedTokens,
                    baseResult + ("failure_reason" to
                        "Safety time limit reached after ${result.generatedTokens}/${definition.generatedTokens} tokens."),
                    "Safety time limit reached after ${result.generatedTokens}/${definition.generatedTokens} tokens."
                )
            }
            if (missingVulkan.isNotEmpty() || unexpectedVulkan.isNotEmpty()) {
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    result.generatedTokens,
                    baseResult,
                    "Requested Vulkan shards ${resolvedProfile.vulkanShardIds.sorted()} but observed ${observedVulkan.sorted()}."
                )
            }
            if (unverifiedFullGpuShards.isNotEmpty()) {
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    result.generatedTokens,
                    baseResult,
                    "Full-GPU verification failed for shards ${unverifiedFullGpuShards.joinToString()}; CPU fallback or host KV residency was observed."
                )
            }
            return StepOutcome(BenchmarkStepStatus.COMPLETED, result.generatedTokens, baseResult)
        } finally {
            sampler.cancelAndJoin()
        }
    }

    private suspend fun runLlamaCppBaseline(
        step: BenchmarkSuiteStep,
        attemptId: String,
        checkpoint: BenchmarkSuiteCheckpoint,
        store: BenchmarkSuiteCheckpointStore,
        telemetry: AndroidDeviceTelemetry,
        manifest: ModelManifest
    ): StepOutcome {
        val definition = step.definition
        val modelManager = RequiredModelManager(this)
        val modelFile = modelManager.installedFile(manifest)
        val memoryBefore = telemetry.snapshot().memory["ram_available_mb"]
            ?: return skippedBaseline("Available RAM could not be measured safely.")
        val requiredRamMb = BenchmarkMemorySafety.requiredAvailableRamMb(manifest.sizeBytes)
        if (memoryBefore < requiredRamMb) {
            return skippedBaseline(
                "Baseline skipped for device safety: available RAM ${memoryBefore} MB is below " +
                    "the estimated ${requiredRamMb} MB requirement."
            )
        }

        val samples = Collections.synchronizedList(mutableListOf(telemetry.snapshot()))
        val benchmarkStartedAt = SystemClock.elapsedRealtime()
        val sampler = serviceScope.launch {
            while (isActive) {
                delay(TELEMETRY_SAMPLE_INTERVAL_MS)
                samples += telemetry.snapshot()
                updateRunningProgress(
                    store,
                    definition.id,
                    attemptId,
                    currentGeneratedTokens(store, definition.id, attemptId),
                    SystemClock.elapsedRealtime() - benchmarkStartedAt,
                    ((SystemClock.elapsedRealtime() - benchmarkStartedAt) * 100 /
                        definition.durationMs).toInt().coerceIn(0, 100),
                    "native baseline"
                )
            }
        }
        try {
            val nativeJson = NativeBridge.nativeRunLlamaCppBaseline(
                modelFile.absolutePath,
                definition.graphThreads,
                definition.prefillTokens,
                definition.generatedTokens,
                definition.durationMs
            )
            val native = JsonParser.parseString(nativeJson).asJsonObject
            val generated = native.get("generated_tokens").asInt
            val prefillMs = native.get("prefill_time_ms").asDouble
            val decodeMs = native.get("decode_time_ms").asDouble
            samples += telemetry.snapshot()
            val telemetryResult = telemetrySummary(samples.toList())
            val latencies = native.getAsJsonArray("decode_latency_ms").map { it.asDouble }
            val decodeTokensPerSecond = if (decodeMs > 0.0) generated * 1000.0 / decodeMs else 0.0
            val result = mapOf(
                "step_id" to definition.id,
                "benchmark_category" to "llama_cpp_baseline",
                "profile" to "llama_cpp_baseline",
                "configured_backend" to "llama.cpp-cpu",
                "actual_inference_backend" to "LLAMA_CPP_CPU",
                "graph_threads" to native.get("cpu_threads").asInt,
                "requested_gpu_layers" to 0,
                "observed_gpu_layers" to 0,
                "requested_vulkan_shards" to emptyList<String>(),
                "observed_vulkan_shards" to emptyList<String>(),
                "measurement_phase" to definition.measurementPhase,
                "target_duration_ms" to definition.durationMs,
                "requested_tokens" to definition.generatedTokens,
                "prefill_tokens" to definition.prefillTokens,
                "generated_tokens" to generated,
                "total_time_ms" to native.get("total_time_ms").asLong,
                "tokens_per_second" to (definition.prefillTokens + generated) * 1000.0 /
                    native.get("total_time_ms").asDouble,
                "prefill_tokens_per_second" to if (prefillMs > 0.0) {
                    definition.prefillTokens * 1000.0 / prefillMs
                } else 0.0,
                "generation_tokens_per_second" to decodeTokensPerSecond,
                "prefill_time_ms" to prefillMs,
                "decode_time_ms" to decodeMs,
                "decode_latency_ms" to latencies,
                "memory" to telemetryResult.memory,
                "thermal" to telemetryResult.thermal,
                "cpu_frequency" to telemetryResult.cpuFrequencies,
                "backend_statistics" to mapOf("llama_cpp_baseline" to true),
                "device" to checkpoint.device
            )
            if (generated == 0) {
                return StepOutcome(
                    BenchmarkStepStatus.FAILED,
                    0,
                    result,
                    "llama.cpp baseline completed no measured decode tokens."
                )
            }
            if (generated != definition.generatedTokens) {
                val reason = "Safety time limit reached after $generated/${definition.generatedTokens} tokens."
                return StepOutcome(BenchmarkStepStatus.FAILED, generated, result, reason)
            }
            return StepOutcome(BenchmarkStepStatus.COMPLETED, generated, result)
        } finally {
            sampler.cancelAndJoin()
        }
    }

    private fun skippedBaseline(reason: String) = StepOutcome(
        BenchmarkStepStatus.SKIPPED,
        0,
        mapOf(
            "benchmark_category" to "llama_cpp_baseline",
            "profile" to "llama_cpp_baseline",
            "configured_backend" to "llama.cpp-cpu",
            "graph_threads" to 4,
            "requested_gpu_layers" to 0,
            "skip_reason" to reason
        ),
        reason
    )

    private fun gpuLayerCount(profile: ShardPlacementProfile): Int =
        profile.vulkanShardIds.sumOf { shardId ->
            val range = QwenShardCatalog.ranges.first { it.shardId == shardId }
            range.layerEnd - range.layerStart + 1
        }

    private fun persistReadiness(checkpoint: BenchmarkSuiteCheckpoint) {
        if (checkpoint.status !in setOf("COMPLETED", "COMPLETED_WITH_FAILURES")) return
        val completed = checkpoint.steps.mapNotNull { step ->
            step.latestAttempt?.takeIf { it.status == BenchmarkStepStatus.COMPLETED }
                ?.let { step.definition to it.result }
        }
        val plans = completed.mapNotNull { (definition, result) ->
            if (definition.kind == BenchmarkStepKind.LLAMA_CPP_BASELINE ||
                definition.generatedTokens != BenchmarkSuitePlan.generatedTokenTargets.last()
            ) return@mapNotNull null
            val phase = result["measurement_phase"] as? String ?: return@mapNotNull null
            if (phase != "screening") return@mapNotNull null
            val profile = (result["profile"] as? String)
                ?.let(ShardPlacementProfile::fromWireValue) ?: return@mapNotNull null
            val memory = result["memory"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
            val thermal = result["thermal"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
            val beforeRam = (memory["ram_available_before_mb"] as? Number)?.toLong() ?: 0L
            val minRam = (memory["ram_available_min_mb"] as? Number)?.toLong() ?: beforeRam
            val beforeTemp = (thermal["temperature_before_c"] as? Number)?.toDouble()
            val peakTemp = (thermal["temperature_peak_c"] as? Number)?.toDouble()
            ShardPlacementPlan(
                profileId = profile.wireValue,
                cpuThreads = (result["graph_threads"] as Number).toInt(),
                sustainedGenerationTokensPerSecond =
                    (result["generation_tokens_per_second"] as Number).toDouble(),
                additionalMemoryMb = (beforeRam - minRam).coerceAtLeast(0L),
                gpuWeightBytes = (result["gpu_weight_bytes"] as? Number)?.toLong() ?: 0L,
                thermalRiseC = if (beforeTemp != null && peakTemp != null) {
                    (peakTemp - beforeTemp).coerceAtLeast(0.0)
                } else null,
                thermalStatusPeak = thermal["thermal_status_peak"] as? String,
                measurementPhase = definition.measurementPhase
            )
        }
        val representative = completed
            .filter { (definition, _) -> definition.kind != BenchmarkStepKind.LLAMA_CPP_BASELINE }
            .maxByOrNull { (_, result) ->
            (result["generation_tokens_per_second"] as? Number)?.toDouble() ?: 0.0
        } ?: return
        val result = representative.second
        val readiness = BenchmarkReadiness.completed(
            prefillTokens = (result["prefill_tokens"] as Number).toInt(),
            generatedTokens = (result["generated_tokens"] as Number).toInt(),
            totalTimeMs = (result["total_time_ms"] as Number).toLong(),
            tokensPerSecond = (result["tokens_per_second"] as Number).toDouble(),
            prefillSpeedTokensPerSecond =
                (result["prefill_tokens_per_second"] as Number).toDouble(),
            generationSpeedTokensPerSecond =
                (result["generation_tokens_per_second"] as Number).toDouble(),
            layerRanges = QwenShardCatalog.ranges,
            benchmarkMode = checkpoint.mode.wireValue,
            benchmarkVersion = HYBRID_BENCHMARK_VERSION,
            selectedPlacement = null,
            verifiedPlacements = plans
        )
        BenchmarkReadinessStore(File(filesDir, BenchmarkReadinessStore.FILE_NAME)).write(readiness)
    }

    private fun beginAttempt(
        checkpoint: BenchmarkSuiteCheckpoint,
        step: BenchmarkSuiteStep
    ): BenchmarkSuiteCheckpoint = checkpoint.copy(
        status = "RUNNING",
        steps = checkpoint.steps.map { current ->
            if (current.definition.id != step.definition.id) current
            else current.copy(
                attempts = current.attempts + BenchmarkStepAttempt(
                    status = BenchmarkStepStatus.RUNNING,
                    uploadStatus = BenchmarkUploadStatus.NOT_REQUIRED
                )
            )
        }
    )

    private fun finishAttempt(
        checkpoint: BenchmarkSuiteCheckpoint,
        stepId: String,
        outcome: StepOutcome
    ): BenchmarkSuiteCheckpoint = checkpoint.copy(
        steps = checkpoint.steps.map { step ->
            if (step.definition.id != stepId) step else {
                val attempts = step.attempts.toMutableList()
                val running = attempts.removeAt(attempts.lastIndex)
                step.copy(
                    attempts = attempts + running.copy(
                        status = outcome.status,
                        finishedAt = Instant.now().toString(),
                        generatedTokens = outcome.generatedTokens,
                        result = outcome.result,
                        errorReason = outcome.errorReason,
                        uploadStatus = BenchmarkUploadStatus.PENDING
                    )
                )
            }
        }
    )

    private fun updateRunningProgress(
        store: BenchmarkSuiteCheckpointStore,
        stepId: String,
        attemptId: String,
        generatedTokens: Int,
        elapsedOrProgress: Long,
        progress: Int? = null,
        phase: String? = null
    ) {
        try {
            synchronized(progressLock) {
                val checkpoint = store.readOrCreate()
                val updated = checkpoint.copy(
                    steps = checkpoint.steps.map { step ->
                        if (step.definition.id != stepId) step else step.copy(
                            attempts = step.attempts.map { attempt ->
                                if (attempt.id == attemptId) attempt.copy(
                                    generatedTokens = generatedTokens,
                                    result = attempt.result + mapOf(
                                        "elapsed_ms" to elapsedOrProgress,
                                        "progress_percent" to progress,
                                        "stage" to phase
                                    )
                                ) else attempt
                            }
                        )
                    }
                )
                store.write(updated)
                val completedSteps = updated.steps.count {
                    it.latestAttempt?.status in setOf(
                        BenchmarkStepStatus.COMPLETED,
                        BenchmarkStepStatus.SKIPPED
                    )
                }
                val stageProgress = progress
                    ?: (generatedTokens * 100 / updated.steps.first {
                        it.definition.id == stepId
                    }.definition.generatedTokens)
                val suiteProgress = ((completedSteps * 100) +
                    (stageProgress.coerceIn(0, 100) * 100 / updated.steps.size)) /
                    updated.steps.size
                notifyStatus(
                    STATUS_RUNNING,
                    "$stepId: ${phase ?: "running"}, generated $generatedTokens tokens",
                    updated,
                    suiteProgress
                )
            }
        } catch (error: Exception) {
            Log.e(TAG, "Could not save or broadcast benchmark progress for $stepId", error)
        }
    }

    private fun currentGeneratedTokens(
        store: BenchmarkSuiteCheckpointStore,
        stepId: String,
        attemptId: String
    ): Int = store.readOrCreate().steps.first { it.definition.id == stepId }
        .attempts.first { it.id == attemptId }.generatedTokens

    private suspend fun uploadAttempt(
        store: BenchmarkSuiteCheckpointStore,
        checkpoint: BenchmarkSuiteCheckpoint,
        stepId: String,
        attemptId: String
    ): BenchmarkSuiteCheckpoint {
        // Child rows reference the suite row, so profile uploads are deferred until
        // the complete suite row can be written atomically in ordering terms.
        return checkpoint
    }

    private suspend fun uploadPendingAttempts(
        store: BenchmarkSuiteCheckpointStore,
        initialCheckpoint: BenchmarkSuiteCheckpoint
    ): BenchmarkSuiteCheckpoint {
        val terminal = initialCheckpoint.copy(
            status = BenchmarkSuiteCheckpointStore.statusFor(initialCheckpoint.steps)
        )
        if (terminal.status !in setOf("COMPLETED", "COMPLETED_WITH_FAILURES")) return initialCheckpoint
        return uploadSuiteAndTests(store, terminal)
    }

    private suspend fun uploadSuiteAndTests(
        store: BenchmarkSuiteCheckpointStore,
        checkpoint: BenchmarkSuiteCheckpoint
    ): BenchmarkSuiteCheckpoint {
        if (checkpoint.status !in setOf("COMPLETED", "COMPLETED_WITH_FAILURES")) return checkpoint
        val uploader = SupabaseBenchmarkUploader()
        var updated = checkpoint
        return try {
            uploader.uploadSuite(supabaseSuiteRow(updated))
            updated.steps.forEach { step ->
                step.attempts.filter {
                    it.uploadStatus == BenchmarkUploadStatus.PENDING && it.result.isNotEmpty()
                }.forEach { attempt ->
                    try {
                        uploader.uploadTest(supabaseTestRow(updated, step.definition.id, attempt))
                        updated = updated.updateAttempt(step.definition.id, attempt.id) {
                            copy(uploadStatus = BenchmarkUploadStatus.UPLOADED, uploadError = null)
                        }
                        store.write(updated)
                    } catch (error: Exception) {
                        Log.e(TAG, "Supabase test upload failed; retaining this attempt for retry", error)
                        updated = updated.updateAttempt(step.definition.id, attempt.id) {
                            copy(uploadError = exceptionReason(error))
                        }
                        store.write(updated)
                        return updated
                    }
                }
            }
            updated
        } catch (error: Exception) {
            Log.e(TAG, "Supabase suite/profile upload failed; keeping rows queued locally", error)
            updated.copy(steps = updated.steps.map { step ->
                step.copy(attempts = step.attempts.map { attempt ->
                    if (attempt.uploadStatus == BenchmarkUploadStatus.PENDING) {
                        attempt.copy(uploadError = exceptionReason(error))
                    } else attempt
                })
            }).also(store::write)
        }
    }

    private fun supabaseSuiteRow(checkpoint: BenchmarkSuiteCheckpoint): Map<String, Any?> {
        val completed = checkpoint.steps.mapNotNull { it.latestAttempt }
            .filter { it.status == BenchmarkStepStatus.COMPLETED }
        val preferredPhase = "comparison"
        val representative = completed
            .filter { it.result["measurement_phase"] == preferredPhase }
            .maxByOrNull {
                (it.result["generation_tokens_per_second"] as? Number)?.toDouble() ?: 0.0
            } ?: completed.maxByOrNull {
                (it.result["generation_tokens_per_second"] as? Number)?.toDouble() ?: 0.0
            }
        val result = representative?.result.orEmpty()
        val profile = (result["profile"] as? String) ?: "cpu_only"
        val memory = result["memory"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val thermal = result["thermal"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        return mapOf(
            "client_run_id" to checkpoint.suiteId,
            "platform" to "android",
            "worker_id" to getOrCreateBenchmarkWorkerId(),
            "app_version" to BuildConfig.VERSION_NAME,
            "model_id" to ModelManifest.PINNED_MODEL_ID,
            "model_version" to ModelManifest.PINNED_MODEL_VERSION,
            "model_artifact_digest" to ModelManifest.PINNED_ARTIFACT_DIGEST,
            "device_model" to Build.MODEL,
            "os_version" to Build.VERSION.RELEASE,
            "backend" to profile,
            "execution_mode" to "single_device_all_shards",
            "benchmark_version" to HYBRID_BENCHMARK_VERSION,
            "benchmark_mode" to checkpoint.mode.wireValue,
            "selected_placement" to if (checkpoint.mode == CharacterizationMode.FULL) profile else null,
            "selected_cpu_threads" to ((result["graph_threads"] as? Number)?.toInt()),
            "prefill_tokens" to ((result["prefill_tokens"] as? Number)?.toInt() ?: 32),
            "generated_tokens" to (representative?.generatedTokens ?: 0),
            "total_time_ms" to ((result["total_time_ms"] as? Number)?.toLong() ?: 0L),
            "tokens_per_second" to ((result["tokens_per_second"] as? Number)?.toDouble() ?: 0.0),
            "prefill_tokens_per_second" to ((result["prefill_tokens_per_second"] as? Number)?.toDouble() ?: 0.0),
            "generation_tokens_per_second" to ((result["generation_tokens_per_second"] as? Number)?.toDouble() ?: 0.0),
            "peak_rss_mb" to memory["peak_app_rss_mb"],
            "peak_gpu_mb" to null,
            "initial_temperature_c" to thermal["temperature_before_c"],
            "peak_temperature_c" to thermal["temperature_peak_c"],
            "final_temperature_c" to thermal["temperature_after_c"],
            "raw_result" to gson.fromJson(gson.toJson(checkpoint), Any::class.java)
        )
    }

    private fun supabaseTestRow(
        checkpoint: BenchmarkSuiteCheckpoint,
        stepId: String,
        attempt: BenchmarkStepAttempt
    ): Map<String, Any?> {
        val step = checkpoint.steps.first { it.definition.id == stepId }
        val definition = step.definition
        val result = attempt.result
        val memory = result["memory"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val thermal = result["thermal"] as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val workerId = getOrCreateBenchmarkWorkerId()
        val actualProfile = (result["profile"] as? String) ?: definition.profile.wireValue
        val latencies = (result["decode_latency_ms"] as? List<*>)
            .orEmpty().mapNotNull { (it as? Number)?.toDouble() }.sorted()
        fun percentile(fraction: Double): Double? = if (latencies.isEmpty()) null else {
            val index = kotlin.math.ceil(latencies.size * fraction).toInt().coerceIn(1, latencies.size) - 1
            latencies[index]
        }
        return mapOf<String, Any?>(
            "client_test_id" to attempt.id,
            "suite_client_run_id" to checkpoint.suiteId,
            "platform" to "android",
            "worker_id" to workerId,
            "model_id" to ModelManifest.PINNED_MODEL_ID,
            "model_version" to ModelManifest.PINNED_MODEL_VERSION,
            "model_artifact_digest" to ModelManifest.PINNED_ARTIFACT_DIGEST,
            "profile" to actualProfile,
            "requested_backends" to mapOf(
                "vulkan_shards" to definition.profile.vulkanShardIds.sorted()
            ),
            "actual_backends" to mapOf(
                "vulkan_shards" to ((result["observed_vulkan_shards"] as? Collection<*>)
                    .orEmpty().map(Any?::toString).sorted()),
                "backend" to result["actual_inference_backend"]
            ),
            "requested_gpu_layers" to ((result["requested_gpu_layers"] as? Number)?.toInt() ?: 0),
            "observed_gpu_layers" to ((result["observed_gpu_layers"] as? Number)?.toInt() ?: 0),
            "cpu_threads" to ((result["graph_threads"] as? Number)?.toInt()
                ?: definition.graphThreads.coerceAtLeast(1)),
            "measurement_phase" to definition.measurementPhase,
            "duration_ms" to ((result["total_time_ms"] as? Number)?.toLong() ?: 0L),
            "generated_tokens" to attempt.generatedTokens,
            "throughput" to ((result["generation_tokens_per_second"] as? Number)?.toDouble() ?: 0.0),
            "latency_percentiles" to mapOf(
                "p50_ms" to percentile(0.50),
                "p95_ms" to percentile(0.95),
                "p99_ms" to percentile(0.99)
            ),
            "memory" to memory,
            "thermal" to thermal,
            "backend_statistics" to (result["backend_statistics"] ?: mapOf(
                "gpu_weight_bytes" to result["gpu_weight_bytes"],
                "observed_vulkan_shards" to result["observed_vulkan_shards"]
            )),
            "status" to attempt.status.name,
            "failure_reason" to attempt.errorReason,
            "raw_result" to mapOf(
                "step_id" to stepId,
                "attempt_id" to attempt.id,
                "result" to result
            )
        )
    }

    private fun getOrCreateBenchmarkWorkerId(): String {
        val preferences = getSharedPreferences(WorkerService.PREFERENCES, Context.MODE_PRIVATE)
        preferences.getString(WorkerService.KEY_WORKER_ID, null)
            ?.takeIf(String::isNotBlank)
            ?.let { return it }

        val workerId = UUID.randomUUID().toString()
        preferences.edit().putString(WorkerService.KEY_WORKER_ID, workerId).apply()
        return workerId
    }

    private fun failureResult(step: BenchmarkSuiteStep, error: Exception): Map<String, Any?> =
        mapOf(
            "step_id" to step.definition.id,
            "profile" to step.definition.profile.wireValue,
            "configured_backend" to step.definition.profile.configuredBackend,
            "requested_tokens" to step.definition.generatedTokens,
            "error_type" to error.javaClass.name,
            "failure_reason" to exceptionReason(error)
        )

    private fun telemetrySummary(samples: List<com.intellihive.worker.benchmark.BenchmarkTelemetrySample>): TelemetrySummary {
        val temperatures = samples.mapNotNull { it.batteryTemperatureC }
        val statuses = samples.mapNotNull { it.thermalStatus }
        val statusOrder = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")
        val allMemory = samples.map { it.memory }
        val rss = allMemory.mapNotNull { it["app_rss_mb"] }
        val pss = allMemory.mapNotNull { it["app_pss_mb"] }
        val nativeHeap = allMemory.mapNotNull { it["native_heap_allocated_mb"] }
        val javaHeap = allMemory.mapNotNull { it["java_heap_used_mb"] }
        val availableRam = allMemory.mapNotNull { it["ram_available_mb"] }
        val majorFaults = allMemory.mapNotNull { it["major_page_faults"] }
        val frequencies = samples.flatMap { it.cpuFrequencyKHz.entries }
            .groupBy({ it.key }, { it.value })
            .mapValues { (_, values) ->
                val sorted = values.sorted()
                mapOf<String, Any?>(
                    "min_khz" to sorted.minOrNull(),
                    "median_khz" to if (sorted.isEmpty()) null else {
                        if (sorted.size % 2 == 0) {
                            (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
                        } else {
                            sorted[sorted.size / 2].toDouble()
                        }
                    },
                    "max_khz" to sorted.maxOrNull()
                )
            }
        val thermal = mapOf(
            "temperature_before_c" to temperatures.firstOrNull(),
            "temperature_peak_c" to temperatures.maxOrNull(),
            "temperature_after_c" to temperatures.lastOrNull(),
            "thermal_status_before" to statuses.firstOrNull(),
            "thermal_status_peak" to statuses.maxByOrNull { statusOrder.indexOf(it).coerceAtLeast(0) },
            "thermal_status_after" to statuses.lastOrNull(),
            "sample_count" to samples.size,
            "status" to if (temperatures.isEmpty() && statuses.isEmpty()) "SKIPPED" else "AVAILABLE"
        )
        val memory = mapOf(
            "peak_app_rss_mb" to rss.maxOrNull(),
            "peak_app_pss_mb" to pss.maxOrNull(),
            "peak_native_heap_allocated_mb" to nativeHeap.maxOrNull(),
            "peak_java_heap_used_mb" to javaHeap.maxOrNull(),
            "ram_available_before_mb" to availableRam.firstOrNull(),
            "ram_available_min_mb" to availableRam.minOrNull(),
            "ram_available_after_mb" to availableRam.lastOrNull(),
            "major_page_faults_before" to majorFaults.firstOrNull(),
            "major_page_faults_after" to majorFaults.lastOrNull(),
            "major_page_faults_delta" to if (majorFaults.size > 1) {
                (majorFaults.last() - majorFaults.first()).coerceAtLeast(0L)
            } else {
                null
            },
            "sample_count" to allMemory.count { it.isNotEmpty() },
            "status" to if (allMemory.any { it.isNotEmpty() }) "AVAILABLE" else "SKIPPED"
        )
        return TelemetrySummary(thermal, memory, frequencies)
    }

    private fun ensureWorkerDisconnected() {
        val preferences = getSharedPreferences(WorkerService.PREFERENCES, Context.MODE_PRIVATE)
        check(!isAppServiceRunning(this, WorkerService::class.java) &&
            !preferences.getBoolean(WorkerService.KEY_CONNECTED, false) &&
            !preferences.getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        ) { "Disconnect the worker and wait for it to stop before running the benchmark." }
    }

    private fun acquireBenchmarkWakeLock() {
        val powerManager = checkNotNull(getSystemService(PowerManager::class.java)) {
            "Android PowerManager is unavailable"
        }
        benchmarkWakeLock?.takeIf { it.isHeld }?.let { return }
        benchmarkWakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$packageName:benchmark"
        ).apply {
            setReferenceCounted(false)
            acquire(WAKE_LOCK_TIMEOUT_MS)
        }
    }

    private fun releaseBenchmarkWakeLock() {
        benchmarkWakeLock?.let { wakeLock ->
            if (wakeLock.isHeld) wakeLock.release()
        }
        benchmarkWakeLock = null
    }

    private fun notifyStatus(
        status: String,
        message: String,
        checkpoint: BenchmarkSuiteCheckpoint? = null,
        progress: Int? = null
    ) {
        val completedCount = checkpoint?.steps?.count {
            it.latestAttempt?.status in setOf(BenchmarkStepStatus.COMPLETED, BenchmarkStepStatus.SKIPPED)
        } ?: 0
        val calculatedProgress = progress ?: checkpoint?.let {
            completedCount * 100 / it.steps.size
        } ?: 0
        sendBroadcast(
            Intent(ACTION_STATUS)
                .setPackage(packageName)
                .putExtra(EXTRA_STATUS, status)
                .putExtra(EXTRA_MESSAGE, message)
                .putExtra(EXTRA_PROGRESS, calculatedProgress)
                .apply {
                    checkpoint?.let { putExtra(EXTRA_METRICS, gson.toJson(it)) }
                }
        )
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, notification(message))
    }

    private fun notification(message: String): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "IntelHive benchmark", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val openActivity = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val cancelIntent = PendingIntent.getService(
            this,
            2,
            Intent(this, BenchmarkService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("IntelHive backend benchmark")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(openActivity)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Stop after current pass", cancelIntent)
            .build()
    }

    override fun onDestroy() {
        benchmarkJob?.cancel()
        benchmarkJob = null
        releaseBenchmarkWakeLock()
        nativeExecutor?.close()
        nativeExecutor = null
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun exceptionReason(error: Throwable): String {
        val messages = generateSequence(error) { it.cause }
            .mapNotNull { cause -> cause.message?.takeIf(String::isNotBlank) }
            .distinct()
            .toList()
        return if (messages.isEmpty()) error.javaClass.simpleName else messages.joinToString(" → ")
    }

    private fun BenchmarkSuiteCheckpoint.updateAttempt(
        stepId: String,
        attemptId: String,
        transform: BenchmarkStepAttempt.() -> BenchmarkStepAttempt
    ): BenchmarkSuiteCheckpoint = copy(
        steps = steps.map { step ->
            if (step.definition.id != stepId) step else step.copy(
                attempts = step.attempts.map { attempt ->
                    if (attempt.id == attemptId) attempt.transform() else attempt
                }
            )
        }
    )

    private data class StepOutcome(
        val status: BenchmarkStepStatus,
        val generatedTokens: Int,
        val result: Map<String, Any?>,
        val errorReason: String? = null
    )

    private data class TelemetrySummary(
        val thermal: Map<String, Any?>,
        val memory: Map<String, Any?>,
        val cpuFrequencies: Map<String, Map<String, Any?>>
    )

    companion object {
        const val ACTION_STATUS = "com.intellihive.worker.BENCHMARK_STATUS"
        const val ACTION_CANCEL = "com.intellihive.worker.CANCEL_BENCHMARK"
        const val ACTION_START_STAGED = "com.intellihive.worker.START_STAGED_BENCHMARK"
        const val ACTION_RETRY_STEP = "com.intellihive.worker.RETRY_BENCHMARK_STEP"
        const val EXTRA_STATUS = "benchmark_status"
        const val EXTRA_MESSAGE = "benchmark_message"
        const val EXTRA_METRICS = "benchmark_metrics"
        const val EXTRA_PROGRESS = "benchmark_progress"
        const val EXTRA_STEP_ID = "benchmark_step_id"
        const val EXTRA_MODE = "benchmark_mode"
        const val STATUS_RUNNING = "RUNNING"
        const val STATUS_COMPLETED = "COMPLETED"
        const val STATUS_FAILED = "FAILED"

        private const val TAG = "BenchmarkService"
        private const val CHANNEL_ID = "native_benchmark"
        private const val NOTIFICATION_ID = 1102
        private const val TELEMETRY_SAMPLE_INTERVAL_MS = 30_000L
        private const val PROGRESS_SAVE_INTERVAL_TOKENS = 4
        private const val NANOS_PER_MILLISECOND = 1_000_000.0
        private const val BYTES_PER_MB = 1024L * 1024L
        private const val BASELINE_RAM_HEADROOM_MB = 512L
        private const val WAKE_LOCK_TIMEOUT_MS = 2 * 60 * 60 * 1_000L
    }
}
