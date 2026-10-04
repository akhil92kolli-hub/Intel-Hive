package com.intellihive.worker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.app.ActivityManager
import android.app.AlertDialog
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.intellihive.worker.model.ModelDownloadState
import com.intellihive.worker.model.ModelManifest
import com.intellihive.worker.model.RequiredModelManager
import com.intellihive.worker.benchmark.BenchmarkReadinessStore
import com.intellihive.worker.benchmark.BenchmarkStatus
import com.intellihive.worker.benchmark.BenchmarkSuiteCheckpoint
import com.intellihive.worker.benchmark.BenchmarkSuiteCheckpointStore
import com.intellihive.worker.benchmark.BenchmarkStepStatus
import com.intellihive.worker.benchmark.CharacterizationMode
import com.intellihive.worker.benchmark.displayName
import com.google.gson.Gson
import com.intellihive.worker.service.BenchmarkService
import com.intellihive.worker.service.WorkerService
import com.intellihive.worker.service.isAppServiceRunning
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var workerStatusText: TextView
    private lateinit var metricsText: TextView
    private lateinit var benchmarkButton: Button
    private lateinit var benchmarkResetButton: Button
    private lateinit var benchmarkSteps: LinearLayout
    private lateinit var workerConnectionButton: Button
    private lateinit var schedulerUrl: EditText
    private lateinit var progressBar: ProgressBar
    private lateinit var modelStatusText: TextView
    private lateinit var modelDescriptionText: TextView
    private lateinit var modelDownloadButton: Button
    private lateinit var modelWifiOnly: CheckBox
    private lateinit var modelProgressBar: ProgressBar
    private lateinit var modelManager: RequiredModelManager
    private var modelManifest: ModelManifest? = null
    private var modelReady = false
    private var modelDownloadJob: Job? = null
    private var receiverRegistered = false
    private var benchmarkReceiverRegistered = false

    private val workerStatusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == WorkerService.ACTION_STATUS) {
                workerStatusText.text = intent.getStringExtra(WorkerService.EXTRA_STATUS)
                    ?: "Worker status unavailable"
                if (intent.hasExtra(WorkerService.EXTRA_CONNECTED)) {
                    val connected = intent.getBooleanExtra(WorkerService.EXTRA_CONNECTED, false)
                    if (connected) {
                        statusText.text = "Disconnect the worker before running a benchmark"
                    }
                    getSharedPreferences("worker_runtime", MODE_PRIVATE)
                        .edit()
                        .putBoolean(WorkerService.KEY_CONNECTED, connected)
                        .putBoolean(WorkerService.KEY_WORKER_REQUESTED, connected)
                        .apply()
                }
                updateWorkerButton()
                updateBenchmarkButton()
            }
        }
    }

    private val benchmarkStatusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != BenchmarkService.ACTION_STATUS) return
            val message = intent.getStringExtra(BenchmarkService.EXTRA_MESSAGE)
                ?: "Benchmark status unavailable"
            statusText.text = message
            intent.getStringExtra(BenchmarkService.EXTRA_METRICS)?.let(::renderCheckpointJson)
            intent.getIntExtra(BenchmarkService.EXTRA_PROGRESS, -1)
                .takeIf { it >= 0 }?.let { progressBar.progress = it }
            benchmarkButton.text = when (intent.getStringExtra(BenchmarkService.EXTRA_STATUS)) {
                BenchmarkService.STATUS_RUNNING -> if (
                    isAppServiceRunning(this@MainActivity, BenchmarkService::class.java)
                ) "Cancel benchmark" else "Resume benchmark"
                BenchmarkService.STATUS_COMPLETED -> "Benchmark complete"
                BenchmarkService.STATUS_FAILED -> "Resume benchmark"
                else -> "Run benchmark"
            }
            refreshSuiteProgress()
            updateBenchmarkButton()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.status_text)
        workerStatusText = findViewById(R.id.worker_status)
        metricsText = findViewById(R.id.metrics_text)
        benchmarkButton = findViewById(R.id.benchmark_button)
        benchmarkResetButton = findViewById(R.id.benchmark_reset_button)
        benchmarkSteps = findViewById(R.id.benchmark_steps)
        workerConnectionButton = findViewById(R.id.worker_connection_button)
        schedulerUrl = findViewById(R.id.scheduler_url)
        progressBar = findViewById(R.id.progress_bar)
        modelStatusText = findViewById(R.id.model_status)
        modelDescriptionText = findViewById(R.id.model_description)
        modelDownloadButton = findViewById(R.id.model_download_button)
        modelWifiOnly = findViewById(R.id.model_wifi_only)
        modelProgressBar = findViewById(R.id.model_progress_bar)
        modelManager = RequiredModelManager(this)

        benchmarkButton.setOnClickListener { startBenchmark() }
        benchmarkResetButton.setOnClickListener { confirmBenchmarkReset() }
        benchmarkButton.isEnabled = false
        workerConnectionButton.setOnClickListener { toggleWorkerConnection() }
        modelDownloadButton.setOnClickListener { toggleModelDownload() }

        updateUI("Prepare this device to join IntelHive")
        updateWorkerButton()
    }

    override fun onStart() {
        super.onStart()
        ContextCompat.registerReceiver(
            this,
            workerStatusReceiver,
            IntentFilter(WorkerService.ACTION_STATUS),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        receiverRegistered = true
        ContextCompat.registerReceiver(
            this,
            benchmarkStatusReceiver,
            IntentFilter(BenchmarkService.ACTION_STATUS),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        benchmarkReceiverRegistered = true
        val preferences = getSharedPreferences("worker_runtime", MODE_PRIVATE)
        val serviceRunning = (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager)
            .getRunningServices(Int.MAX_VALUE)
            .any { it.service.className == WorkerService::class.java.name }
        if (!serviceRunning) {
            preferences.edit()
                .putBoolean(WorkerService.KEY_CONNECTED, false)
                .putBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
                .apply()
        }
        workerStatusText.text = if (serviceRunning && preferences.getBoolean(WorkerService.KEY_CONNECTED, false)) {
            "Worker ready"
        } else if (serviceRunning) {
            "Connecting to scheduler..."
        } else {
            "Worker disconnected"
        }
        updateWorkerButton()
        refreshModelLifecycle()
        refreshBenchmarkReadiness()
    }

    override fun onStop() {
        modelDownloadJob?.cancel()
        modelDownloadJob = null
        if (receiverRegistered) {
            unregisterReceiver(workerStatusReceiver)
            receiverRegistered = false
        }
        if (benchmarkReceiverRegistered) {
            unregisterReceiver(benchmarkStatusReceiver)
            benchmarkReceiverRegistered = false
        }
        super.onStop()
    }

    private fun startBenchmark() {
        if (isAppServiceRunning(this, BenchmarkService::class.java)) {
            try {
                startService(Intent(this, BenchmarkService::class.java).setAction(BenchmarkService.ACTION_CANCEL))
                benchmarkButton.isEnabled = false
                statusText.text = "Cancellation requested; waiting for the current native inference pass"
            } catch (error: Exception) {
                statusText.text = "Could not cancel characterization: ${error.message ?: "unknown error"}"
            }
            return
        }
        val preferences = getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
        if (isAppServiceRunning(this, WorkerService::class.java) ||
            preferences.getBoolean(WorkerService.KEY_CONNECTED, false) ||
            preferences.getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        ) {
            statusText.text = "Disconnect the worker before running a benchmark"
            return
        }
        val existing = runCatching {
            BenchmarkSuiteCheckpointStore(
                java.io.File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME)
            ).readOrCreate(CharacterizationMode.QUICK)
        }.getOrNull()
        if (existing != null && existing.steps.any { it.attempts.isNotEmpty() } &&
            existing.status !in setOf("COMPLETED", "COMPLETED_WITH_FAILURES")
        ) {
            launchBenchmark(existing.mode)
            return
        }
        if (existing != null && existing.steps.any { step ->
                step.attempts.any {
                    it.uploadStatus == com.intellihive.worker.benchmark.BenchmarkUploadStatus.PENDING &&
                        it.result.isNotEmpty()
                }
            }
        ) {
            launchBenchmark(existing.mode)
            return
        }
        AlertDialog.Builder(this)
            .setTitle("Benchmark mode")
            .setItems(
                arrayOf(
                    "Quick (32 tokens per backend)",
                    "Full (32, 64, and 128 tokens per backend)"
                )
            ) { _, index ->
                launchBenchmark(if (index == 0) CharacterizationMode.QUICK else CharacterizationMode.FULL)
            }
            .show()
    }

    private fun launchBenchmark(mode: CharacterizationMode) {
        benchmarkButton.isEnabled = false
        statusText.text = "Starting ${mode.wireValue} benchmark..."
        try {
            ContextCompat.startForegroundService(
                this,
                Intent(this, BenchmarkService::class.java)
                    .setAction(BenchmarkService.ACTION_START_STAGED)
                    .putExtra(BenchmarkService.EXTRA_MODE, mode.wireValue)
            )
        } catch (error: Exception) {
            statusText.text = "Could not start benchmark: ${error.message ?: "unknown error"}"
            benchmarkButton.isEnabled = true
        }
    }

    private fun retryBenchmarkStep(stepId: String) {
        if (isAppServiceRunning(this, BenchmarkService::class.java)) return
        val intent = Intent(this, BenchmarkService::class.java)
            .setAction(BenchmarkService.ACTION_RETRY_STEP)
            .putExtra(BenchmarkService.EXTRA_STEP_ID, stepId)
        ContextCompat.startForegroundService(this, intent)
        statusText.text = "Retrying $stepId..."
        updateBenchmarkButton()
    }

    private fun confirmBenchmarkReset() {
        if (isAppServiceRunning(this, BenchmarkService::class.java)) {
            statusText.text = "Stop the active benchmark before resetting it"
            return
        }
        AlertDialog.Builder(this)
            .setTitle("Reset benchmark?")
            .setMessage("This clears local progress and starts a new run. Supabase history is retained.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Reset") { _, _ ->
                lifecycleScope.launch {
                    try {
                        val checkpoint = withContext(Dispatchers.IO) {
                            val store = BenchmarkSuiteCheckpointStore(
                                java.io.File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME)
                            )
                            val currentMode = store.readOrCreate(CharacterizationMode.QUICK).mode
                            store.reset(currentMode).also {
                                BenchmarkReadinessStore(
                                    java.io.File(filesDir, BenchmarkReadinessStore.FILE_NAME)
                                ).write(
                                    com.intellihive.worker.benchmark.BenchmarkReadiness(
                                        BenchmarkStatus.NOT_RUN
                                    )
                                )
                            }
                        }
                        renderCheckpoint(checkpoint)
                        progressBar.progress = 0
                        statusText.text = "Benchmark reset. Ready to start from the first CPU step."
                        updateBenchmarkButton()
                    } catch (error: Exception) {
                        statusText.text = "Could not reset benchmark: ${error.message}"
                    }
                }
            }
            .show()
    }

    private fun toggleWorkerConnection() {
        if (!modelReady) {
            workerStatusText.text = "Download and verify the required model before connecting"
            return
        }
        val preferences = getSharedPreferences("worker_runtime", MODE_PRIVATE)
        val requested = preferences.getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        if (requested) {
            stopService(Intent(this, WorkerService::class.java))
            preferences.edit().putBoolean(WorkerService.KEY_WORKER_REQUESTED, false).apply()
            workerStatusText.text = "Worker disconnected"
        } else {
            val endpoint = schedulerUrl.text.toString().trim()
            if (endpoint.isBlank()) {
                workerStatusText.text = "Enter the scheduler WebSocket URL"
                return
            }
            val serviceIntent = Intent(this, WorkerService::class.java)
                .putExtra(WorkerService.EXTRA_SERVER_URL, endpoint)
            ContextCompat.startForegroundService(this, serviceIntent)
            preferences.edit().putBoolean(WorkerService.KEY_WORKER_REQUESTED, true).apply()
            workerStatusText.text = "Connecting to scheduler..."
        }
        updateWorkerButton()
        updateBenchmarkButton()
    }

    private fun updateWorkerButton() {
        if (!::workerConnectionButton.isInitialized) return
        val requested = getSharedPreferences("worker_runtime", MODE_PRIVATE)
            .getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        workerConnectionButton.text = if (requested) "Disconnect worker" else "Connect as worker"
        workerConnectionButton.isEnabled = requested || modelReady
    }

    private fun refreshBenchmarkReadiness() {
        lifecycleScope.launch {
            try {
                val checkpoint = withContext(Dispatchers.IO) {
                    BenchmarkSuiteCheckpointStore(
                        java.io.File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME)
                    ).readOrCreate()
                }
                renderCheckpoint(checkpoint)
                updateBenchmarkButton()
            } catch (error: Exception) {
                statusText.text = "Could not load staged benchmark state: ${error.message}"
                updateBenchmarkButton()
            }
        }
    }

    private fun refreshSuiteProgress() {
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    BenchmarkSuiteCheckpointStore(
                        java.io.File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME)
                    ).readOrCreate()
                }
            }.onSuccess(::renderCheckpoint)
                .onFailure { statusText.text = "Could not refresh benchmark steps: ${it.message}" }
        }
    }

    private fun renderCheckpointJson(json: String) {
        runCatching { Gson().fromJson(json, BenchmarkSuiteCheckpoint::class.java) }
            .onSuccess { checkpoint -> if (checkpoint != null) renderCheckpoint(checkpoint) }
            .onFailure { statusText.text = "Could not display benchmark progress: ${it.message}" }
    }

    private fun renderCheckpoint(checkpoint: BenchmarkSuiteCheckpoint) {
        benchmarkSteps.removeAllViews()
        val grouped = checkpoint.steps.groupBy { it.definition.kind }
        grouped.forEach { (kind, steps) ->
            val heading = TextView(this).apply {
                text = when (kind) {
                    com.intellihive.worker.benchmark.BenchmarkStepKind.LLAMA_CPP_BASELINE ->
                        "Upstream llama.cpp CPU baseline"
                    com.intellihive.worker.benchmark.BenchmarkStepKind.CPU_ONLY ->
                        "IntelHive CPU-only layered shards"
                    com.intellihive.worker.benchmark.BenchmarkStepKind.VULKAN_ONLY ->
                        "IntelHive Vulkan-only layered shards"
                    com.intellihive.worker.benchmark.BenchmarkStepKind.HYBRID ->
                        "IntelHive hybrid layered shards"
                }
                textSize = 16f
                setPadding(0, 14, 0, 6)
            }
            benchmarkSteps.addView(heading)
            steps.forEach { step ->
                val attempt = step.latestAttempt
                val state = attempt?.status ?: BenchmarkStepStatus.PENDING
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(10, 8, 10, 8)
                }
                val progressTokens = attempt?.generatedTokens ?: 0
                val rate = (attempt?.result?.get("generation_tokens_per_second") as? Number)
                    ?.toDouble()
                val summary = buildString {
                    append("${step.definition.generatedTokens} tokens, ")
                    append("${step.definition.graphThreads} CPU threads, ")
                    append("${step.definition.durationMs / 1000}s — ${state.name}")
                    val requestedGpuLayers = (attempt?.result?.get("requested_gpu_layers") as? Number)
                        ?.toInt() ?: if (kind == com.intellihive.worker.benchmark.BenchmarkStepKind.VULKAN_ONLY) {
                            36
                        } else if (kind == com.intellihive.worker.benchmark.BenchmarkStepKind.HYBRID) {
                            20
                        } else 0
                    append(" • GPU layers $requestedGpuLayers")
                    if (state == BenchmarkStepStatus.RUNNING && progressTokens > 0) {
                        append(" ($progressTokens generated)")
                    }
                    if (state == BenchmarkStepStatus.RUNNING) {
                        val elapsedSeconds =
                            (attempt?.result?.get("elapsed_ms") as? Number)?.toLong()?.div(1000)
                        val stage = attempt?.result?.get("stage") as? String
                        if (elapsedSeconds != null) {
                            append(" • ${stage ?: "running"} ${elapsedSeconds}s")
                        }
                    }
                    if (rate != null) append(" • ${"%.3f".format(java.util.Locale.US, rate)} tok/s")
                    if (attempt?.uploadStatus == com.intellihive.worker.benchmark.BenchmarkUploadStatus.PENDING &&
                        attempt.result.isNotEmpty()
                    ) append(" • upload pending")
                    attempt?.errorReason?.let { appendLine(); append(it) }
                    attempt?.uploadError?.let { appendLine(); append("Upload: $it") }
                }
                row.addView(TextView(this).apply {
                    text = summary
                    textSize = 13f
                })
                if (state in setOf(
                        BenchmarkStepStatus.FAILED,
                        BenchmarkStepStatus.CANCELLED,
                        BenchmarkStepStatus.SKIPPED
                    )
                ) {
                    row.addView(Button(this).apply {
                        text = "Retry this step"
                        setOnClickListener { retryBenchmarkStep(step.definition.id) }
                    })
                }
                benchmarkSteps.addView(row)
            }
        }
    }

    private fun updateBenchmarkButton() {
        if (!::benchmarkButton.isInitialized) return
        val benchmarkRunning = isAppServiceRunning(this, BenchmarkService::class.java)
        if (benchmarkRunning) {
            benchmarkButton.text = "Cancel benchmark"
            benchmarkButton.isEnabled = true
            return
        }
        val workerActive = isAppServiceRunning(this, WorkerService::class.java) ||
            getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
                .getBoolean(WorkerService.KEY_CONNECTED, false) ||
            getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
                .getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        benchmarkButton.isEnabled = modelReady && !workerActive
        benchmarkResetButton.isEnabled = !workerActive
        if (workerActive) return
        val checkpoint = runCatching {
            BenchmarkSuiteCheckpointStore(
                java.io.File(filesDir, BenchmarkSuiteCheckpointStore.FILE_NAME)
            ).readOrCreate()
        }.getOrNull()
        val suiteStatus = checkpoint?.status
        val pendingUploads = checkpoint?.steps?.any { step ->
            step.attempts.any {
                it.uploadStatus == com.intellihive.worker.benchmark.BenchmarkUploadStatus.PENDING &&
                    it.result.isNotEmpty()
            }
        } == true
        benchmarkButton.text = when {
            pendingUploads -> "Retry pending Supabase uploads"
            suiteStatus == "COMPLETED" -> "Benchmark complete"
            suiteStatus == "COMPLETED_WITH_FAILURES" -> "Retry failed steps below"
            suiteStatus in setOf("RUNNING", "PENDING", "CANCELLED") -> "Resume benchmark"
            else -> "Run benchmark"
        }
        if (!pendingUploads &&
            (suiteStatus == "COMPLETED" || suiteStatus == "COMPLETED_WITH_FAILURES")
        ) {
            benchmarkButton.isEnabled = false
        }
    }

    private fun refreshModelLifecycle() {
        lifecycleScope.launch {
            modelDownloadButton.isEnabled = false
            modelStatusText.text = "Loading required model configuration..."
            try {
                val manifest = modelManager.loadManifest()
                modelManifest = manifest
                modelDescriptionText.text =
                    "${manifest.displayName} • ${manifest.sizeLabel} download • about 2.4 GB free storage required"
                if (modelManager.isInstalled(manifest)) {
                    showModelReadyState()
                    return@launch
                }

                modelReady = false
                modelStatusText.text = "Model required to participate in compute jobs"
                modelDownloadButton.text = "Download model (${manifest.sizeLabel})"
                modelDownloadButton.isEnabled = true
                updateWorkerButton()
                resumeModelDownload(manifest)
            } catch (error: Exception) {
                modelReady = false
                modelStatusText.text = "Model setup unavailable: ${error.message ?: "configuration error"}"
                modelDescriptionText.text = "Could not load or validate the required model configuration."
                modelDownloadButton.text = "Model configuration unavailable"
                modelDownloadButton.isEnabled = false
                updateWorkerButton()
            }
        }
    }

    private fun showModelReadyState() {
        modelReady = true
        modelStatusText.text =
            "Model verified • Native engine will initialize when the worker connects"
        modelDownloadButton.text = "Model verified"
        modelDownloadButton.isEnabled = false
        benchmarkButton.text = "Run device characterization"
        updateBenchmarkButton()
        if (isAppServiceRunning(this, WorkerService::class.java)) {
            statusText.text = "Disconnect the worker before running a benchmark"
        }
        modelProgressBar.isIndeterminate = false
        modelProgressBar.progress = 100
        updateWorkerButton()
    }

    private fun toggleModelDownload() {
        val manifest = modelManifest ?: return
        val activeId = modelManager.activeDownloadId(manifest)
        if (activeId != null) {
            modelManager.cancel(activeId)
            modelDownloadJob?.cancel()
            modelDownloadJob = null
            modelProgressBar.progress = 0
            modelStatusText.text = "Model download cancelled"
            modelDownloadButton.text = "Download model (${manifest.sizeLabel})"
            modelDownloadButton.isEnabled = true
            return
        }

        try {
            val id = modelManager.enqueueDownload(manifest, wifiOnly = modelWifiOnly.isChecked)
            if (id == RequiredModelManager.INVALID_DOWNLOAD_ID) {
                showModelReadyState()
            } else {
                modelDownloadButton.text = "Cancel download"
                modelStatusText.text = if (modelWifiOnly.isChecked) {
                    "Downloading over Wi-Fi only..."
                } else {
                    "Downloading model..."
                }
                watchModelDownload(manifest, id)
            }
        } catch (error: Exception) {
            modelStatusText.text = "Could not start model download: ${error.message}"
        }
    }

    private fun resumeModelDownload(manifest: ModelManifest) {
        val id = modelManager.activeDownloadId(manifest) ?: return
        when (val state = modelManager.query(id)) {
            is ModelDownloadState.Pending, is ModelDownloadState.Downloading -> {
                modelDownloadButton.text = "Cancel download"
                watchModelDownload(manifest, id)
            }
            ModelDownloadState.Downloaded -> watchModelDownload(manifest, id)
            is ModelDownloadState.Failed -> {
                modelStatusText.text = state.message
                modelManager.cancel(id)
            }
            ModelDownloadState.Missing -> modelManager.cancel(id)
        }
    }

    private fun watchModelDownload(manifest: ModelManifest, downloadId: Long) {
        modelDownloadJob?.cancel()
        modelProgressBar.isIndeterminate = false
        modelDownloadJob = lifecycleScope.launch {
            while (isActive) {
                when (val state = modelManager.query(downloadId)) {
                    is ModelDownloadState.Pending -> {
                        modelStatusText.text = state.message
                        modelProgressBar.isIndeterminate = true
                    }
                    is ModelDownloadState.Downloading -> {
                        modelProgressBar.isIndeterminate = false
                        if (state.totalBytes > 0) {
                            modelProgressBar.progress =
                                (state.bytesDownloaded * 100 / state.totalBytes).toInt().coerceIn(0, 100)
                        }
                        modelStatusText.text =
                            "Downloading model: ${modelProgressBar.progress}%"
                    }
                    ModelDownloadState.Downloaded -> {
                        modelStatusText.text = "Download complete. Verifying file size and SHA-256..."
                        modelDownloadButton.isEnabled = false
                        try {
                            modelManager.verifyAndInstall(manifest)
                            showModelReadyState()
                        } catch (error: Exception) {
                            modelStatusText.text =
                                "Model verification failed: ${error.message ?: "unknown error"}"
                            modelDownloadButton.text = "Retry download"
                            modelDownloadButton.isEnabled = true
                            modelManager.cancel(downloadId)
                        }
                        return@launch
                    }
                    is ModelDownloadState.Failed -> {
                        modelStatusText.text = state.message
                        modelDownloadButton.text = "Retry download"
                        modelDownloadButton.isEnabled = true
                        modelManager.cancel(downloadId)
                        return@launch
                    }
                    ModelDownloadState.Missing -> {
                        modelStatusText.text = "Android download record is unavailable"
                        modelDownloadButton.text = "Retry download"
                        modelDownloadButton.isEnabled = true
                        return@launch
                    }
                }
                delay(1_000)
            }
        }
    }

    private fun updateUI(status: String) {
        runOnUiThread { statusText.text = status }
    }

    fun updateMetrics(metrics: String) {
        runOnUiThread {
            metricsText.text = metrics
            benchmarkButton.isEnabled = true
            progressBar.progress = 100
        }
    }
}
