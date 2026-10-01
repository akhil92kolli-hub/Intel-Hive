package com.intellihive.worker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.app.ActivityManager
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
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
import com.intellihive.worker.service.BenchmarkService
import com.intellihive.worker.service.WorkerService
import com.intellihive.worker.service.isAppServiceRunning
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var workerStatusText: TextView
    private lateinit var metricsText: TextView
    private lateinit var benchmarkButton: Button
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
            intent.getStringExtra(BenchmarkService.EXTRA_METRICS)?.let { metricsText.text = it }
            intent.getIntExtra(BenchmarkService.EXTRA_PROGRESS, -1)
                .takeIf { it >= 0 }?.let { progressBar.progress = it }
            benchmarkButton.text = when (intent.getStringExtra(BenchmarkService.EXTRA_STATUS)) {
                BenchmarkService.STATUS_RUNNING -> "Benchmark running..."
                BenchmarkService.STATUS_COMPLETED -> "Run benchmark again"
                BenchmarkService.STATUS_FAILED -> "Retry benchmark"
                else -> "Run native shard benchmark"
            }
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
        val preferences = getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
        if (isAppServiceRunning(this, WorkerService::class.java) ||
            preferences.getBoolean(WorkerService.KEY_CONNECTED, false) ||
            preferences.getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        ) {
            statusText.text = "Disconnect the worker before running a benchmark"
            return
        }
        benchmarkButton.isEnabled = false
        progressBar.progress = 0
        statusText.text = "Starting native shard benchmark..."
        try {
            val intent = Intent(this, BenchmarkService::class.java)
                .putExtra(BenchmarkService.EXTRA_PREFILL_TOKENS, 32)
                .putExtra(BenchmarkService.EXTRA_GENERATED_TOKENS, 8)
            ContextCompat.startForegroundService(this, intent)
        } catch (error: Exception) {
            statusText.text = "Could not start benchmark: ${error.message ?: "unknown error"}"
            benchmarkButton.isEnabled = true
        }
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
                val readiness = BenchmarkReadinessStore(
                    java.io.File(filesDir, BenchmarkReadinessStore.FILE_NAME)
                ).read()
                if (readiness.status == BenchmarkStatus.COMPLETED) {
                    metricsText.text = buildString {
                        appendLine("Status: ${readiness.status}")
                        appendLine("Model: ${readiness.modelId} ${readiness.modelVersion}")
                        appendLine("Execution mode: ${readiness.executionMode?.wireValue ?: "unknown"}")
                        appendLine("Throughput: ${"%.2f".format(java.util.Locale.US, readiness.tokensPerSecond)} tokens/s")
                        appendLine("Prefill: ${"%.2f".format(java.util.Locale.US, readiness.prefillSpeedTokensPerSecond)} tokens/s")
                        appendLine("Decode: ${"%.2f".format(java.util.Locale.US, readiness.generationSpeedTokensPerSecond)} tokens/s")
                        append("Shards: ${readiness.layerRanges.joinToString { "${it.layerStart}-${it.layerEnd}" }}")
                    }
                    statusText.text = "Benchmark completed. Connect to report worker readiness."
                } else if (readiness.status == BenchmarkStatus.FAILED) {
                    statusText.text = "Last benchmark failed: ${readiness.errorMessage}"
                } else if (readiness.status == BenchmarkStatus.RUNNING) {
                    statusText.text = "A native benchmark was interrupted; retry to report its failure."
                }
                benchmarkButton.text = when (readiness.status) {
                    BenchmarkStatus.NOT_RUN -> "Run native shard benchmark"
                    BenchmarkStatus.RUNNING -> "Recover benchmark"
                    BenchmarkStatus.COMPLETED -> "Run benchmark again"
                    BenchmarkStatus.FAILED -> "Retry benchmark"
                }
                updateBenchmarkButton()
            } catch (error: Exception) {
                statusText.text = "Could not load benchmark state: ${error.message}"
                updateBenchmarkButton()
            }
        }
    }

    private fun updateBenchmarkButton() {
        if (!::benchmarkButton.isInitialized) return
        val workerActive = isAppServiceRunning(this, WorkerService::class.java) ||
            getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
                .getBoolean(WorkerService.KEY_CONNECTED, false) ||
            getSharedPreferences(WorkerService.PREFERENCES, MODE_PRIVATE)
                .getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        val benchmarkRunning = isAppServiceRunning(this, BenchmarkService::class.java)
        benchmarkButton.isEnabled = modelReady && !workerActive && !benchmarkRunning
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
        benchmarkButton.text = "Run native shard benchmark"
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
