package com.intellihive.worker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.app.ActivityManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.intellihive.worker.service.BenchmarkService
import com.intellihive.worker.service.WorkerService
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var workerStatusText: TextView
    private lateinit var metricsText: TextView
    private lateinit var benchmarkButton: Button
    private lateinit var workerConnectionButton: Button
    private lateinit var schedulerUrl: EditText
    private lateinit var progressBar: ProgressBar
    private var receiverRegistered = false

    private val workerStatusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == WorkerService.ACTION_STATUS) {
                workerStatusText.text = intent.getStringExtra(WorkerService.EXTRA_STATUS)
                    ?: "Worker status unavailable"
                updateWorkerButton()
            }
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

        benchmarkButton.setOnClickListener { startBenchmark() }
        workerConnectionButton.setOnClickListener { toggleWorkerConnection() }

        updateUI("Ready to benchmark Qwen2.5-3B-Instruct Q4_K_M")
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
            "Connected (unbenchmarked)"
        } else if (serviceRunning) {
            "Connecting to scheduler..."
        } else {
            "Worker disconnected"
        }
        updateWorkerButton()
    }

    override fun onStop() {
        if (receiverRegistered) {
            unregisterReceiver(workerStatusReceiver)
            receiverRegistered = false
        }
        super.onStop()
    }

    private fun startBenchmark() {
        benchmarkButton.isEnabled = false
        progressBar.progress = 0
        updateUI("Starting benchmark...")

        lifecycleScope.launch {
            try {
                val intent = Intent(this@MainActivity, BenchmarkService::class.java)
                    .putExtra("prefill_tokens", 128)
                    .putExtra("generated_tokens", 100)
                startService(intent)
            } catch (error: Exception) {
                updateUI("Error: ${error.message}")
                benchmarkButton.isEnabled = true
            }
        }
    }

    private fun toggleWorkerConnection() {
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
    }

    private fun updateWorkerButton() {
        if (!::workerConnectionButton.isInitialized) return
        val requested = getSharedPreferences("worker_runtime", MODE_PRIVATE)
            .getBoolean(WorkerService.KEY_WORKER_REQUESTED, false)
        workerConnectionButton.text = if (requested) "Disconnect worker" else "Connect as worker"
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
