package com.intellihive.worker

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.intellihive.worker.service.BenchmarkService
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var metricsText: TextView
    private lateinit var benchmarkButton: Button
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.status_text)
        metricsText = findViewById(R.id.metrics_text)
        benchmarkButton = findViewById(R.id.benchmark_button)
        progressBar = findViewById(R.id.progress_bar)

        benchmarkButton.setOnClickListener {
            startBenchmark()
        }

        updateUI("Ready to benchmark Qwen2.5-3B-Instruct Q4_K_M")
    }

    private fun startBenchmark() {
        benchmarkButton.isEnabled = false
        progressBar.progress = 0
        updateUI("Starting benchmark...")

        lifecycleScope.launch {
            try {
                val intent = Intent(this@MainActivity, BenchmarkService::class.java)
                intent.putExtra("prefill_tokens", 128)
                intent.putExtra("generated_tokens", 100)
                startService(intent)
            } catch (e: Exception) {
                updateUI("Error: ${e.message}")
                benchmarkButton.isEnabled = true
            }
        }
    }

    private fun updateUI(status: String) {
        runOnUiThread {
            statusText.text = status
        }
    }

    fun updateMetrics(metrics: String) {
        runOnUiThread {
            metricsText.text = metrics
            benchmarkButton.isEnabled = true
            progressBar.progress = 100
        }
    }
}
