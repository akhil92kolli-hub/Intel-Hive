package com.intellihive.worker.service

import com.intellihive.worker.inference.Activation
import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.intellihive.worker.MainActivity
import com.intellihive.worker.benchmark.BenchmarkReadiness
import com.intellihive.worker.benchmark.BenchmarkReadinessStore
import com.intellihive.worker.benchmark.BenchmarkStatus
import com.intellihive.worker.inference.QwenShardCatalog
import com.intellihive.worker.inference.NativeLayerRangeShardExecutor
import com.intellihive.worker.inference.NativeRuntime
import com.intellihive.worker.inference.NativeTransportShardExecutor
import com.intellihive.worker.inference.UnavailableNativeShardExecutor
import com.intellihive.worker.model.RequiredModelManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.TimeUnit

class WorkerService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var client: OkHttpClient? = null
    private var socket: WebSocket? = null
    private var initializationJob: Job? = null
    private var heartbeatJob: Job? = null
    private var workerId: String = ""
    private var heartbeatIntervalSeconds = DEFAULT_HEARTBEAT_SECONDS
    private var registered = false
    private var terminalStatusReported = false
    private var benchmarkReadiness = BenchmarkReadiness(BenchmarkStatus.NOT_RUN)
    private var shardCatalogPrepared = false
    private val assignmentExecutorLock = Any()
    private var assignmentExecutor: ShardExecutor? = null
    private val executionMutex = Mutex()
    private val activeAssignments = HashSet<String>()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (socket != null || initializationJob?.isActive == true) return START_NOT_STICKY
        terminalStatusReported = false

        val endpoint = intent?.getStringExtra(EXTRA_SERVER_URL)?.trim().orEmpty()
        if (!endpoint.startsWith("ws://") && !endpoint.startsWith("wss://")) {
            reportStatus("Enter a valid ws:// or wss:// scheduler URL", connected = false)
            stopSelf(startId)
            return START_NOT_STICKY
        }

        startForeground(NOTIFICATION_ID, notification("Verifying model and preparing native engine"))
        workerId = getOrCreateWorkerId()
        val request = try {
            Request.Builder().url(endpoint).build()
        } catch (error: IllegalArgumentException) {
            Log.e(TAG, "Invalid scheduler URL: $endpoint", error)
            reportStatus("Invalid scheduler URL", connected = false)
            stopSelf(startId)
            return START_NOT_STICKY
        }

        initializationJob = serviceScope.launch {
            var nativeExecutor: NativeLayerRangeShardExecutor? = null
            try {
                check(!isAppServiceRunning(this@WorkerService, BenchmarkService::class.java)) {
                    "Stop the benchmark before connecting the worker"
                }
                val readinessStore = BenchmarkReadinessStore(
                    java.io.File(filesDir, BenchmarkReadinessStore.FILE_NAME)
                )
                benchmarkReadiness = readinessStore.read()
                if (benchmarkReadiness.status == BenchmarkStatus.RUNNING) {
                    benchmarkReadiness = BenchmarkReadiness.failed(
                        "The previous benchmark was interrupted before it completed"
                    )
                    readinessStore.write(benchmarkReadiness)
                }
                val models = RequiredModelManager(this@WorkerService)
                val manifest = models.loadManifest()
                check(models.isInstalled(manifest)) {
                    "Required model is not verified. Download it before connecting the worker."
                }
                check(NativeRuntime.isAvailable()) {
                    "Native engine unavailable: ${NativeRuntime.unavailableReason()}"
                }
                nativeExecutor = NativeLayerRangeShardExecutor(this@WorkerService)
                nativeExecutor.prepareModel()
                nativeExecutor.prepareShards(QwenShardCatalog.ranges)
                shardCatalogPrepared = true
                synchronized(assignmentExecutorLock) {
                    assignmentExecutor = NativeTransportShardExecutor(nativeExecutor)
                }
                reportProgress("Verified model and native engine ready; connecting to scheduler")

                val httpClient = OkHttpClient.Builder()
                    .pingInterval(HEARTBEAT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    .build()
                client = httpClient
                socket = httpClient.newWebSocket(request, WorkerSocketListener())
            } catch (error: kotlinx.coroutines.CancellationException) {
                nativeExecutor?.close()
                throw error
            } catch (error: Exception) {
                nativeExecutor?.close()
                Log.e(TAG, "Worker model preparation failed", error)
                reportStatus(
                    "Worker could not become ready: ${error.message ?: "model setup failed"}",
                    connected = false
                )
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        val wasConnected = registered || socket != null
        registered = false
        initializationJob?.cancel()
        initializationJob = null
        heartbeatJob?.cancel()
        socket?.close(1000, "worker stopped")
        socket = null
        client?.dispatcher?.cancelAll()
        client = null
        serviceScope.cancel()
        synchronized(assignmentExecutorLock) {
            (assignmentExecutor as? AutoCloseable)?.close()
            assignmentExecutor = null
        }
        if (wasConnected && !terminalStatusReported) {
            reportStatus("Worker disconnected", connected = false)
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private inner class WorkerSocketListener : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.i(TAG, "Connected to scheduler")
            if (!webSocket.send(envelope("register", registrationPayload()).toString())) {
                fail("Could not send worker registration")
            }
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            try {
                val message = JSONObject(text)
                when (message.optString("type")) {
                    "register_ack" -> handleRegistrationAck(message.optJSONObject("payload"))
                    "heartbeat_ack" -> Unit
                    "job_assignment" -> handleJobAssignment(webSocket, message.optJSONObject("payload"))
                    "sequence_end" -> handleSequenceEnd(webSocket, message.optJSONObject("payload"))
                    "error" -> {
                        val detail = message.optJSONObject("payload")?.optString("message")
                            ?.takeIf(String::isNotBlank) ?: "Scheduler rejected worker message"
                        fail(detail)
                    }
                    else -> Log.w(TAG, "Ignoring unsupported scheduler message: ${message.optString("type")}")
                }
            } catch (error: Exception) {
                Log.e(TAG, "Invalid scheduler message", error)
                fail("Scheduler sent an invalid message")
            }
        }

        override fun onFailure(webSocket: WebSocket, error: Throwable, response: Response?) {
            Log.e(TAG, "Scheduler connection failed", error)
            fail("Scheduler connection failed: ${error.message ?: "unknown error"}")
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            if (registered) reportStatus("Scheduler connection closed: $reason", connected = false)
            stopSelf()
        }
    }

    private fun handleRegistrationAck(payload: JSONObject?) {
        if (payload == null || !payload.optBoolean("accepted")) {
            val reason = payload?.optString("reason")?.takeIf(String::isNotBlank)
                ?: "Scheduler rejected registration"
            fail(reason)
            return
        }
        if (payload.optString("worker_id") != workerId) {
            fail("Scheduler acknowledged a different worker ID")
            return
        }

        heartbeatIntervalSeconds = payload.optInt(
            "heartbeat_interval_seconds",
            DEFAULT_HEARTBEAT_SECONDS
        ).coerceIn(5, 300)
        registered = true
        val readyMessage = if (benchmarkReadiness.status == BenchmarkStatus.COMPLETED) {
            "Worker ready (benchmark completed)"
        } else {
            "Worker ready (benchmark not run)"
        }
        reportStatus(readyMessage, connected = true)
        heartbeatJob = serviceScope.launch {
            while (isActive) {
                delay(TimeUnit.SECONDS.toMillis(heartbeatIntervalSeconds.toLong()))
                val activeSocket = socket ?: break
                if (!activeSocket.send(envelope("heartbeat", heartbeatPayload()).toString())) {
                    fail("Could not send heartbeat")
                    break
                }
            }
        }
    }

    private fun handleJobAssignment(webSocket: WebSocket, payload: JSONObject?) {
        if (!registered || socket !== webSocket) {
            fail("Received an assignment before worker registration completed")
            return
        }
        if (payload == null) {
            fail("Scheduler sent an assignment without a payload")
            return
        }

        val assignment = try {
            workerProtocolGson.fromJson(payload.toString(), WorkerJobAssignment::class.java)
        } catch (error: Exception) {
            Log.e(TAG, "Could not parse scheduler assignment", error)
            fail("Scheduler sent a malformed job assignment")
            return
        }
        val validationError = assignment.validate() ?: if (assignment.workerId != workerId) "assignment worker_id does not match this worker" else null
        if (validationError != null) {
            sendAssignmentFailure(webSocket, assignment, validationError)
            return
        }
        synchronized(activeAssignments) {
            if (!activeAssignments.add(assignment.assignmentId)) {
                sendAssignmentFailure(webSocket, assignment, "duplicate assignment_id")
                return
            }
        }

        if (!sendProtocolMessage(
                webSocket,
                "job_accepted",
                WorkerJobAccepted(assignment.assignmentId, accepted = true)
            )
        ) {
            synchronized(activeAssignments) { activeAssignments.remove(assignment.assignmentId) }
            fail("Could not acknowledge inference assignment")
            return
        }

        serviceScope.launch {
            try {
                executionMutex.withLock {
                    val result = assignmentExecutor().execute(assignment)
                    validateExecutionResult(assignment, result)?.let { error ->
                        throw IllegalStateException(error)
                    }
                    val completion = WorkerJobComplete(
                        assignmentId = assignment.assignmentId,
                        jobId = assignment.jobId,
                        workerId = workerId,
                        sequenceId = assignment.sequenceId,
                        position = assignment.position,
                        passOrdinal = result.passOrdinal,
                        kvTokenOffsetBefore = result.kvTokenOffsetBefore,
                        kvTokenOffsetAfter = result.kvTokenOffsetAfter,
                        activation = result.activation,
                        sampledTokenId = result.sampledTokenId,
                        endOfSequence = result.endOfSequence,
                        generatedText = result.generatedText,
                        tokensPerSecond = result.tokensPerSecond
                    )
                    if (!sendProtocolMessage(webSocket, "job_complete", completion)) {
                        Log.e(TAG, "Could not send completion for ${assignment.assignmentId}")
                        webSocket.close(1011, "could not send inference completion")
                    }
                }
            } catch (error: kotlinx.coroutines.CancellationException) {
                throw error
            } catch (error: Exception) {
                Log.e(TAG, "Shard execution failed for ${assignment.assignmentId}", error)
                sendAssignmentFailure(
                    webSocket,
                    assignment,
                    error.message ?: "shard execution failed"
                )
            } finally {
                synchronized(activeAssignments) { activeAssignments.remove(assignment.assignmentId) }
            }
        }
    }

    private fun handleSequenceEnd(webSocket: WebSocket, payload: JSONObject?) {
        if (!registered || socket !== webSocket) return
        if (payload == null) {
            fail("Scheduler sent a sequence_end without a payload")
            return
        }
        val jobId = payload.optString("job_id")
        val sequenceId = payload.optString("sequence_id")
        if (jobId.isBlank() || sequenceId.isBlank() || payload.opt("completed") !is Boolean) {
            fail("Scheduler sent an invalid sequence_end")
            return
        }
        val completed = payload.optBoolean("completed")
        serviceScope.launch {
            try {
                executionMutex.withLock {
                    assignmentExecutor?.endSequence(jobId, sequenceId, completed)
                }
            } catch (error: kotlinx.coroutines.CancellationException) {
                throw error
            } catch (error: Exception) {
                Log.e(TAG, "Could not release inference sequence $sequenceId", error)
            }
        }
    }

    private fun validateExecutionResult(
        assignment: WorkerJobAssignment,
        result: ShardExecutionResult
    ): String? {
        if (assignment.finalShard) {
            val hasToken = result.sampledTokenId != null
            if (hasToken == result.endOfSequence || result.activation != null) {
                return "final shard must return exactly one sampled token or EOS"
            }
            if (result.sampledTokenId != null &&
                result.sampledTokenId !in 0L..0xFFFFFFFFL
            ) {
                return "sampled token ID is out of range"
            }
        } else if (result.activation == null ||
            result.sampledTokenId != null || result.endOfSequence
        ) {
            return "non-final shard must return only a non-empty activation"
        }
        return null
    }

    private fun sendAssignmentFailure(
        webSocket: WebSocket,
        assignment: WorkerJobAssignment,
        reason: String
    ) {
        if (assignment.assignmentId.isBlank() || assignment.jobId.isBlank()) {
            fail("Cannot correlate malformed job assignment failure")
            return
        }
        if (!sendProtocolMessage(
                webSocket,
                "job_failed",
                WorkerJobFailed(
                    assignmentId = assignment.assignmentId,
                    jobId = assignment.jobId,
                    workerId = workerId,
                    reason = reason
                )
            )
        ) {
            Log.e(TAG, "Could not report failed assignment ${assignment.assignmentId}")
            webSocket.close(1011, "could not report inference failure")
        }
    }

    private fun sendProtocolMessage(webSocket: WebSocket, type: String, payload: Any): Boolean =
        webSocket.send(
            workerProtocolGson.toJson(
                WorkerEnvelope(type, workerProtocolGson.toJsonTree(payload))
            )
        )

    private fun assignmentExecutor(): ShardExecutor = synchronized(assignmentExecutorLock) {
        assignmentExecutor ?: NativeTransportShardExecutor(
            if (NativeRuntime.isAvailable()) {
                NativeLayerRangeShardExecutor(this)
            } else {
                UnavailableNativeShardExecutor(
                    NativeRuntime.unavailableReason() ?: "native runtime is unavailable"
                )
            }
        ).also { assignmentExecutor = it }
    }

    private fun registrationPayload(): JSONObject {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memory = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryLevel = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, 0) ?: 0
        val batteryScale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batteryPercent = if (batteryScale > 0) {
            (batteryLevel * 100 / batteryScale).coerceIn(0, 100)
        } else {
            0
        }
        val chargingStatus = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val charging = chargingStatus == BatteryManager.BATTERY_STATUS_CHARGING ||
            chargingStatus == BatteryManager.BATTERY_STATUS_FULL

        val payload = JSONObject()
            .put("protocol_version", PROTOCOL_VERSION)
            .put("worker_id", workerId)
            .put("platform", "android")
            .put("os_version", Build.VERSION.RELEASE)
            .put("device_id", workerId)
            .put("device_public_key", "")
            .put("memory", JSONObject()
                .put("total_mb", memory.totalMem / BYTES_PER_MB)
                .put("available_mb", memory.availMem / BYTES_PER_MB))
            .put("compute", JSONObject()
                .put("cpu", JSONObject().put("available", true))
                .put("gpu", JSONObject().put("available", false))
                .put("npu", JSONObject().put("available", false)))
            .put("inference", JSONObject()
                .put("runtime", "llama.cpp")
                .put("backend", "cpu")
                .put("benchmark_status", benchmarkReadiness.status.name)
                .apply {
                    benchmarkReadiness.tokensPerSecond?.let { tokensPerSecond ->
                        put(
                            "performance",
                            JSONObject()
                                .put("tokens_per_second", tokensPerSecond)
                                .put(
                                    "execution_mode",
                                    benchmarkReadiness.executionMode?.wireValue
                                )
                        )
                    }
                })
            .put("network", JSONObject().put("type", networkType()))
            .put("power", JSONObject()
                .put("battery_percent", batteryPercent)
                .put("charging", charging))
        if (shardCatalogPrepared) {
            val shards = org.json.JSONArray()
            QwenShardCatalog.ranges.forEach { range ->
                shards.put(
                    JSONObject()
                        .put("model_id", benchmarkReadiness.modelId)
                        .put("model_version", benchmarkReadiness.modelVersion)
                        .put("model_artifact_digest", benchmarkReadiness.modelArtifactDigest)
                        .put("shard_id", range.shardId)
                        .put("layer_start", range.layerStart)
                        .put("layer_end", range.layerEnd)
                )
            }
            payload.put("loaded_shards", shards)
        }
        return payload
    }

    private fun heartbeatPayload(): JSONObject {
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryLevel = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, 0) ?: 0
        val batteryScale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batteryPercent = if (batteryScale > 0) {
            (batteryLevel * 100 / batteryScale).coerceIn(0, 100)
        } else {
            0
        }
        val chargingStatus = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        return JSONObject()
            .put("worker_id", workerId)
            .put(
                "state",
                when {
                    !registered -> "REGISTERING"
                    benchmarkReadiness.status == BenchmarkStatus.COMPLETED -> "READY"
                    else -> "UNBENCHMARKED"
                }
            )
            .put("battery_percent", batteryPercent)
            .put("charging", chargingStatus == BatteryManager.BATTERY_STATUS_CHARGING ||
                chargingStatus == BatteryManager.BATTERY_STATUS_FULL)
            .put("timestamp", java.time.Instant.now().toString())
    }

    private fun networkType(): String {
        val manager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = manager.activeNetwork ?: return "unknown"
        val capabilities = manager.getNetworkCapabilities(network) ?: return "unknown"
        return when {
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
            else -> "unknown"
        }
    }

    private fun envelope(type: String, payload: JSONObject): JSONObject =
        JSONObject().put("type", type).put("payload", payload)

    private fun fail(message: String) {
        Log.e(TAG, message)
        reportStatus(message, connected = false)
        stopSelf()
    }

    private fun reportStatus(message: String, connected: Boolean) {
        terminalStatusReported = !connected
        getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_CONNECTED, connected)
            .putBoolean(KEY_WORKER_REQUESTED, connected)
            .apply()
        sendBroadcast(
            Intent(ACTION_STATUS)
                .setPackage(packageName)
                .putExtra(EXTRA_STATUS, message)
                .putExtra(EXTRA_CONNECTED, connected)
        )
        if (connected) {
            updateNotification(message)
        }
    }

    private fun reportProgress(message: String) {
        sendBroadcast(Intent(ACTION_STATUS).setPackage(packageName).putExtra(EXTRA_STATUS, message))
        updateNotification(message)
    }

    private fun getOrCreateWorkerId(): String {
        val preferences = getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val existing = preferences.getString(KEY_WORKER_ID, null)
        if (!existing.isNullOrBlank()) return existing
        val created = UUID.randomUUID().toString()
        preferences.edit().putString(KEY_WORKER_ID, created).apply()
        return created
    }

    private fun notification(message: String): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "IntelHive worker", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val openActivity = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("IntelHive worker")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(openActivity)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(message: String) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(message))
    }

    companion object {
        const val ACTION_STATUS = "com.intellihive.worker.WORKER_STATUS"
        const val ACTION_STOP = "com.intellihive.worker.STOP_WORKER"
        const val EXTRA_SERVER_URL = "server_url"
        const val EXTRA_STATUS = "status"
        const val EXTRA_CONNECTED = "connected"
        private const val TAG = "WorkerService"
        private const val CHANNEL_ID = "worker_runtime"
        private const val NOTIFICATION_ID = 1101
        private const val DEFAULT_HEARTBEAT_SECONDS = 15
        private const val HEARTBEAT_TIMEOUT_SECONDS = 45L
        private const val BYTES_PER_MB = 1024L * 1024L
        private const val PROTOCOL_VERSION = "phase-0-v3"
        const val PREFERENCES = "worker_runtime"
        const val KEY_WORKER_ID = "worker_id"
        const val KEY_CONNECTED = "connected"
        const val KEY_WORKER_REQUESTED = "worker_requested"
    }
}
