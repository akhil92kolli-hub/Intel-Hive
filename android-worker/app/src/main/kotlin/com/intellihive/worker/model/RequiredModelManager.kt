package com.intellihive.worker.model

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.os.StatFs
import com.intellihive.worker.BuildConfig
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

sealed interface ModelDownloadState {
    data object Missing : ModelDownloadState
    data class Pending(val message: String) : ModelDownloadState
    data class Downloading(val bytesDownloaded: Long, val totalBytes: Long) : ModelDownloadState
    data object Downloaded : ModelDownloadState
    data class Failed(val message: String) : ModelDownloadState
}

class RequiredModelManager(
    context: Context,
    private val manifestUrl: String = BuildConfig.MODEL_MANIFEST_URL
) {
    private val appContext = context.applicationContext
    private val downloadManager =
        appContext.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val preferences = appContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val httpClient = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    val modelDirectory: File
        get() = defaultModelDirectory(appContext)

    fun installedFile(manifest: ModelManifest): File = File(modelDirectory, manifest.fileName)

    suspend fun loadManifest(): ModelManifest = withContext(Dispatchers.IO) {
        val json = if (manifestUrl.isBlank()) {
            appContext.assets.open(ModelManifest.ASSET_FILE_NAME).bufferedReader().use { it.readText() }
        } else {
            require(manifestUrl.isHttpsUrl()) { "Configured model manifest URL must use HTTPS" }
            val request = Request.Builder().url(manifestUrl).get().build()
            httpClient.newCall(request).execute().use { response ->
                check(response.isSuccessful) {
                    "Model manifest request failed with HTTP ${response.code}"
                }
                val body = checkNotNull(response.body) { "Model manifest response was empty" }
                check(body.contentLength() <= MAX_MANIFEST_BYTES) {
                    "Model manifest response exceeds the size limit"
                }
                body.byteStream().use { input ->
                    val output = ByteArrayOutputStream()
                    val buffer = ByteArray(8 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        check(output.size().toLong() + count <= MAX_MANIFEST_BYTES) {
                            "Model manifest response exceeds the size limit"
                        }
                        output.write(buffer, 0, count)
                    }
                    String(output.toByteArray(), StandardCharsets.UTF_8)
                }
            }
        }
        ModelManifest.fromJson(json)
    }

    fun isInstalled(manifest: ModelManifest): Boolean {
        val file = installedFile(manifest)
        return file.isFile &&
            file.length() == manifest.sizeBytes &&
            preferences.getString(KEY_VERIFIED_DIGEST, null) == manifest.sha256 &&
            preferences.getString(KEY_VERIFIED_VERSION, null) == manifest.version
    }

    fun activeDownloadId(manifest: ModelManifest): Long? {
        if (preferences.getString(KEY_DOWNLOAD_DIGEST, null) != manifest.sha256) return null
        return preferences.getLong(KEY_DOWNLOAD_ID, INVALID_DOWNLOAD_ID)
            .takeIf { it != INVALID_DOWNLOAD_ID }
    }

    fun enqueueDownload(manifest: ModelManifest, wifiOnly: Boolean): Long {
        manifest.validate()?.let { error ->
            throw IllegalArgumentException("Refusing to download invalid model manifest: $error")
        }
        if (isInstalled(manifest)) return DownloadManager.INVALID_DOWNLOAD_ID

        check((modelDirectory.isDirectory || modelDirectory.mkdirs()) && modelDirectory.canWrite()) {
            "App-specific model storage is unavailable or not writable"
        }
        val availableBytes = StatFs(modelDirectory.absolutePath).availableBytes
        require(availableBytes >= manifest.sizeBytes + STORAGE_HEADROOM_BYTES) {
            "Not enough free storage: need ${manifest.sizeLabel} plus temporary download space"
        }
        cancelStaleDownload(manifest.sha256)
        activeDownloadId(manifest)?.let { id ->
            when (query(id)) {
                is ModelDownloadState.Failed, ModelDownloadState.Missing, ModelDownloadState.Downloaded -> Unit
                else -> return id
            }
        }
        val partial = File(modelDirectory, "${manifest.fileName}.part")
        check(!partial.exists() || partial.delete()) {
            "Could not remove the incomplete model download before retrying"
        }

        val allowedNetworks = if (wifiOnly) {
            DownloadManager.Request.NETWORK_WIFI
        } else {
            DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE
        }
        val request = DownloadManager.Request(Uri.parse(manifest.downloadUrl))
            .setTitle(manifest.displayName)
            .setDescription("Downloading verified IntelHive worker model")
            .setMimeType("application/octet-stream")
            .setAllowedNetworkTypes(allowedNetworks)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationInExternalFilesDir(
                appContext,
                Environment.DIRECTORY_DOCUMENTS,
                "models/${manifest.fileName}.part"
            )
        val id = downloadManager.enqueue(request)
        preferences.edit()
            .putLong(KEY_DOWNLOAD_ID, id)
            .putString(KEY_DOWNLOAD_DIGEST, manifest.sha256)
            .apply()
        return id
    }

    fun query(downloadId: Long): ModelDownloadState {
        val cursor = downloadManager.query(DownloadManager.Query().setFilterById(downloadId))
            ?: return ModelDownloadState.Missing
        cursor.use {
            if (!it.moveToFirst()) return ModelDownloadState.Missing
            return when (it.getInt(it.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                DownloadManager.STATUS_PENDING -> ModelDownloadState.Pending("Waiting for network")
                DownloadManager.STATUS_RUNNING -> {
                    val downloaded = it.getLong(
                        it.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                    )
                    val total = it.getLong(
                        it.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                    )
                    ModelDownloadState.Downloading(downloaded, total)
                }
                DownloadManager.STATUS_PAUSED -> ModelDownloadState.Pending(
                    "Download paused: ${pauseReason(it.getInt(
                        it.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)
                    ))}"
                )
                DownloadManager.STATUS_SUCCESSFUL -> ModelDownloadState.Downloaded
                DownloadManager.STATUS_FAILED -> ModelDownloadState.Failed(
                    "Android download failed: ${failureReason(it.getInt(
                        it.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)
                    ))}"
                )
                else -> ModelDownloadState.Failed("Android reported an unknown download state")
            }
        }
    }

    fun cancel(downloadId: Long) {
        downloadManager.remove(downloadId)
        preferences.edit().remove(KEY_DOWNLOAD_ID).remove(KEY_DOWNLOAD_DIGEST).apply()
    }

    suspend fun verifyAndInstall(manifest: ModelManifest) = withContext(Dispatchers.IO) {
        val partial = File(modelDirectory, "${manifest.fileName}.part")
        require(partial.isFile) { "Downloaded model file is missing" }
        require(partial.length() == manifest.sizeBytes) {
            "Downloaded model size mismatch: expected ${manifest.sizeBytes}, got ${partial.length()}"
        }
        val actualDigest = sha256(partial)
        require("sha256:$actualDigest" == manifest.sha256) {
            "Downloaded model SHA-256 mismatch; refusing to install it"
        }

        val destination = installedFile(manifest)
        Files.move(
            partial.toPath(),
            destination.toPath(),
            StandardCopyOption.REPLACE_EXISTING
        )
        check(preferences.edit()
            .putString(KEY_VERIFIED_DIGEST, manifest.sha256)
            .putString(KEY_VERIFIED_VERSION, manifest.version)
            .remove(KEY_DOWNLOAD_ID)
            .remove(KEY_DOWNLOAD_DIGEST)
            .commit()
        ) { "Verified model state could not be saved" }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val digits = "0123456789abcdef"
        return buildString(64) {
            digest.digest().forEach { byte ->
                val value = byte.toInt() and 0xff
                append(digits[value ushr 4])
                append(digits[value and 0x0f])
            }
        }
    }

    private fun pauseReason(reason: Int): String = when (reason) {
        DownloadManager.PAUSED_WAITING_FOR_NETWORK -> "waiting for an allowed network"
        DownloadManager.PAUSED_WAITING_TO_RETRY -> "retrying"
        DownloadManager.PAUSED_QUEUED_FOR_WIFI -> "waiting for Wi-Fi"
        DownloadManager.PAUSED_UNKNOWN -> "unknown cause"
        else -> "reason $reason"
    }

    private fun failureReason(reason: Int): String = when (reason) {
        DownloadManager.ERROR_CANNOT_RESUME -> "cannot resume"
        DownloadManager.ERROR_DEVICE_NOT_FOUND -> "storage unavailable"
        DownloadManager.ERROR_FILE_ALREADY_EXISTS -> "destination already exists"
        DownloadManager.ERROR_FILE_ERROR -> "storage error"
        DownloadManager.ERROR_HTTP_DATA_ERROR -> "HTTP data error"
        DownloadManager.ERROR_INSUFFICIENT_SPACE -> "insufficient storage"
        DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "too many redirects"
        DownloadManager.ERROR_UNHANDLED_HTTP_CODE -> "unsupported HTTP response"
        DownloadManager.ERROR_UNKNOWN -> "unknown download error"
        else -> "reason $reason"
    }

    private fun String.isHttpsUrl(): Boolean =
        runCatching { Uri.parse(this).let { it.scheme == "https" && !it.host.isNullOrBlank() } }
            .getOrDefault(false)

    private fun cancelStaleDownload(expectedDigest: String) {
        val currentId = preferences.getLong(KEY_DOWNLOAD_ID, INVALID_DOWNLOAD_ID)
        if (currentId != INVALID_DOWNLOAD_ID &&
            preferences.getString(KEY_DOWNLOAD_DIGEST, null) != expectedDigest
        ) {
            downloadManager.remove(currentId)
            preferences.edit().remove(KEY_DOWNLOAD_ID).remove(KEY_DOWNLOAD_DIGEST).apply()
        }
    }

    companion object {
        private const val PREFERENCES = "required_model"
        private const val KEY_DOWNLOAD_ID = "download_id"
        private const val KEY_DOWNLOAD_DIGEST = "download_digest"
        private const val KEY_VERIFIED_DIGEST = "verified_digest"
        private const val KEY_VERIFIED_VERSION = "verified_version"
        private const val INVALID_DOWNLOAD_ID = -1L
        private const val MAX_MANIFEST_BYTES = 256 * 1024L
        private const val STORAGE_HEADROOM_BYTES = 256 * 1024 * 1024L

        fun defaultModelDirectory(context: Context): File =
            context.applicationContext.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
                ?.let { File(it, "models") }
                ?: error("App-specific external storage is unavailable")

        fun defaultModelFile(context: Context): File =
            File(defaultModelDirectory(context), ModelManifest.EXPECTED_FILE_NAME)
    }
}
