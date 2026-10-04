package com.intellihive.worker.benchmark

import com.google.gson.Gson
import com.intellihive.worker.BuildConfig
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class SupabaseBenchmarkUploader(
    private val client: OkHttpClient = OkHttpClient(),
    private val gson: Gson = Gson()
) {
    suspend fun uploadSuite(row: Map<String, Any?>) = upload("benchmark_results", row)

    suspend fun uploadTest(row: Map<String, Any?>) = upload("benchmark_test_results", row)

    @Deprecated("Use uploadSuite for the suite row")
    suspend fun upload(row: Map<String, Any?>) = uploadSuite(row)

    private suspend fun upload(table: String, row: Map<String, Any?>) =
        withContext(Dispatchers.IO) {
            val conflictColumn = if (table == "benchmark_results") "client_run_id" else "client_test_id"
            val endpoint = BuildConfig.SUPABASE_URL.trimEnd('/') +
                "/rest/v1/$table?on_conflict=$conflictColumn"
            val key = BuildConfig.SUPABASE_PUBLISHABLE_KEY
            check(endpoint.startsWith("https://") && key.startsWith("sb_publishable_")) {
                "Supabase benchmark upload is not configured"
            }

            var lastFailure: IOException? = null
            for (attempt in 0 until MAX_ATTEMPTS) {
                try {
                    val request = Request.Builder()
                        .url(endpoint)
                        .header("apikey", key)
                        .header("Prefer", "resolution=ignore-duplicates,return=minimal")
                        .post(gson.toJson(row).toRequestBody(JSON_MEDIA_TYPE))
                        .build()

                    client.newCall(request).execute().use { response ->
                        if (!response.isSuccessful) {
                            val detail = response.body?.string()?.take(512).orEmpty()
                            throw UploadHttpException(
                                response.code,
                                "Supabase $table upload failed with HTTP ${response.code}" +
                                    if (detail.isBlank()) "" else ": $detail"
                            )
                        }
                    }
                    return@withContext
                } catch (error: IOException) {
                    lastFailure = error
                    if ((error is UploadHttpException &&
                            error.statusCode in 400..499 && error.statusCode != 429) ||
                        attempt == MAX_ATTEMPTS - 1
                    ) {
                        throw error
                    }
                    delay(RETRY_DELAYS_MS[attempt])
                }
            }
            throw checkNotNull(lastFailure)
        }

    companion object {
        private const val MAX_ATTEMPTS = 3
        private val RETRY_DELAYS_MS = longArrayOf(1_000L, 4_000L)
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}

private class UploadHttpException(
    val statusCode: Int,
    message: String
) : IOException(message)
