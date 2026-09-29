package com.intellihive.worker.benchmark

import com.google.gson.Gson
import com.intellihive.worker.BuildConfig
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class SupabaseBenchmarkUploader(
    private val client: OkHttpClient = OkHttpClient(),
    private val gson: Gson = Gson()
) {
    suspend fun upload(row: Map<String, Any?>) = withContext(Dispatchers.IO) {
        val endpoint = BuildConfig.SUPABASE_URL.trimEnd('/') + "/rest/v1/benchmark_results"
        val key = BuildConfig.SUPABASE_PUBLISHABLE_KEY
        check(endpoint.startsWith("https://") && key.startsWith("sb_publishable_")) {
            "Supabase benchmark upload is not configured"
        }

        val request = Request.Builder()
            .url(endpoint)
            .header("apikey", key)
            .header("Authorization", "Bearer $key")
            .header("Prefer", "return=minimal")
            .post(gson.toJson(row).toRequestBody(JSON_MEDIA_TYPE))
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val detail = response.body?.string()?.take(512).orEmpty()
                throw IOException(
                    "Supabase benchmark upload failed with HTTP ${response.code}" +
                        if (detail.isBlank()) "" else ": $detail"
                )
            }
        }
    }

    companion object {
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}
