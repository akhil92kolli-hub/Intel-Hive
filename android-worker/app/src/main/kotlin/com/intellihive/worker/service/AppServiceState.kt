package com.intellihive.worker.service

import android.app.ActivityManager
import android.content.Context

fun isAppServiceRunning(context: Context, serviceClass: Class<*>): Boolean {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    return activityManager.getRunningServices(Int.MAX_VALUE)
        .any { it.service.className == serviceClass.name }
}
