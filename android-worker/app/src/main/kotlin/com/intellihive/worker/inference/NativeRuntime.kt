package com.intellihive.worker.inference

/** Loads the packaged JNI library before a native shard executor is created. */
object NativeRuntime {
    private val loaded: Boolean = runCatching {
        System.loadLibrary("intelhive_jni")
        nativeIsAvailable()
    }.getOrDefault(false)

    fun isAvailable(): Boolean = loaded

    private external fun nativeIsAvailable(): Boolean
}
