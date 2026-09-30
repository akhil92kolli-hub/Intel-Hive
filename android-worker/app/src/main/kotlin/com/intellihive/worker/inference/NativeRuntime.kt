package com.intellihive.worker.inference

/** Loads the packaged JNI library before a native shard executor is created. */
object NativeRuntime {
    private val loadError: String? = try {
        System.loadLibrary("intelhive_jni")
        if (!NativeBridge.nativeIsAvailable()) {
            "IntelHive JNI bridge did not report itself available"
        } else {
            null
        }
    } catch (error: LinkageError) {
        error.message ?: error.javaClass.simpleName
    }

    fun isAvailable(): Boolean = loadError == null

    fun unavailableReason(): String? = loadError
}
