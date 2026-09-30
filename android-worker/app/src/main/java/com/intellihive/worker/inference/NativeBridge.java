package com.intellihive.worker.inference;

/** Stable Java/JNI boundary used by the Kotlin execution layer. */
public final class NativeBridge {
    private NativeBridge() {}

    public static native boolean nativeIsAvailable();
    public static native long nativeLoadModel(String modelPath);
    public static native int nativeLayerCount(long modelHandle);
    public static native int nativeEmbeddingSize(long modelHandle);
    public static native long nativeCreateShard(long modelHandle, int firstLayer, int lastLayer);
    public static native byte[] nativeExecute(
        long shardHandle,
        String sequenceId,
        int tokenOffset,
        int tokenCount,
        long[] tokenIds,
        byte[] activation
    );
    public static native void nativeEndSequence(long shardHandle, String sequenceId);
    public static native void nativeDestroyShard(long shardHandle);
    public static native void nativeUnloadModel(long modelHandle);
}
