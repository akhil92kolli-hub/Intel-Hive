package com.intellihive.worker.inference;

/** Stable Java/JNI boundary used by the Kotlin execution layer. */
public final class NativeBridge {
    private NativeBridge() {}

    public static native boolean nativeIsAvailable();
    public static native String nativeVulkanInfo();
    public static native String nativeRunLlamaCppBaseline(
        String modelPath,
        int cpuThreads,
        int prefillTokens,
        int generatedTokens,
        long maxDurationMs
    );
    public static native long nativeLoadModel(String modelPath);
    public static native int nativeLayerCount(long modelHandle);
    public static native int nativeEmbeddingSize(long modelHandle);
    public static native long[] nativeTokenize(long modelHandle, byte[] utf8Prompt);
    public static native long nativeCreateShard(
        long modelHandle,
        int firstLayer,
        int lastLayer,
        int graphThreads,
        boolean useVulkan
    );
    public static native long nativeLayerWeightBytes(long modelHandle, int firstLayer, int lastLayer);
    public static native boolean nativeShardUsedVulkan(long shardHandle);
    public static native String nativeShardBackendStats(long shardHandle);
    public static native byte[] nativeExecute(
        long shardHandle,
        String sequenceId,
        int tokenOffset,
        int tokenCount,
        long[] tokenIds,
        byte[] activation,
        long maxDurationMs
    );
    public static native void nativeEndSequence(long shardHandle, String sequenceId);
    public static native void nativeDestroyShard(long shardHandle);
    public static native void nativeUnloadModel(long modelHandle);
}
