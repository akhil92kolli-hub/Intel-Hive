#include <android/log.h>
#include <jni.h>

#include "llama.h"

namespace {
constexpr const char* kLogTag = "IntelHiveJNI";
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM*, void*) {
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "IntelHive native layer-range library loaded");
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_intellihive_worker_inference_NativeRuntime_nativeIsAvailable(JNIEnv*, jobject) {
    // Referencing llama.cpp here verifies that the packaged JNI target links the
    // same native backend as the host feasibility test.
    return llama_print_system_info() != nullptr ? JNI_TRUE : JNI_FALSE;
}
