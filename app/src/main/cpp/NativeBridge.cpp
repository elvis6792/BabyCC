#include <jni.h>
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "RespiratoryAnalyzer.h"

#define LOG_TAG "NativeBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static RespiratoryAnalyzer* analyzer = nullptr;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_hsu_baby_1cc_NativeAnalyzer_initModel(JNIEnv *env, jobject thiz, jobject asset_manager,
                                               jstring model_path) {
    if (analyzer != nullptr) {
        delete analyzer;
        analyzer = nullptr;
    }

    const char *path = env->GetStringUTFChars(model_path, nullptr);
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);

    AAsset *asset = AAssetManager_open(mgr, path, AASSET_MODE_BUFFER);
    env->ReleaseStringUTFChars(model_path, path);

    if (asset == nullptr) {
        LOGE("Failed to open model from assets");
        return JNI_FALSE;
    }

    const void *model_data = AAsset_getBuffer(asset);
    off_t model_size = AAsset_getLength(asset);

    analyzer = new RespiratoryAnalyzer();
    bool success = analyzer->initModel(model_data, model_size);

    AAsset_close(asset);

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_hsu_baby_1cc_NativeAnalyzer_setFIRKernel(JNIEnv *env, jobject thiz, jfloatArray fir_kernel) {
    if (analyzer == nullptr) return;

    jsize len = env->GetArrayLength(fir_kernel);
    if (len != 65) {
        LOGE("FIR kernel must be exactly 65 taps.");
        return;
    }

    jfloat *kernel_elements = env->GetFloatArrayElements(fir_kernel, nullptr);
    std::vector<float> kernel(kernel_elements, kernel_elements + len);
    env->ReleaseFloatArrayElements(fir_kernel, kernel_elements, JNI_ABORT);

    analyzer->setFIRKernel(kernel);
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_hsu_baby_1cc_NativeAnalyzer_processAudio(JNIEnv *env, jobject thiz, jfloatArray audio_data) {
    if (analyzer == nullptr) return nullptr;

    jsize len = env->GetArrayLength(audio_data);
    jfloat *audio_elements = env->GetFloatArrayElements(audio_data, nullptr);
    std::vector<float> audio(audio_elements, audio_elements + len);
    env->ReleaseFloatArrayElements(audio_data, audio_elements, JNI_ABORT);

    std::pair<float, float> probs = analyzer->processAudio(audio);

    jfloatArray result = env->NewFloatArray(2);
    float output[2] = {probs.first, probs.second};
    env->SetFloatArrayRegion(result, 0, 2, output);

    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_hsu_baby_1cc_NativeAnalyzer_processAudioWithMel(JNIEnv *env, jobject thiz,
                                                         jfloatArray audio_data) {
    if (analyzer == nullptr) return nullptr;

    jsize len = env->GetArrayLength(audio_data);
    jfloat *audio_elements = env->GetFloatArrayElements(audio_data, nullptr);
    std::vector<float> audio(audio_elements, audio_elements + len);
    env->ReleaseFloatArrayElements(audio_data, audio_elements, JNI_ABORT);

    auto [probs, mel] = analyzer->processAudioWithMel(audio);

    // 2 values for probs + mel size
    jsize result_size = 2 + mel.size();
    jfloatArray result = env->NewFloatArray(result_size);

    std::vector<float> output;
    output.reserve(result_size);
    output.push_back(probs.first);
    output.push_back(probs.second);
    output.insert(output.end(), mel.begin(), mel.end());

    env->SetFloatArrayRegion(result, 0, result_size, output.data());

    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_hsu_baby_1cc_NativeAnalyzer_release(JNIEnv *env, jobject thiz) {
    if (analyzer != nullptr) {
        delete analyzer;
        analyzer = nullptr;
    }
}
