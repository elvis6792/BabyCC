#include "RespiratoryAnalyzer.h"
#include <cmath>
#include <android/log.h>
#include <numeric>
#include <stdexcept>

#define LOG_TAG "RespiratoryAnalyzer"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

RespiratoryAnalyzer::RespiratoryAnalyzer() : env(ORT_LOGGING_LEVEL_WARNING, "RespiratoryAnalyzer") {
}

RespiratoryAnalyzer::~RespiratoryAnalyzer() {
    if (session != nullptr) {
        delete session;
        session = nullptr;
    }
}

bool RespiratoryAnalyzer::initModel(const void* model_data, size_t model_size) {
    try {
        if (model_data == nullptr || model_size == 0) {
            LOGE("Invalid model data or size");
            return false;
        }

        model_buffer.assign((const uint8_t*)model_data, (const uint8_t*)model_data + model_size);

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = new Ort::Session(env, model_buffer.data(), model_buffer.size(), session_options);
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to load ONNX model: %s", e.what());
        return false;
    }
}

void RespiratoryAnalyzer::setFIRKernel(const std::vector<float>& kernel) {
    fir_kernel = kernel;
}

void RespiratoryAnalyzer::applyFIRFilter(std::vector<float>& audio) {
    if (fir_kernel.empty()) return;
    return; // Bypass FIR to keep raw audio consistent with PC inference

    float sum_abs = 0.0f;
    for (float v : fir_kernel) sum_abs += std::abs(v);
    if (sum_abs < 1e-4f) return;

    int audio_len = audio.size();
    int kernel_len = fir_kernel.size();
    std::vector<float> filtered(audio_len, 0.0f);

    for (int i = 0; i < audio_len; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_len; ++j) {
            if (i - j >= 0) {
                sum += audio[i - j] * fir_kernel[j];
            }
        }
        filtered[i] = sum;
    }
    audio = filtered;
}

std::pair<float, float> RespiratoryAnalyzer::processAudio(const std::vector<float>& audio) {
    auto res = processAudioWithMel(audio);
    return res.first;
}

std::pair<std::pair<float, float>, std::vector<float>> RespiratoryAnalyzer::processAudioWithMel(const std::vector<float>& audio) {
    std::vector<float> audio_copy = audio;
    applyFIRFilter(audio_copy);

    // Extract 3-channel HPSS Mel Spectrogram tensor [3 * 80 * 157]
    std::vector<float> tensor_3ch = feature_extractor.extractHPSSMelSpectrogram(audio_copy, TARGET_FRAMES, 3.0f);

    std::pair<float, float> probs = runInference(tensor_3ch);

    // Extract channel 0 (Full mel spectrogram) [80 * 157] for UI visualization
    std::vector<float> mel_full(tensor_3ch.begin(), tensor_3ch.begin() + N_MELS * TARGET_FRAMES);

    return {probs, mel_full};
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::pair<float, float> RespiratoryAnalyzer::runInference(const std::vector<float>& tensor_3ch) {
    if (session == nullptr) {
        return {0.0f, 0.0f};
    }

    size_t input_tensor_size = 3 * N_MELS * TARGET_FRAMES;
    if (tensor_3ch.size() < input_tensor_size) {
        return {0.0f, 0.0f};
    }

    std::vector<int64_t> input_node_dims = {1, 3, N_MELS, TARGET_FRAMES};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(tensor_3ch.data()), input_tensor_size, input_node_dims.data(), input_node_dims.size());

    const char* input_names[] = {"mel"};
    const char* output_names[] = {"wheeze_logit", "cough_logit"};

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 2);

    float wheeze_logit = *output_tensors[0].GetTensorMutableData<float>();
    float cough_logit = *output_tensors[1].GetTensorMutableData<float>();

    float wheeze_prob = sigmoid(wheeze_logit);
    float cough_prob = sigmoid(cough_logit);

    LOGI("AI Inference Result -> Wheeze(w): %.4f, Cough(c): %.4f (logits: w=%.2f, c=%.2f)",
         wheeze_prob, cough_prob, wheeze_logit, cough_logit);

    return {wheeze_prob, cough_prob};
}
