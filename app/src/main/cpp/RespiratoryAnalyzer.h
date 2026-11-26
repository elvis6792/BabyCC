#pragma once

#include <vector>
#include <utility>
#include <onnxruntime_cxx_api.h>
#include "FeatureExtractor.h"

class RespiratoryAnalyzer {
public:
    RespiratoryAnalyzer();
    ~RespiratoryAnalyzer();

    bool initModel(const void* model_data, size_t model_size);
    void setFIRKernel(const std::vector<float>& kernel);

    // Returns <wheeze_prob, cough_prob>
    std::pair<float, float> processAudio(const std::vector<float>& audio);

    // Returns <<wheeze_prob, cough_prob>, flat_mel_spectrogram>
    std::pair<std::pair<float, float>, std::vector<float>> processAudioWithMel(const std::vector<float>& audio);

private:
    void applyFIRFilter(std::vector<float>& audio);
    std::pair<float, float> runInference(const std::vector<float>& tensor_3ch);

    Ort::Env env;
    Ort::Session* session = nullptr;
    std::vector<uint8_t> model_buffer;

    std::vector<float> fir_kernel;
    FeatureExtractor feature_extractor{16000, 1024, 256, 80};

    // Config constants
    const int N_MELS = 80;
    const int TARGET_FRAMES = 157;
};
