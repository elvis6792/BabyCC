#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <complex>

class FeatureExtractor {
public:
    FeatureExtractor(int sample_rate = 16000, int n_fft = 1024, int hop_length = 256, int n_mels = 80);
    ~FeatureExtractor();

    // Outputs [3, N_MELS, TARGET_FRAMES] flattened vector.
    // Channel order: Full, Harmonic, Percussive; raw dB values.
    std::vector<float> extractHPSSMelSpectrogram(const std::vector<float>& audio, int target_frames, float margin = 3.0f);

private:
    int sr_;
    int n_fft_;
    int hop_length_;
    int n_mels_;

    std::vector<float> window_;
    std::vector<std::vector<float>> mel_basis_;

    void initHannWindow();
    void initMelBasis();
    void computeFFT(std::vector<std::complex<float>>& x);
    std::vector<std::vector<float>> computeSTFTMagnitude(const std::vector<float>& audio);
    std::vector<float> medianFilter1D(const std::vector<float>& input, int kernel_size);
    std::vector<std::vector<float>> applyMelFilter(const std::vector<std::vector<float>>& mag_spec);
    std::vector<std::vector<float>> powerToDb(const std::vector<std::vector<float>>& mel_spec, float ref_max);
};

#endif // FEATURE_EXTRACTOR_H
