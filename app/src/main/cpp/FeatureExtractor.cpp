#include "FeatureExtractor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <android/log.h>
#include "kiss_fft.h"

#define LOG_TAG "FeatureExtractor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeatureExtractor::FeatureExtractor(int sample_rate, int n_fft, int hop_length, int n_mels)
    : sr_(sample_rate), n_fft_(n_fft), hop_length_(hop_length), n_mels_(n_mels) {
    initHannWindow();
    initMelBasis();
}

FeatureExtractor::~FeatureExtractor() {}

void FeatureExtractor::initHannWindow() {
    window_.resize(n_fft_);
    for (int i = 0; i < n_fft_; ++i) {
        window_[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / n_fft_); // Symmetric Hann
    }
}

void FeatureExtractor::initMelBasis() {
    mel_basis_.assign(n_mels_, std::vector<float>(n_fft_ / 2 + 1, 0.0f));
    float f_min = 0.0f; // Align with librosa default fmin=0.0
    float f_max = sr_ / 2.0f;

    // Slaney Mel Scale (librosa default htk=False)
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp; // 15.0f
    const float logstep = std::log(6.4f) / 27.0f;

    auto hz_to_mel = [&](float hz) {
        if (hz < min_log_hz) return (hz - f_min) / f_sp;
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    };

    auto mel_to_hz = [&](float mel) {
        if (mel < min_log_mel) return f_min + f_sp * mel;
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    };

    float min_mel = hz_to_mel(f_min);
    float max_mel = hz_to_mel(f_max);

    std::vector<float> mel_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; ++i) {
        mel_points[i] = mel_to_hz(min_mel + i * (max_mel - min_mel) / (n_mels_ + 1));
    }

    std::vector<float> fft_freqs(n_fft_ / 2 + 1);
    for (int i = 0; i <= n_fft_ / 2; ++i) {
        fft_freqs[i] = i * static_cast<float>(sr_) / n_fft_;
    }

    for (int i = 0; i < n_mels_; ++i) {
        float f_m_minus = mel_points[i];
        float f_m = mel_points[i + 1];
        float f_m_plus = mel_points[i + 2];
        for (int j = 0; j <= n_fft_ / 2; ++j) {
            float f = fft_freqs[j];
            float lower = (f - f_m_minus) / (f_m - f_m_minus);
            float upper = (f_m_plus - f) / (f_m_plus - f_m);
            mel_basis_[i][j] = std::max(0.0f, std::min(lower, upper));
        }
    }

    // Slaney Area Normalization (librosa norm='slaney')
    for (int i = 0; i < n_mels_; ++i) {
        float enorm = 2.0f / (mel_points[i + 2] - mel_points[i]);
        for (int j = 0; j <= n_fft_ / 2; ++j) {
            mel_basis_[i][j] *= enorm;
        }
    }
}

void FeatureExtractor::computeFFT(std::vector<std::complex<float>>& x) {
    int n = x.size();
    if (n <= 1) return;

    std::vector<std::complex<float>> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }
    computeFFT(even);
    computeFFT(odd);

    for (int k = 0; k < n / 2; ++k) {
        std::complex<float> t = std::polar(1.0f, -2.0f * static_cast<float>(M_PI) * k / n) * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
    }
}

std::vector<std::vector<float>> FeatureExtractor::computeSTFTMagnitude(const std::vector<float>& audio) {
    int pad_len = n_fft_ / 2; // reflect padding (librosa center=True)
    std::vector<float> padded(audio.size() + 2 * pad_len);

    if (audio.size() > 1) {
        for (int i = 0; i < pad_len; ++i) padded[pad_len - 1 - i] = audio[i + 1];
        for (size_t i = 0; i < audio.size(); ++i) padded[pad_len + i] = audio[i];
        for (int i = 0; i < pad_len; ++i) padded[pad_len + audio.size() + i] = audio[audio.size() - 2 - i];
    } else {
        std::fill(padded.begin(), padded.end(), 0.0f);
    }

    int n_frames = 1 + (padded.size() - n_fft_) / hop_length_;
    std::vector<std::vector<float>> mag(n_fft_ / 2 + 1, std::vector<float>(n_frames, 0.0f));
    std::vector<std::complex<float>> buf(n_fft_);

    for (int t = 0; t < n_frames; ++t) {
        int start = t * hop_length_;
        for (int i = 0; i < n_fft_; ++i) {
            buf[i] = std::complex<float>(padded[start + i] * window_[i], 0.0f);
        }
        computeFFT(buf);
        for (int f = 0; f <= n_fft_ / 2; ++f) {
            mag[f][t] = std::abs(buf[f]);
        }
    }
    return mag;
}

std::vector<float> FeatureExtractor::medianFilter1D(const std::vector<float>& input, int kernel_size) {
    int n = input.size();
    std::vector<float> output(n);
    int pad = kernel_size / 2;
    std::vector<float> win(kernel_size);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int idx = i + j - pad;
            if (idx < 0) idx = -idx;
            else if (idx >= n) idx = 2 * n - 2 - idx;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            win[j] = input[idx];
        }
        std::nth_element(win.begin(), win.begin() + pad, win.end());
        output[i] = win[pad];
    }
    return output;
}

std::vector<std::vector<float>> FeatureExtractor::applyMelFilter(const std::vector<std::vector<float>>& mag) {
    int n_frames = mag[0].size();
    std::vector<std::vector<float>> mel(n_mels_, std::vector<float>(n_frames, 0.0f));
    for (int t = 0; t < n_frames; ++t) {
        for (int m = 0; m < n_mels_; ++m) {
            float sum = 0.0f;
            for (size_t f = 0; f < mag.size(); ++f) {
                sum += mel_basis_[m][f] * (mag[f][t] * mag[f][t]); // power = mag^2
            }
            mel[m][t] = sum;
        }
    }
    return mel;
}

std::vector<std::vector<float>> FeatureExtractor::powerToDb(const std::vector<std::vector<float>>& mel, float ref_max) {
    int n_frames = mel[0].size();
    std::vector<std::vector<float>> db(n_mels_, std::vector<float>(n_frames, 0.0f));
    float ref_log = 10.0f * std::log10(std::max(1e-10f, ref_max));

    for (int m = 0; m < n_mels_; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            float val = 10.0f * std::log10(std::max(1e-10f, mel[m][t])) - ref_log;
            db[m][t] = std::max(val, -80.0f); // Standard librosa power_to_db(S, ref=ref, top_db=80.0)
        }
    }
    return db;
}

std::vector<float> FeatureExtractor::extractHPSSMelSpectrogram(const std::vector<float>& audio, int target_frames, float margin) {
    auto mag = computeSTFTMagnitude(audio);
    int n_freqs = mag.size();
    int n_frames = mag[0].size();

    auto H = mag;
    auto P = mag;
    const int kernel_size = 31; // librosa hpss default kernel

    for (int f = 0; f < n_freqs; ++f) {
        H[f] = medianFilter1D(mag[f], kernel_size);
    }
    for (int t = 0; t < n_frames; ++t) {
        std::vector<float> col(n_freqs);
        for (int f = 0; f < n_freqs; ++f) col[f] = mag[f][t];
        std::vector<float> fc = medianFilter1D(col, kernel_size);
        for (int f = 0; f < n_freqs; ++f) P[f][t] = fc[f];
    }

    // Standard librosa.effects.hpss(margin=(3.0, 3.0)): softmask
    const float margin_sq = margin * margin; // 3.0 * 3.0 = 9.0f
    for (int f = 0; f < n_freqs; ++f) {
        for (int t = 0; t < n_frames; ++t) {
            float h2 = H[f][t] * H[f][t];
            float p2 = P[f][t] * P[f][t];
            float m_h = h2 / (h2 + margin_sq * p2 + 1e-10f);
            float m_p = p2 / (p2 + margin_sq * h2 + 1e-10f);
            H[f][t] = mag[f][t] * m_h;
            P[f][t] = mag[f][t] * m_p;
        }
    }

    auto Fm = applyMelFilter(mag);
    auto Hm = applyMelFilter(H);
    auto Pm = applyMelFilter(P);

    float ref_max = 0.0f; // Global Full channel peak power
    for (int m = 0; m < n_mels_; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            if (Fm[m][t] > ref_max) ref_max = Fm[m][t];
        }
    }

    float ref_log = 10.0f * std::log10(std::max(1e-10f, ref_max));
    LOGI("FE Baseline | ref_log=%.2f", ref_log);

    auto Fd = powerToDb(Fm, ref_max);
    auto Hd = powerToDb(Hm, ref_max);
    auto Pd = powerToDb(Pm, ref_max);

    std::vector<float> out(3 * n_mels_ * target_frames, -100.0f); // pad value = -100.0f

    for (int c = 0; c < 3; ++c) {
        const auto& spec = (c == 0) ? Fd : ((c == 1) ? Hd : Pd);
        for (int m = 0; m < n_mels_; ++m) {
            for (int t = 0; t < std::min(n_frames, target_frames); ++t) {
                out[c * (n_mels_ * target_frames) + m * target_frames + t] = spec[m][t];
            }
        }
    }

    return out;
}
