import time
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure, label as ndlabel
import onnxruntime as ort
import json
import warnings
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config


class AsthmaInferencePipeline:
    def __init__(self, onnx_model_path: str, num_threads: int = 4):
        self.onnx_model_path = onnx_model_path
        self._init_engine(num_threads)
        
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.frames_100ms = max(2, int(np.ceil(0.10 * Config.SAMPLE_RATE / Config.HOP_LENGTH)))

    def _init_engine(self, num_threads: int):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(self.onnx_model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _pad_with_ambient_noise(self, y: np.ndarray, target_length: int, sr: int) -> np.ndarray:
        current_len = len(y)
        if current_len >= target_length:
            return y[:target_length]

        deficit = target_length - current_len
        pad_left = deficit // 2
        pad_right = deficit - pad_left

        edge_len = min(current_len, max(128, int(sr * 0.05)))
        head_rms = np.sqrt(np.mean(y[:edge_len] ** 2) + 1e-8)
        tail_rms = np.sqrt(np.mean(y[-edge_len:] ** 2) + 1e-8)
        bg_rms = float(min(head_rms, tail_rms))
        target_std = max(bg_rms, 1e-5)

        rng = np.random.default_rng(42)
        left_noise = rng.normal(0.0, target_std, pad_left).astype(np.float32)
        right_noise = rng.normal(0.0, target_std, pad_right).astype(np.float32)

        fade_len = min(int(sr * 0.005), current_len // 2, pad_left, pad_right)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            left_noise[-fade_len:] = left_noise[-fade_len:] * fade_out + y[:fade_len] * fade_in
            right_noise[:fade_len] = right_noise[:fade_len] * fade_in + y[-fade_len:] * fade_out

        return np.concatenate([left_noise, y, right_noise]).astype(np.float32)

    def _align_and_extract_mel(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        rms = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0]
        if len(rms) == 0 or np.max(rms) <= 1e-6:
            y_aligned = np.zeros(Config.TARGET_SAMPLES, dtype=np.float32)
        else:
            peak_height = max(float(np.mean(rms) + 0.1 * np.std(rms)), float(0.05 * np.max(rms)), 1e-4)
            peak_distance = max(1, int(0.25 * sr / Config.HOP_LENGTH))
            peaks, _ = find_peaks(rms, height=peak_height, distance=peak_distance)
            main_peak = int(peaks[np.argmax(rms[peaks])]) if len(peaks) > 0 else int(np.argmax(rms))
            peak_sample = main_peak * Config.HOP_LENGTH
            
            start_sample = max(0, peak_sample - int(Config.TARGET_SAMPLES * 0.30))
            end_sample = start_sample + Config.TARGET_SAMPLES
            
            if end_sample > len(y):
                start_sample = max(0, start_sample - (end_sample - len(y)))
                end_sample = len(y)
                
            y_out = y[start_sample:end_sample]
            y_aligned = self._pad_with_ambient_noise(y_out, Config.TARGET_SAMPLES, sr)

        y_harmonic, y_percussive = librosa.effects.hpss(y_aligned, margin=(3.0, 3.0))
        
        mel_args = {'sr': sr, 'n_mels': Config.N_MELS, 'n_fft': Config.N_FFT, 'hop_length': Config.HOP_LENGTH}
        mel_full = librosa.feature.melspectrogram(y=y_aligned, **mel_args)
        mel_harmonic = librosa.feature.melspectrogram(y=y_harmonic, **mel_args)
        mel_percussive = librosa.feature.melspectrogram(y=y_percussive, **mel_args)
        
        ref_max = max(float(np.max(mel_full)), 1e-8)
        
        def to_db_and_pad(mel):
            log_mel = librosa.power_to_db(mel, ref=ref_max)
            pad_val = min(float(np.min(log_mel)), -80.0) if log_mel.size else -100.0
            if log_mel.shape[1] < Config.TARGET_FRAMES:
                return np.pad(log_mel, ((0, 0), (0, Config.TARGET_FRAMES - log_mel.shape[1])), constant_values=pad_val)
            return log_mel[:, :Config.TARGET_FRAMES]

        tensor_mel = np.stack([
            to_db_and_pad(mel_full),
            to_db_and_pad(mel_harmonic),
            to_db_and_pad(mel_percussive)
        ]).astype(np.float32)

        tensor_mel = np.clip((tensor_mel + 100.0) / 100.0, 0.0, 1.0)
        tensor_mel = (tensor_mel - self.imagenet_mean) / self.imagenet_std

        return np.expand_dims(tensor_mel, axis=0), y_aligned

    def _run_xai_dsp(self, y: np.ndarray, sr: int) -> dict:
        stft_full = np.abs(librosa.stft(y, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=Config.N_FFT)
        n_frames = stft_full.shape[1]
        freq_bin_hz = sr / Config.N_FFT

        rms_raw = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0][:n_frames]
        max_rms = float(np.max(rms_raw)) + 1e-8
        max_rms_idx = int(np.argmax(rms_raw))

        breathing_mask = (rms_raw > max(0.02 * max_rms, 1e-6)) & (rms_raw < 0.50 * max_rms)
        valid_trajectories = []

        if int(np.sum(breathing_mask)) >= self.frames_100ms:
            stft_db = librosa.amplitude_to_db(stft_full, ref=np.max)
            band_mask = (freqs >= 200) & (freqs <= 3000)
            stft_band = stft_db[band_mask, :]

            freq_baseline = median_filter(stft_band, size=(9, 1))
            highlight = stft_band - freq_baseline
            noise_floor = float(np.percentile(highlight, 25))

            peaks_binary = highlight > max(6.0, noise_floor + 8.0)
            dilated = binary_dilation(peaks_binary, structure=generate_binary_structure(2, 2), iterations=1)
            labeled_map, n_components = ndlabel(dilated)
            cough_guard = max(2, int(0.08 * sr / Config.HOP_LENGTH))

            for comp_id in range(1, n_components + 1):
                pixels = np.argwhere(labeled_map == comp_id)
                if len(pixels) < self.frames_100ms: continue

                freq_idx_arr = pixels[:, 0]
                time_idx_arr = pixels[:, 1]
                t_min, t_max = int(np.min(time_idx_arr)), min(n_frames - 1, int(np.max(time_idx_arr)))

                if (t_max - t_min + 1) < self.frames_100ms: continue

                instant_bws = []
                for t in range(t_min, t_max + 1):
                    f_pts = freq_idx_arr[time_idx_arr == t]
                    if len(f_pts) > 0:
                        instant_bws.append((np.max(f_pts) - np.min(f_pts) + 1) * freq_bin_hz)

                if len(instant_bws) == 0 or np.median(instant_bws) > 300.0:
                    continue

                traj = []
                valid_times = []
                for t in range(t_min, t_max + 1):
                    if t >= n_frames or abs(t - max_rms_idx) < cough_guard: continue
                    frame_freq = freq_idx_arr[time_idx_arr == t]
                    if len(frame_freq) == 0: continue

                    w = np.clip(highlight[frame_freq, t], 0.0, None)
                    if np.sum(w) >= 1e-8:
                        traj.append(float(np.average(frame_freq, weights=w)))
                    else:
                        traj.append(float(np.mean(frame_freq)))
                    valid_times.append(t)

                if len(traj) < self.frames_100ms:
                    continue

                traj_arr = np.array(traj, dtype=float)
                t_axis = np.arange(len(traj_arr), dtype=float)

                coeffs = np.polyfit(t_axis, traj_arr, 2)
                residual_hz = float(np.std(traj_arr - np.polyval(coeffs, t_axis))) * freq_bin_hz

                if residual_hz < 150.0:
                    valid_trajectories.append(valid_times)

        tracks_count = len(valid_trajectories)

        if tracks_count > 8:
            tracks_count = 0
            valid_trajectories = []

        max_span_ms = max([len(tr) for tr in valid_trajectories]) * Config.HOP_LENGTH / sr * 1000 if tracks_count > 0 else 0

        return {
            "narrowband_tracks": tracks_count,
            "duration_ms": int(max_span_ms),
            "harmonics_groups": max(0, tracks_count - 1)
        }

    def _trim_silence(self, y: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
        """移除為了維度對齊而補上的尾部靜音 (Padding)，確保 XAI 能量計算正確"""
        active_indices = np.where(np.abs(y) > threshold)[0]
        if len(active_indices) > 0:
            return y[:active_indices[-1] + 1]
        return y

    def predict(self, y: np.ndarray, sr: int) -> dict:
        # 1. 執行模型推論
        input_tensor, _ = self._align_and_extract_mel(y, sr)
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        logits = self.session.run(None, ort_inputs)[0][0]

        probs = 1.0 / (1.0 + np.exp(-logits))

        wheeze_prob = float(probs[0])
        duration_prob = float(probs[1])
        type_prob = float(probs[2])

        is_wheeze = wheeze_prob > 0.5
        diagnosis = "Wheezing Detected" if is_wheeze else "No Wheeze"
        sound_type = "cough" if type_prob > 0.5 else "exhale"

        # 2. XAI 物理特徵提取
        xai_metrics = {"narrowband_tracks": 0, "duration_ms": 0, "harmonics_groups": 0}

        y_active = self._trim_silence(y)

        if len(y_active) > int(sr * 0.1):
            try:
                y_harmonic, _ = librosa.effects.hpss(y_active, margin=(3.0, 3.0))
                S_harmonic = np.abs(librosa.stft(y_harmonic, n_fft=1024, hop_length=256))

                rms_energy = librosa.feature.rms(S=S_harmonic, frame_length=1024, hop_length=256)[0]
                peaks, _ = find_peaks(rms_energy, height=np.mean(rms_energy) * 1.5)
                
                xai_metrics["narrowband_tracks"] = int(len(peaks))
                xai_metrics["duration_ms"] = int((len(y_active) / sr) * 1000)
                xai_metrics["harmonics_groups"] = 1 if len(peaks) > 2 else 0
            except Exception as e:
                warnings.warn(f"XAI DSP 解析失敗: {str(e)}")

        result = {
            "timestamp": int(time.time()),
            "sound_type": sound_type,
            "type_confidence": round(type_prob, 2),
            "clinical_assessment": {
                "diagnosis": diagnosis,
                "model_confidence": round(wheeze_prob, 2),
                "duration_score": round(duration_prob, 2),
                "xai_metrics": xai_metrics
            }
        }
        return result