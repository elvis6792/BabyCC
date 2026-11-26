import os
import glob
import hashlib
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import (
    median_filter,
    binary_dilation,
    generate_binary_structure,
    label as ndlabel
)
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import Config

warnings.filterwarnings('ignore')


SYNTH_DIR = './data/synthetic_wheeze_v3'


def is_synthetic_data(filepath: str) -> bool:
    """判斷是否為合成資料（檔名包含 synth_wheeze）"""
    return 'synth_wheeze' in str(filepath).lower()


def pad_with_ambient_noise(
    y: np.ndarray,
    target_length: int,
    sr: int,
    seed_key: str = ""
) -> np.ndarray:
    """
    RMS-guided ambient noise synthesis padding.
    Replaces deterministic tiling to prevent periodic artifact mirroring in spectrograms.
    """
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

    if seed_key:
        seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) % (2**32)
    else:
        seed = 42
    rng = np.random.default_rng(seed)

    left_noise = rng.normal(0.0, target_std, pad_left).astype(np.float32)
    right_noise = rng.normal(0.0, target_std, pad_right).astype(np.float32)

    fade_len = min(int(sr * 0.005), current_len // 2, pad_left, pad_right)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        left_noise[-fade_len:] = left_noise[-fade_len:] * fade_out + y[:fade_len] * fade_in
        right_noise[:fade_len] = right_noise[:fade_len] * fade_in + y[-fade_len:] * fade_out

    return np.concatenate([left_noise, y, right_noise]).astype(np.float32)


def compute_snr(y):
    if len(y) == 0:
        return 0.0
    abs_y = np.abs(y)
    signal_rms = np.percentile(abs_y, 90)
    noise_rms = np.percentile(abs_y, 10)
    if noise_rms < 1e-6:
        noise_rms = 1e-6
    return 20.0 * np.log10(signal_rms / noise_rms)


def get_teacher_asthma_scores(y, sr=Config.SAMPLE_RATE):
    """
    氣喘專用 2D teacher scores (Wheezing, Duration).
    """
    features = np.zeros(2, dtype=np.float32)
    try:
        stft_full = np.abs(librosa.stft(y, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=Config.N_FFT)
        n_frames = stft_full.shape[1]

        rms_raw = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0][:n_frames]
        if len(rms_raw) == 0:
            return features

        max_rms_raw = float(np.max(rms_raw)) + 1e-8
        max_rms_idx = int(np.argmax(rms_raw))
        noise_rms = float(np.percentile(rms_raw, 10)) if len(rms_raw) > 0 else 0.0

        active_threshold = max(0.10 * max_rms_raw, noise_rms * 2.0, 1e-6)
        active_mask = rms_raw > active_threshold
        if np.sum(active_mask) == 0:
            return features

        # ── 0. Wheezing (連續喘鳴音) ──
        freq_bin_hz = sr / Config.N_FFT
        min_wheeze_frames = max(4, int(np.ceil(0.15 * sr / Config.HOP_LENGTH)))
        min_breathing_frames = max(8, int(np.ceil(0.30 * sr / Config.HOP_LENGTH)))

        breathing_mask = ((rms_raw > max(0.02 * max_rms_raw, noise_rms * 1.5, 1e-6))
                          & (rms_raw < 0.50 * max_rms_raw))
        breathing_frames = int(np.sum(breathing_mask))

        if breathing_frames >= min_breathing_frames:
            stft_db = librosa.amplitude_to_db(stft_full, ref=np.max)
            band_mask = (freqs >= 200) & (freqs <= 3000)
            stft_band = stft_db[band_mask, :]

            freq_baseline = median_filter(stft_band, size=(9, 1))
            highlight = stft_band - freq_baseline

            breathing_cols = np.where(breathing_mask[:stft_band.shape[1]])[0]
            if len(breathing_cols) > 0:
                noise_floor = float(np.percentile(highlight[:, breathing_cols], 25))
            else:
                noise_floor = float(np.percentile(highlight, 25))

            peaks_binary = highlight > max(6.0, noise_floor + 8.0)
            dilated = binary_dilation(peaks_binary, structure=generate_binary_structure(2, 2), iterations=1)
            labeled_map, n_components = ndlabel(dilated)

            wheeze_frames_set = np.zeros(n_frames, dtype=bool)
            cough_guard = max(2, int(0.08 * sr / Config.HOP_LENGTH))

            for comp_id in range(1, n_components + 1):
                pixels = np.argwhere(labeled_map == comp_id)
                if len(pixels) < max(10, min_wheeze_frames):
                    continue

                freq_idx_arr, time_idx_arr = pixels[:, 0], pixels[:, 1]
                t_min = int(np.min(time_idx_arr))
                t_max = min(n_frames - 1, int(np.max(time_idx_arr)))
                span = max(1, t_max - t_min + 1)

                if span < min_wheeze_frames:
                    continue
                if int(np.sum(breathing_mask[t_min:t_max + 1])) < min_wheeze_frames:
                    continue

                bandwidth_hz = (np.max(freq_idx_arr) - np.min(freq_idx_arr) + 1) * freq_bin_hz
                if bandwidth_hz > 180.0:
                    continue

                traj = []
                valid_times = []
                for t in range(t_min, t_max + 1):
                    if t >= n_frames or not breathing_mask[t]:
                        continue
                    if abs(t - max_rms_idx) < cough_guard:
                        continue
                    frame_freq = freq_idx_arr[time_idx_arr == t]
                    if len(frame_freq) == 0:
                        continue
                    w = np.clip(highlight[frame_freq, t], 0.0, None)
                    traj.append(
                        float(np.average(frame_freq, weights=w))
                        if np.sum(w) >= 1e-8
                        else float(np.mean(frame_freq))
                    )
                    valid_times.append(t)

                if len(traj) < min_wheeze_frames or (len(valid_times) / float(span)) < 0.6:
                    continue

                traj_arr = np.array(traj, dtype=float)
                jumps_hz = np.abs(np.diff(traj_arr)) * freq_bin_hz
                if np.sum(jumps_hz > 50.0) > len(traj_arr) * 0.2:
                    continue

                t_axis = np.arange(len(traj_arr), dtype=float)
                coeffs = np.polyfit(t_axis, traj_arr, 1)
                residual_hz = float(np.std(traj_arr - np.polyval(coeffs, t_axis))) * freq_bin_hz
                range_hz = float(np.max(traj_arr) - np.min(traj_arr)) * freq_bin_hz
                slope_hz_per_sec = float(coeffs[0]) * freq_bin_hz * (sr / Config.HOP_LENGTH)

                if (residual_hz < 70.0 and range_hz < 200.0) or \
                   (residual_hz < 110.0 and abs(slope_hz_per_sec) < 150.0):
                    wheeze_frames_set[valid_times] = True

            wheeze_ratio = float(np.sum(wheeze_frames_set)) / float(max(1, breathing_frames))
            evidence = min(1.0, breathing_frames / float(max(1, int(0.7 * sr / Config.HOP_LENGTH))))
            features[0] = min(1.0, wheeze_ratio * evidence * 0.7)

        # ── 1. Duration (有效音長) ──
        active_duration = float(np.sum(active_mask) * Config.HOP_LENGTH / sr)
        features[1] = min(1.0, active_duration / 1.0)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, 0.0, 1.0)
    except Exception as e:
        print(f"[Warning] teacher score failed: {e}")
    return features


def _pad_or_trim_mel(
    log_mel: np.ndarray,
    target_frames: int,
    pad_value: float = -100.0
) -> np.ndarray:
    """
    Ensure the Log-Mel spectrogram strictly matches the target time frames.
    """
    current_frames = log_mel.shape[1]
    if current_frames > target_frames:
        return log_mel[:, :target_frames]
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        return np.pad(
            log_mel,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=pad_value
        )
    return log_mel


def infer_type_score(file_path_obj: Path) -> float:
    """
    推斷 cough / exhale type score。
    """
    path_str = str(file_path_obj).lower()

    if 'synth_wheeze_cough' in path_str:
        return 1.0
    if 'synth_wheeze_exhale' in path_str:
        return 0.0

    # UK 原始資料
    if 'cough' in path_str:
        return 1.0
    if 'exhale' in path_str or 'exhalation' in path_str:
        return 0.0

    raise ValueError(
        f"嚴重錯誤：無法從路徑辨識聲音類型 (Cough/Exhale)，"
        f"請檢查資料夾結構！路徑：{file_path_obj}"
    )


def process_and_save(filepath: str, save_dir: str) -> bool:
    """
    Extract physics-informed HPSS 3-channel acoustic features and teacher scores.
    """
    try:
        file_path_obj = Path(filepath)
        save_dir_obj = Path(save_dir)
        save_dir_obj.mkdir(parents=True, exist_ok=True)

        y_raw, sr = librosa.load(str(file_path_obj), sr=Config.SAMPLE_RATE)
        if len(y_raw) == 0:
            return False

        synthetic = is_synthetic_data(filepath)

        if synthetic and len(y_raw) <= int(Config.TARGET_SAMPLES * 1.3):
            y = y_raw
        else:
            if not synthetic:
                if compute_snr(y_raw) < 15.0:
                    return False

            rms = librosa.feature.rms(
                y=y_raw,
                frame_length=Config.N_FFT,
                hop_length=Config.HOP_LENGTH
            )[0]
            if len(rms) == 0 or np.max(rms) <= 1e-6:
                return False

            if synthetic:
                # 合成資料：直接用 RMS 最大值作為 peak（避免 find_peaks 找不到弱喘鳴）
                main_peak = int(np.argmax(rms))
            else:
                # UK 原始資料：嚴格 peak 檢測
                peak_height = max(
                    float(np.mean(rms) + 0.1 * np.std(rms)),
                    float(0.05 * np.max(rms)),
                    1e-4
                )
                peak_distance = max(1, int(0.25 * sr / Config.HOP_LENGTH))
                peaks, _ = find_peaks(
                    rms,
                    height=peak_height,
                    distance=peak_distance
                )
                if len(peaks) == 0:
                    return False
                main_peak = int(peaks[np.argmax(rms[peaks])])

            peak_sample = main_peak * Config.HOP_LENGTH

            # 保留更多 peak 後段，對 whooping / decay 較友善
            peak_left = int(Config.TARGET_SAMPLES * 0.30)
            start_sample = peak_sample - peak_left
            end_sample = start_sample + Config.TARGET_SAMPLES

            if start_sample < 0:
                end_sample = min(len(y_raw), end_sample - start_sample)
                start_sample = 0
            if end_sample > len(y_raw):
                start_sample = max(0, start_sample - (end_sample - len(y_raw)))
                end_sample = len(y_raw)

            y = y_raw[start_sample:end_sample]

        if len(y) < Config.TARGET_SAMPLES:
            y = pad_with_ambient_noise(
                y, Config.TARGET_SAMPLES, sr,
                seed_key=str(file_path_obj)
            )
        elif len(y) > Config.TARGET_SAMPLES:
            y = y[:Config.TARGET_SAMPLES]

        # ── 嚴格組合 3D 特徵 [Wheezing, Duration, SoundType] ──
        teacher_scores = get_teacher_asthma_scores(y, sr)
        type_score = infer_type_score(file_path_obj)
        teacher_scores = np.append(teacher_scores, type_score).astype(np.float32)

        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(3.0, 3.0))

        mel_full = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        mel_harmonic = librosa.feature.melspectrogram(
            y=y_harmonic, sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        mel_percussive = librosa.feature.melspectrogram(
            y=y_percussive, sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )

        ref_max = float(np.max(mel_full)) + 1e-8
        log_mel_full = librosa.power_to_db(mel_full, ref=ref_max)
        log_mel_harmonic = librosa.power_to_db(mel_harmonic, ref=ref_max)
        log_mel_percussive = librosa.power_to_db(mel_percussive, ref=ref_max)

        pad_value = float(np.min(log_mel_full)) if log_mel_full.size else -100.0
        pad_value = min(pad_value, -80.0)

        log_mel_full = _pad_or_trim_mel(log_mel_full, Config.TARGET_FRAMES, pad_value=pad_value)
        log_mel_harmonic = _pad_or_trim_mel(log_mel_harmonic, Config.TARGET_FRAMES, pad_value=pad_value)
        log_mel_percussive = _pad_or_trim_mel(log_mel_percussive, Config.TARGET_FRAMES, pad_value=pad_value)

        tensor_mel = torch.stack([
            torch.from_numpy(log_mel_full).float(),
            torch.from_numpy(log_mel_harmonic).float(),
            torch.from_numpy(log_mel_percussive).float()
        ])

        save_path = save_dir_obj / f"{file_path_obj.stem}.pt"
        torch.save(
            {
                'mel': tensor_mel,
                'scores': torch.from_numpy(teacher_scores)
            },
            str(save_path)
        )
        return True
    except Exception as e:
        print(f"\n[Warning] 檔案處理失敗 {filepath}: {str(e)}")
        return False


def main():
    print("==========================================================")
    print("啟動第一階段：DSP 特徵提取（合成資料整合版）")
    print("==========================================================")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 原始 UK 資料
    all_files = glob.glob(
        os.path.join(Config.UK_AUDIO_DIR, "**/*.wav"),
        recursive=True
    )

    if os.path.exists(SYNTH_DIR):
        synth_files = glob.glob(os.path.join(SYNTH_DIR, "*.wav"))
        all_files.extend(synth_files)
        print(f"加入合成資料：{len(synth_files)} 筆")
    else:
        print(f"[Warning] 合成資料路徑不存在：{SYNTH_DIR}")

    if not all_files:
        print("[Error] 找不到任何 WAV 檔案，請檢查 Config.UK_AUDIO_DIR 和 SYNTH_DIR！")
        return

    print(f"總計待處理：{len(all_files)} 筆")

    max_cores = max(1, os.cpu_count() - 1)
    print(f"啟動 {max_cores} 個進程進行平行特徵提取...")

    success_count = 0
    synth_success = 0

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {
            executor.submit(process_and_save, f, Config.CACHE_DIR): f
            for f in all_files
        }
        for fut in tqdm(as_completed(futures), total=len(all_files), desc="Extracting & Saving"):
            try:
                src = futures[fut]
                if fut.result():
                    success_count += 1
                    if is_synthetic_data(src):
                        synth_success += 1
            except Exception as e:
                print(f"進程崩潰: {e}")

    print(f"\n預處理完成！成功提取 {success_count} 筆有效特徵。")
    print(f"   其中合成資料：{synth_success} 筆")


if __name__ == "__main__":
    main()