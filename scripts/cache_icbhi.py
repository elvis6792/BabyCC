"""
scripts/cache_icbhi.py
將 ICBHI 特徵預計算為 .pt 快取，消滅訓練時的 on-the-fly DSP 瓶頸。
"""
import hashlib
import librosa
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from config import Config

ICBHI_DIR = Path('./data/ICBHI_final_database')
ICBHI_CACHE_DIR = Path('./cache/ICBHI_cache')


def cache_key(wav: str, start: float) -> str:
    return hashlib.md5(f"{wav}_{start}".encode()).hexdigest()


def compute_mel(y: np.ndarray, sr: int) -> torch.Tensor:
    target_len = Config.TARGET_SAMPLES
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    y_h, y_p = librosa.effects.hpss(y, margin=(3.0, 3.0))

    mel_f = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=Config.N_MELS,
        n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    mel_h = librosa.feature.melspectrogram(
        y=y_h, sr=sr, n_mels=Config.N_MELS,
        n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    mel_p = librosa.feature.melspectrogram(
        y=y_p, sr=sr, n_mels=Config.N_MELS,
        n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )

    ref = float(np.max(mel_f)) + 1e-8
    log_f = librosa.power_to_db(mel_f, ref=ref)
    log_h = librosa.power_to_db(mel_h, ref=ref)
    log_p = librosa.power_to_db(mel_p, ref=ref)

    def pt(m):
        if m.shape[1] > Config.TARGET_FRAMES:
            return m[:, :Config.TARGET_FRAMES]
        elif m.shape[1] < Config.TARGET_FRAMES:
            return np.pad(
                m, ((0, 0), (0, Config.TARGET_FRAMES - m.shape[1])),
                mode='constant', constant_values=-100.0
            )
        return m

    return torch.stack([
        torch.from_numpy(pt(log_f)).float(),
        torch.from_numpy(pt(log_h)).float(),
        torch.from_numpy(pt(log_p)).float()
    ])


def main():
    ICBHI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for wav_path in tqdm(sorted(ICBHI_DIR.glob('*.wav')), desc="Caching ICBHI"):
        txt_path = wav_path.with_suffix('.txt')
        if not txt_path.exists():
            continue
        try:
            y_full, sr = librosa.load(str(wav_path), sr=Config.SAMPLE_RATE, mono=True)
        except Exception:
            continue

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                except ValueError:
                    continue

                seg = y_full[int(start * sr):int(end * sr)]
                if len(seg) < int(sr * 0.1):
                    seg = np.zeros(int(sr * 0.1), dtype=np.float32)

                mel = compute_mel(seg, sr)
                key = cache_key(str(wav_path), start)
                torch.save({
                    'mel': mel,
                    'wheeze': float(wheeze),
                    'crackle': float(crackle),
                }, ICBHI_CACHE_DIR / f"{key}.pt")
                count += 1

    print(f"ICBHI 快取完成：{count} 段 -> {ICBHI_CACHE_DIR}")


if __name__ == "__main__":
    main()