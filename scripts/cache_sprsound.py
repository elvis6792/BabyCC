"""
scripts/cache_sprsound.py (v2 - 方案 B)
來源：
  - 原 SPR Classification：70/30 hash 切分（train/holdout）
  - BioCAS 2023 / 2024：全數進 train
  - BioCAS 2025：永不出現（最終裁判）
"""
import json, hashlib, shutil
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from config import Config

BASE = Path('./data/SPRSound-main')
SOURCES = [
    ('spr', BASE / 'Classification' / 'train_classification_json',
            BASE / 'Classification' / 'train_classification_wav', 'hash'),
    ('b23', BASE / 'BioCAS2023' / 'test2023_json',
            BASE / 'BioCAS2023' / 'test2023_wav', 'all_train'),
    ('b24', BASE / 'BioCAS2024' / 'test2024_json',
            BASE / 'BioCAS2024' / 'test2024_wav', 'all_train'),
]
OUT_DIR = Path('./cache/spr_cache')
SALT, TRAIN_RATIO = 'spr_split', 0.7

WHEEZE_LABELS = {'wheeze', 'wheeze+crackle', 'stridor'}
NEG_LABELS = {'normal', 'fine crackle', 'coarse crackle'}


def keep_train(stem):
    b = int(hashlib.md5((SALT + stem).encode()).hexdigest(), 16) % 100
    return b < int(TRAIN_RATIO * 100)


def extract_mel(y, sr):
    win = Config.TARGET_SAMPLES
    if len(y) < win:
        pad = win - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode='reflect')
    elif len(y) > win:
        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        if len(rms) > 0:
            pk = int(np.argmax(rms)) * hop
            s = max(0, pk - win // 2)
            e = s + win
            if e > len(y):
                e, s = len(y), max(0, len(y) - win)
            y = y[s:e]
        else:
            y = y[:win]
    y_h, y_p = librosa.effects.hpss(y, margin=(3.0, 3.0))
    m_f = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS,
                                         n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    ref = float(np.max(m_f)) + 1e-8
    chans = []
    for yy in (y, y_h, y_p):
        m = librosa.feature.melspectrogram(y=yy, sr=sr, n_mels=Config.N_MELS,
                                           n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
        db = librosa.power_to_db(m, ref=ref)
        if db.shape[1] > Config.TARGET_FRAMES:
            db = db[:, :Config.TARGET_FRAMES]
        elif db.shape[1] < Config.TARGET_FRAMES:
            db = np.pad(db, ((0, 0), (0, Config.TARGET_FRAMES - db.shape[1])),
                        mode='constant', constant_values=-100.0)
        chans.append(torch.from_numpy(db).float())
    return torch.stack(chans)


def main():
    for sub in ('train', 'holdout'):          # 清舊快取，避免殘留
        d = OUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    counts = {'train': 0, 'holdout': 0}
    for tag, json_dir, wav_dir, mode in SOURCES:
        for jf in tqdm(sorted(json_dir.glob('*.json')), desc=f"[{tag}]"):
            try:
                meta = json.load(open(jf))
            except Exception:
                continue
            split = 'train' if mode == 'all_train' else \
                    ('train' if keep_train(jf.stem) else 'holdout')
            hits = list(wav_dir.glob(f"{jf.stem}.*"))
            if not hits:
                continue
            try:
                y, sr0 = librosa.load(str(hits[0]), sr=None, mono=True)
            except Exception:
                continue
            if sr0 != Config.SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr0, target_sr=Config.SAMPLE_RATE)
            for i, ann in enumerate(meta.get('event_annotation', [])):
                if not isinstance(ann, dict):
                    continue
                label = str(ann.get('type', ann.get('label', ''))).lower().strip()
                if label in WHEEZE_LABELS:
                    w = 1.0
                elif label in NEG_LABELS:
                    w = 0.0
                else:
                    continue                       # rhonchi 剔除
                s = int(float(ann.get('start', 0)) / 1000 * Config.SAMPLE_RATE)
                e = int(float(ann.get('end', 0)) / 1000 * Config.SAMPLE_RATE)
                seg = y[s:e]
                if len(seg) < int(0.1 * Config.SAMPLE_RATE):
                    continue
                torch.save({'mel': extract_mel(seg, Config.SAMPLE_RATE), 'wheeze': w},
                           OUT_DIR / split / f"{tag}_{jf.stem}_{i}.pt")
                counts[split] += 1

    print(f"SPR cache v2 完成：train={counts['train']}，holdout={counts['holdout']}")


if __name__ == '__main__':
    main()