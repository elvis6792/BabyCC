"""
scripts/recut_three_cough.py (v2)
效能優化：
  1. 開頭一次性建立 stem→path 索引（拔除 per-file rglob 爬行）
  2. 原碼率解碼、只對切段 resample（拔除整檔重採樣）
  3. ProcessPoolExecutor 平行處理
"""
import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import Config
from step1_extract_features import process_and_save

UK_RAW_DIR = Path('./data/UK_COVID19/audio')
METADATA_CSV = Path('./data/UK_COVID19/audio_metadata.csv')
OUT_COUGH_DIR = Path('./data/UK_audio_preprocessed/cough')
CACHE_DIR = Path(Config.CACHE_DIR)

GAP_ALLOW_SEC = 0.8
TAIL_SEC = 0.35
MAX_LEN_SEC = 5.0


def build_index():
    idx = {}
    for p in UK_RAW_DIR.rglob('*'):
        if p.suffix.lower() in ('.wav', '.mp3', '.m4a', '.flac'):
            idx.setdefault(p.stem, p)
    return idx


def cut_and_save(task):
    stem, raw = task
    if raw is None:
        return stem, -1.0, False
    try:
        y, sr_n = librosa.load(str(raw), sr=None, mono=True)   # 只解碼，不重採樣
    except Exception:
        return stem, -1.0, False

    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if len(rms) == 0:
        return stem, -1.0, False
    peak_frame = int(np.argmax(rms))
    start = max(0, peak_frame * hop - int(0.05 * sr_n))
    threshold = (float(np.max(rms)) + 1e-8) * 10 ** (-30.0 / 20.0)
    gap_allow = int(GAP_ALLOW_SEC * sr_n / hop)
    end_frame = peak_frame
    since_above = 0
    for i in range(peak_frame, len(rms)):
        if rms[i] > threshold:
            end_frame = i
            since_above = 0
        else:
            since_above += 1
            if since_above > gap_allow:
                break
    end = min(len(y), (end_frame + 1) * hop + int(TAIL_SEC * sr_n))
    end = min(end, start + int(MAX_LEN_SEC * sr_n))
    seg = y[start:end]
    if len(seg) < int(0.4 * sr_n):
        return stem, -1.0, False

    # 只對切段 resample
    if sr_n != Config.SAMPLE_RATE:
        seg = librosa.resample(seg, orig_sr=sr_n, target_sr=Config.SAMPLE_RATE)

    OUT_COUGH_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_COUGH_DIR / f"{stem}.wav"), seg, Config.SAMPLE_RATE)

    old_pt = CACHE_DIR / f"{stem}.pt"
    if old_pt.exists():
        old_pt.unlink()
    ok = process_and_save(str(OUT_COUGH_DIR / f"{stem}.wav"), str(CACHE_DIR))
    return stem, len(seg) / Config.SAMPLE_RATE, ok


def main():
    df = pd.read_csv(METADATA_CSV)
    stems = []
    if 'three_cough_file_name' in df.columns:
        stems = [os.path.splitext(str(v))[0]
                 for v in df['three_cough_file_name'].dropna()]
    stems = sorted(set(stems))

    print("建立 raw 索引（一次性，約 10–30 秒）...")
    index = build_index()
    tasks = [(s, index.get(s)) for s in stems]
    missing = sum(1 for _, p in tasks if p is None)
    print(f"three_cough 目標：{len(stems)} 筆（缺 raw：{missing}）")

    done = 0
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
        futs = [ex.submit(cut_and_save, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Recutting"):
            stem, dur, ok = fut.result()
            if ok:
                done += 1

    print(f"\n路線 B 重切完成：{done}/{len(stems)}")


if __name__ == '__main__':
    main()