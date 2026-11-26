# scripts/build_proxy_val.py
"""用 ICBHI val 檔（訓練沒見過）產生手機域代理驗證資料（儀表板版）。"""
import hashlib
import traceback
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from synthesize_phone_wheeze_v3 import (
    PhoneMicSimulatorV4, physical_mix, loop_fill_wheeze, UK_EXHALE_DIR
)
from step1_extract_features import process_and_save
from config import Config

ICBHI_DIR = Path('./data/ICBHI_final_database')
OUT = Path('./data/UK_COVID19/proxy_val_wheeze')
PROXY_CACHE = './data/UK_COVID19/proxy_val_cache'
SR = Config.SAMPLE_RATE


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim = PhoneMicSimulatorV4(SR)
    exhale_files = list(UK_EXHALE_DIR.glob('*.wav'))
    if not exhale_files:
        print("[Error] 找不到 UK 哈氣背景！")
        return

    wav_written = 0
    pt_saved = 0
    fail_reasons = {}

    for wav_path in tqdm(sorted(ICBHI_DIR.glob('*.wav'))):
        # 只用 val 檔
        if int(hashlib.md5(wav_path.name.encode()).hexdigest(), 16) % 100 < 80:
            continue
        txt = wav_path.with_suffix('.txt')
        if not txt.exists():
            continue
        try:
            y_full, sr = librosa.load(str(wav_path), sr=SR, mono=True)
        except Exception:
            continue

        for line in txt.read_text().strip().split('\n'):
            parts = line.split()
            if len(parts) < 4 or int(parts[3]) != 1:
                continue
            seg = y_full[int(float(parts[0]) * sr):int(float(parts[1]) * sr)]
            if len(seg) < int(0.3 * sr):
                continue

            try:
                w_base = sim.process_base(seg)

                bg, _ = librosa.load(
                    str(exhale_files[np.random.randint(len(exhale_files))]),
                    sr=SR, mono=True
                )
                if len(bg) < len(w_base) + int(0.5 * sr):
                    continue

                rms = librosa.feature.rms(y=bg, frame_length=512, hop_length=128)[0]
                af = np.where(rms > np.max(rms) * 0.25)[0]
                if len(af) < 5:
                    continue
                a0, a1 = int(af[0] * 128), int(af[-1] * 128)
                w_len = min(len(w_base), a1 - a0)
                if w_len < int(0.1 * sr):
                    continue

                w_filled = loop_fill_wheeze(w_base, w_len, SR)

                fade_len = int(0.03 * SR)
                if w_len > fade_len * 3:
                    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    w_filled = w_filled.copy()
                    w_filled[:fade_len] *= fade_in
                    w_filled[-fade_len:] *= fade_out

                ins = np.random.randint(a0, max(a0 + 1, a1 - w_len))

                bg_local = bg[ins:ins + w_len]
                if len(bg_local) >= 2048 and len(w_filled) >= 2048:
                    w_final = sim.airflow_envelope_transfer(w_filled, bg_local)
                else:
                    w_final = w_filled

                track = np.zeros(len(bg), dtype=np.float32)
                track[ins:ins + w_len] = w_final[:w_len]

                mixed = physical_mix(
                    bg, track, SR,
                    snr_db=-12.0,  # 喘鳴能量只有背景的 25%
                    wheeze_region=(ins, ins + w_len)
                )

                out_path = OUT / f"synth_wheeze_exhale_proxy_{wav_written:03d}.wav"
                sf.write(str(out_path), mixed, SR)
                wav_written += 1

                ok = process_and_save(str(out_path), PROXY_CACHE)
                if ok:
                    pt_saved += 1
                else:
                    fail_reasons["process_and_save_False"] = \
                        fail_reasons.get("process_and_save_False", 0) + 1

            except Exception as e:
                key = f"Exception_{type(e).__name__}"
                fail_reasons[key] = fail_reasons.get(key, 0) + 1
                if sum(fail_reasons.values()) <= 3:
                    traceback.print_exc()

    print(f"\nwav 寫入：{wav_written} 筆 -> {OUT}")
    print(f".pt 快取：{pt_saved} 筆 -> {PROXY_CACHE}")
    if fail_reasons:
        print(f"[Warning] 失敗統計：{fail_reasons}")


if __name__ == '__main__':
    main()