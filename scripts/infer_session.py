"""
scripts/infer_session.py
會話層分析腳本：滑窗掃描全音檔，統計 Wheeze / Cough 機率，輸出診斷結論
"""
import argparse
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path
from config import Config

HOP_SEC = 1.0
WHEEZE_THRESH = 0.55
MILD_WHEEZE_THRESH = 0.20
MIN_EPISODE = 3  # 最少連續 3 窗 (3 秒) 算一個喘鳴陣發


def get_mel_in_memory(y: np.ndarray, sr: int) -> np.ndarray:
    """記憶體內純淨特徵提取（與步驟 1-3 完全對齊）"""
    win = Config.TARGET_SAMPLES
    if len(y) < win:
        pad = win - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="reflect")
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

    m_f = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=Config.N_MELS,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
    )
    ref = float(np.max(m_f)) + 1e-8

    chans = []
    for yy in (y, y_h, y_p):
        m = librosa.feature.melspectrogram(
            y=yy,
            sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
        )
        db = librosa.power_to_db(m, ref=ref)
        if db.shape[1] > Config.TARGET_FRAMES:
            db = db[:, : Config.TARGET_FRAMES]
        elif db.shape[1] < Config.TARGET_FRAMES:
            db = np.pad(
                db,
                ((0, 0), (0, Config.TARGET_FRAMES - db.shape[1])),
                mode="constant",
                constant_values=-100.0,
            )
        chans.append(db)

    mel_3ch = np.stack(chans, axis=0)
    return mel_3ch[None, ...]


def main():
    parser = argparse.ArgumentParser(description="會話層音訊分析")
    parser.add_argument("--file", type=str, required=True, help="要分析的音檔路徑")
    parser.add_argument(
        "--model",
        type=str,
        default="asthma_monitor_final.onnx",
        help="ONNX 模型路徑",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="phone",
        choices=["phone", "stetho"],
        help="模式：phone 或 stetho",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[Error] 找不到音檔：{file_path}")
        return

    print(f"載入長錄音：{file_path.name}（mode={args.mode}）")
    y, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)

    session = ort.InferenceSession(args.model)
    hop_samples = int(HOP_SEC * sr)
    win_samples = Config.TARGET_SAMPLES

    results = []
    print("滑窗分析中...")
    for start in range(0, len(y) - win_samples + 1, hop_samples):
        mel = get_mel_in_memory(y[start : start + win_samples], sr)
        w_logit, c_logit = session.run(
            ["wheeze_logit", "cough_logit"], {"mel": mel}
        )
        results.append(
            {
                "time": f"{start / sr:4.1f}s",
                "wheeze": float(1 / (1 + np.exp(-w_logit[0][0]))),
                "cough": float(1 / (1 + np.exp(-c_logit[0][0]))),
            }
        )

    print("\n" + "=" * 50)
    print(f"會話層分析報告：{file_path.name}（{args.mode}）")
    print("=" * 50)

    max_w = max((r["wheeze"] for r in results), default=0.0)

    strong_hits = 0
    mild_hits = 0

    for r in results:
        is_pure_cough = r["cough"] >= 0.15
        eff_w = 0.0 if is_pure_cough else r["wheeze"]

        if eff_w >= 0.55:
            strong_hits += 1
        elif eff_w >= 0.20:
            mild_hits += 1

    print(f"總視窗數：{len(results)}")
    print(f"強喘鳴視窗 (W>=0.55)：{strong_hits}")
    print(f"微弱喘鳴視窗 (W in [0.20, 0.54])：{mild_hits}")
    print(f"Peak Wheeze Prob：{max_w:.4f}")

    consecutive = 0
    max_consecutive = 0
    for r in results:
        if r["wheeze"] >= 0.20:
            consecutive += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0

    print(f"最長連續喘鳴段落：{max_consecutive} 窗")

    if args.mode == "stetho":
        if max_consecutive >= 3 or strong_hits >= 2:
            print("\n結論：【ALARM - 高度懷疑氣道狹窄 / 喘鳴發作】")
            print("偵測到連續性窄頻喘鳴段落。")
        elif max_consecutive >= 2 or strong_hits >= 1 or (strong_hits + mild_hits >= 3):
            print("\n結論：【WARNING - 偵測到疑似短暫氣道狹窄音】")
        else:
            print("\n結論：【NORMAL - 目前無明顯喘鳴跡象】")
    else:
        if max_consecutive >= 3 or strong_hits >= 2:
            print("\n結論：【ALARM - 高度懷疑喘鳴發作】")
            print("連續多窗捕獲顯著/微弱喘鳴特徵。")
        elif max_consecutive >= 2 or strong_hits >= 1 or (strong_hits + mild_hits >= 3):
            print("\n結論：【WARNING - 偵測到微弱/疑似喘鳴訊號】")
            print("建議拉近手機至發聲處重新量測 15 秒，或改用聽診器確診。")
        elif is_isolated_squeak:
            print(
                "\n結論：【NORMAL - 捕捉到單點短促過渡音/碰撞音 (Squeak"
                " <1.0s)】"
            )
            print("屬於單點偶發機械碰撞或短促尾音，未達到氣喘發作門檻。")
        else:
            print("\n結論：【NORMAL - 目前無明顯喘鳴跡象】")

    print("\n[時間軸細節]")
    for r in results:
        w = r["wheeze"]
        c = r["cough"]
        w_tag = "[HIGH]" if w >= 0.55 else ("[MILD]" if w >= 0.20 else "[LOW ]")
        c_tag = "[COUGH]" if c >= 0.50 else "[     ]"
        print(f"{r['time']} | W:{w:.2f} {w_tag} | C:{c:.2f} {c_tag}")


if __name__ == "__main__":
    main()
