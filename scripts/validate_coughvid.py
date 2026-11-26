"""
scripts/validate_coughvid.py (v3 多數決共識版)
以 COUGHVID 多位評分者 (Raters) 的「多數決」作為聲學標準答案：
  A. 敏感度：專家共識喘鳴組的擊中率
  B. 特異性：專家共識無喘鳴組的誤報率
  C. 整體 AUROC（共識正 vs 共識負）
"""
import re
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from config import Config
from step1_extract_features import pad_with_ambient_noise

COUGHVID_DIR = Path('./data/COUGHVID')
META_CSV = COUGHVID_DIR / 'metadata_compiled.csv'
AUDIO_DIRS = [COUGHVID_DIR, COUGHVID_DIR / 'wav', COUGHVID_DIR / 'audio']

WHEEZE_THRESH = 0.7
CONSENSUS_THRESH = 2  # 至少 2 位專家聽見才算「真喘鳴」


def find_audio(uuid):
    for d in AUDIO_DIRS:
        for ext in ('.wav', '.mp3'):
            p = d / (uuid + ext)
            if p.exists():
                return p
    return None


def extract_mel(y, sr, seed_key):
    """與 step1 完全一致的 peak 對齊視窗 + 3 通道 HPSS mel"""
    rms = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0]
    if len(rms) == 0 or np.max(rms) <= 1e-6:
        return None
    peak_sample = int(np.argmax(rms)) * Config.HOP_LENGTH
    start = peak_sample - int(Config.TARGET_SAMPLES * 0.30)
    end = start + Config.TARGET_SAMPLES
    if start < 0:
        end = min(len(y), end - start)
        start = 0
    if end > len(y):
        start = max(0, start - (end - len(y)))
        end = len(y)
    seg = y[start:end]
    if len(seg) < Config.TARGET_SAMPLES:
        seg = pad_with_ambient_noise(seg, Config.TARGET_SAMPLES, sr, seed_key=seed_key)

    y_h, y_p = librosa.effects.hpss(seg, margin=(3.0, 3.0))
    m_f = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=Config.N_MELS,
                                         n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    ref = float(np.max(m_f)) + 1e-8
    chans = []
    for yy in (seg, y_h, y_p):
        m = librosa.feature.melspectrogram(y=yy, sr=sr, n_mels=Config.N_MELS,
                                           n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
        db = librosa.power_to_db(m, ref=ref)
        if db.shape[1] > Config.TARGET_FRAMES:
            db = db[:, :Config.TARGET_FRAMES]
        elif db.shape[1] < Config.TARGET_FRAMES:
            db = np.pad(db, ((0, 0), (0, Config.TARGET_FRAMES - db.shape[1])),
                        mode='constant', constant_values=-100.0)
        chans.append(torch.from_numpy(db).float())
    return torch.stack(chans).unsqueeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='asthma_monitor_final.onnx')
    ap.add_argument('--max-neg', type=int, default=1500, help='負樣本上限(加速)')
    args = ap.parse_args()

    df = pd.read_csv(META_CSV)
    
    # 動態抓取所有 wheezing_X 欄位
    wheeze_cols = [c for c in df.columns if re.match(r'^wheezing_\d+$', str(c))]
    if not wheeze_cols:
        print("[Error] 找不到任何 wheezing_X 專家標注欄位！")
        return
    print(f"找到 {len(wheeze_cols)} 位專家標注欄位: {wheeze_cols}")

    # 安全轉換為數值 (處理 True/False, 1/0, 或 NaN)
    for c in wheeze_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 計算有效票數與喘鳴票數
    df['valid_votes'] = df[wheeze_cols].apply(lambda row: sum(row.notna()), axis=1)
    df['wheeze_votes'] = df[wheeze_cols].sum(axis=1)

    # 過濾掉評分者太少 (小於 2 人) 的錄音
    df_valid = df[df['valid_votes'] >= 2].copy()
    print(f"有效評分錄音 (≥2位專家): {len(df_valid)} 筆")

    # 定義正負組 (多數決)
    exp_pos = df_valid[df_valid['wheeze_votes'] >= CONSENSUS_THRESH]
    exp_neg = df_valid[df_valid['wheeze_votes'] == 0]
    
    # 取樣負樣本以平衡計算時間
    if len(exp_neg) > args.max_neg:
        exp_neg = exp_neg.sample(args.max_neg, random_state=42)
        
    print(f"專家共識喘鳴組 (≥{CONSENSUS_THRESH}票): n={len(exp_pos)}")
    print(f"專家共識無喘鳴組 (0票): n={len(exp_neg)}")

    if len(exp_pos) < 10:
        print("[Warning] 專家共識喘鳴組樣本極少，結果僅供探索參考")

    session = ort.InferenceSession(args.model)
    pos_w, neg_w, pos_c = [], [], []

    def run(group, store_w, store_c=None, desc=""):
        for _, r in tqdm(group.iterrows(), total=len(group), desc=desc):
            wav = find_audio(str(r['uuid']))
            if wav is None:
                continue
            try:
                y, sr = librosa.load(str(wav), sr=Config.SAMPLE_RATE, mono=True)
            except Exception:
                continue
            mel = extract_mel(y, sr, seed_key=str(r['uuid']))
            if mel is None:
                continue
            w_logit, c_logit = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
            store_w.append(float(1 / (1 + np.exp(-w_logit[0][0]))))
            if store_c is not None:
                store_c.append(float(1 / (1 + np.exp(-c_logit[0][0]))))

    run(exp_pos, pos_w, pos_c, desc="專家喘鳴組")
    run(exp_neg, neg_w, desc="專家無喘鳴組")

    pos_w, neg_w = np.array(pos_w), np.array(neg_w)
    
    print("\n" + "=" * 60)
    print("COUGHVID 外部驗證報告 (多數決共識版)")
    print("=" * 60)
    
    if len(pos_w):
        print(f"專家共識喘鳴組 (n={len(pos_w)})：")
        print(f"   擊中率 @0.7 = {(pos_w > WHEEZE_THRESH).mean() * 100:.1f}%，"
              f"@0.5 = {(pos_w > 0.5).mean() * 100:.1f}%，median = {np.median(pos_w):.3f}")
        print(f"   （咳嗽頭 sanity：cough median = {np.median(pos_c):.3f}）")
        
    if len(neg_w):
        print(f"專家共識無喘鳴組 (n={len(neg_w)})：")
        print(f"   誤報率 @0.7 = {(neg_w > WHEEZE_THRESH).mean() * 100:.1f}%，median = {np.median(neg_w):.3f}")
        
    if len(pos_w) and len(neg_w):
        auroc = roc_auc_score(np.concatenate([np.ones(len(pos_w)), np.zeros(len(neg_w))]),
                              np.concatenate([pos_w, neg_w]))
        print(f"外部 AUROC（共識正 vs 共識負）= {auroc:.4f}")

    print("\n── 判讀指南 ──")
    print("1. AUROC ≥ 0.80：模型的耳朵與人類醫師群體高度一致！(極佳)")
    print("2. 擊中率 @0.7 ≥ 50%：在真實世界咳嗽中，能抓出一半以上醫師聽到的喘鳴。")
    print("3. 誤報率 @0.7 < 10%：不會把普通咳嗽誤判為喘鳴。")


if __name__ == '__main__':
    main()