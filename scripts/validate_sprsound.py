"""
scripts/validate_sprsound.py (v5 週期級 Max-Pooling 修正版)
修正：1. 反射填充消除邊緣效應 2. 滑窗 Max-Pooling 對齊 Cycle-level 標籤
"""
import json
import numpy as np
import librosa
import onnxruntime as ort
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from config import Config

JSON_DIR = Path('./data/SPRSound-main/Classification/train_classification_json')
WAV_DIR = Path('./data/SPRSound-main/Classification/train_classification_wav')
MODEL_PATH = 'asthma_monitor_final.onnx'

WHEEZE_LABELS = {'wheeze', 'wheeze+crackle', 'stridor'}
NON_WHEEZE_LABELS = {'normal', 'fine crackle', 'coarse crackle', 'rhonchi'} # 把 rhonchi 加回來，看模型真實泛化


def extract_mel_fixed(y, sr):
    """假設 y 長度已嚴格等於 TARGET_SAMPLES"""
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
    return torch.stack(chans).unsqueeze(0).numpy()


def predict_cycle_max_pool(y_seg, sr, session):
    """對一個呼吸週期進行滑窗，取最大機率 (Max Pooling)"""
    win = Config.TARGET_SAMPLES
    hop = win // 2  # 50% overlap (1秒)
    
    if len(y_seg) < win:
        # 短音訊：使用反射填充 (Reflect)，完美消除 zero-padding 的邊緣效應
        pad_len = win - len(y_seg)
        y_padded = np.pad(y_seg, (pad_len // 2, pad_len - pad_len // 2), mode='reflect')
        mel = extract_mel_fixed(y_padded, sr)
        w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
        return float(1 / (1 + np.exp(-w_logit[0][0])))
        
    # 長音訊：滑動窗口
    probs = []
    for start in range(0, len(y_seg) - win + 1, hop):
        chunk = y_seg[start:start+win]
        mel = extract_mel_fixed(chunk, sr)
        w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
        prob = float(1 / (1 + np.exp(-w_logit[0][0])))
        probs.append(prob)
        
    if not probs: # 防禦性編程
        chunk = y_seg[-win:]
        mel = extract_mel_fixed(chunk, sr)
        w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
        probs.append(float(1 / (1 + np.exp(-w_logit[0][0]))))
        
    return max(probs) # 只要週期內有一處喘鳴，就算 Wheeze


def parse_event(ann):
    if isinstance(ann, dict):
        start = float(ann.get('start', 0)) / 1000.0
        end = float(ann.get('end', 0)) / 1000.0
        label = str(ann.get('type', ann.get('label', ''))).lower().strip()
        return start, end, label
    return None


def report(name, recs):
    if not recs: return
    probs = np.array([r[0] for r in recs])
    trues = np.array([r[1] for r in recs])
    if len(np.unique(trues)) < 2: return
    
    auroc = roc_auc_score(trues, probs)
    fpr, tpr, th = roc_curve(trues, probs)
    i = min(int(np.argmax(tpr - fpr)), len(th) - 1)
    pred = (probs >= th[i]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(trues, pred, average='binary', zero_division=0)
    
    print(f"\n{name}（n={len(recs)}，wheeze={int(trues.sum())}）")
    print(f"  AUROC   = {auroc:.4f}")
    print(f"  Prec    = {p:.4f}")
    print(f"  Recall  = {r:.4f}")
    print(f"  F1-Score= {f1:.4f} @最佳閾值 {th[i]:.3f}")
    print(f"  機率分布: Wheeze mean={probs[trues==1].mean():.3f}, Non-Wheeze mean={probs[trues==0].mean():.3f}")


def main():
    print("SPRSound 最終期末考 v5 (Max-Pooling 修正) 啟動...")
    session = ort.InferenceSession(MODEL_PATH)
    json_files = sorted(JSON_DIR.glob('*.json'))

    records = []
    stats = {'w': 0, 'nw': 0}

    for jf in tqdm(json_files, desc="處理 SPRSound"):
        try:
            meta = json.load(open(jf, 'r'))
        except Exception: continue
        events = meta.get('event_annotation', [])

        wav_hits = list(WAV_DIR.glob(f"{jf.stem}.*"))
        if not wav_hits: continue
        try:
            y_full, sr0 = librosa.load(str(wav_hits[0]), sr=None, mono=True)
        except Exception: continue
        if sr0 != Config.SAMPLE_RATE:
            y_full = librosa.resample(y_full, orig_sr=sr0, target_sr=Config.SAMPLE_RATE)

        for ann in events:
            parsed = parse_event(ann)
            if parsed is None: continue
            s, e, label = parsed
            if e <= s: continue
                
            seg = y_full[int(s * Config.SAMPLE_RATE):int(e * Config.SAMPLE_RATE)]
            if len(seg) < int(0.1 * Config.SAMPLE_RATE): continue
                
            if label in WHEEZE_LABELS:
                y = 1; stats['w'] += 1
            elif label in NON_WHEEZE_LABELS:
                y = 0; stats['nw'] += 1
            else: continue
                
            # 使用 Max-Pooling 預測整個週期
            prob = predict_cycle_max_pool(seg, Config.SAMPLE_RATE, session)
            records.append((prob, y))

    print(f"\n資料統計：Wheeze={stats['w']}, Non-Wheeze={stats['nw']}")
    print("=" * 60)
    report("SPRSound 跨域期末考 (Cycle-level Max-Pooling)", records)
    print("=" * 60)


if __name__ == '__main__':
    main()