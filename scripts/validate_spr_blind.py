"""
scripts/validate_spr_blind.py
SPRSound 歷年盲測集 (BioCAS 2023-2025) 終極驗證
這是證明模型泛化能力的最高級別證據（時間外驗證）。
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

BASE_DIR = Path('./data/SPRSound-main')
YEARS = [2023, 2024, 2025]
MODEL_PATH = 'asthma_monitor_final.onnx'

# 與訓練時完全一致的標籤定義
WHEEZE_LABELS = {'wheeze', 'wheeze+crackle', 'stridor'}
NEG_LABELS = {'normal', 'fine crackle', 'coarse crackle'}

def extract_mel(y, sr):
    """與 cache_sprsound.py 完全一致的特徵提取 (RMS Peak + Reflect Pad)"""
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
            if e > len(y): e, s = len(y), max(0, len(y) - win)
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
        if db.shape[1] > Config.TARGET_FRAMES: db = db[:, :Config.TARGET_FRAMES]
        elif db.shape[1] < Config.TARGET_FRAMES:
            db = np.pad(db, ((0, 0), (0, Config.TARGET_FRAMES - db.shape[1])),
                        mode='constant', constant_values=-100.0)
        chans.append(torch.from_numpy(db).float())
    return torch.stack(chans).unsqueeze(0).numpy()

def process_year(year, session):
    json_dir = BASE_DIR / f'BioCAS{year}' / f'test{year}_json'
    wav_dir = BASE_DIR / f'BioCAS{year}' / f'test{year}_wav'
    
    if not json_dir.exists() or not wav_dir.exists():
        print(f"[Warning] {year} 資料夾不存在，跳過")
        return [], []
        
    json_files = sorted(json_dir.glob('*.json'))
    probs, trues = [], []
    stats = {'w': 0, 'nw': 0, 'skip': 0}
    
    for jf in tqdm(json_files, desc=f"BioCAS {year}", leave=False):
        try:
            meta = json.load(open(jf))
        except: continue
        
        hits = list(wav_dir.glob(f"{jf.stem}.*"))
        if not hits: continue
        try:
            y, sr0 = librosa.load(str(hits[0]), sr=None, mono=True)
        except: continue
        if sr0 != Config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr0, target_sr=Config.SAMPLE_RATE)
            
        for ann in meta.get('event_annotation', []):
            if not isinstance(ann, dict): continue
            label = str(ann.get('type', ann.get('label', ''))).lower().strip()
            
            if label in WHEEZE_LABELS: w = 1.0; stats['w'] += 1
            elif label in NEG_LABELS: w = 0.0; stats['nw'] += 1
            else: stats['skip'] += 1; continue  # 跳過 rhonchi 或未知標籤
            
            s = int(float(ann.get('start', 0)) / 1000.0 * Config.SAMPLE_RATE)
            e = int(float(ann.get('end', 0)) / 1000.0 * Config.SAMPLE_RATE)
            seg = y[s:e]
            if len(seg) < int(0.1 * Config.SAMPLE_RATE): continue
            
            mel = extract_mel(seg, Config.SAMPLE_RATE)
            w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
            prob = float(1 / (1 + np.exp(-w_logit[0][0])))
            probs.append(prob)
            trues.append(w)
            
    print(f"\nBioCAS {year} 統計：Wheeze={stats['w']}, Non-Wheeze={stats['nw']}, 跳過={stats['skip']}")
    return probs, trues

def report(name, probs, trues):
    probs, trues = np.array(probs), np.array(trues)
    if len(probs) == 0:
        print(f"{name}：無有效樣本")
        return
    if len(np.unique(trues)) < 2:
        print(f"{name}：類別不足 (可能該年份標籤格式不同或全為單一類)")
        print(f"  預測機率分布: mean={probs.mean():.3f}, max={probs.max():.3f}, min={probs.min():.3f}")
        return
        
    auroc = roc_auc_score(trues, probs)
    fpr, tpr, th = roc_curve(trues, probs)
    i = min(int(np.argmax(tpr - fpr)), len(th) - 1)
    pred = (probs >= th[i]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(trues, pred, average='binary', zero_division=0)
    
    print(f"  AUROC   = {auroc:.4f}")
    print(f"  Prec    = {p:.4f} | Recall  = {r:.4f} | F1 = {f1:.4f} @thr {th[i]:.3f}")
    print(f"  機率分布: Wheeze mean={probs[trues==1].mean():.3f}, Neg mean={probs[trues==0].mean():.3f}")

def main():
    print("SPRSound 歷年盲測集 (BioCAS 2023-2025) 終極驗證啟動...")
    session = ort.InferenceSession(MODEL_PATH)
    
    all_probs, all_trues = [], []
    
    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"正在處理 BioCAS {year} 測試集...")
        print(f"{'='*60}")
        p, t = process_year(year, session)
        if p:
            report(f"BioCAS {year}", p, t)
            all_probs.extend(p)
            all_trues.extend(t)
            
    print(f"\n{'='*60}")
    print("總體盲測成績 (BioCAS 2023 + 2024 + 2025)")
    print(f"{'='*60}")
    report("All Years (Total)", all_probs, all_trues)

if __name__ == '__main__':
    main()