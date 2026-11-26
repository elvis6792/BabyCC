"""
scripts/diagnose_sprsound.py
法醫級診斷：拆解 SPRSound 7 大標籤的預測機率，找出模型的「認知盲區」。
"""
import json
import numpy as np
import librosa
import onnxruntime as ort
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from config import Config

JSON_DIR = Path('./data/SPRSound-main/Classification/train_classification_json')
WAV_DIR = Path('./data/SPRSound-main/Classification/train_classification_wav')
MODEL_PATH = 'asthma_monitor_final.onnx'

def extract_mel_peak(y, sr):
    """單窗 Peak Alignment (避免 Max-Pooling 放大噪聲)"""
    win = Config.TARGET_SAMPLES
    if len(y) < win:
        pad_len = win - len(y)
        y = np.pad(y, (pad_len // 2, pad_len - pad_len // 2), mode='reflect')
    elif len(y) > win:
        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        if len(rms) > 0:
            peak_sample = int(np.argmax(rms)) * hop
            start = max(0, peak_sample - win // 2)
            end = start + win
            if end > len(y):
                end = len(y)
                start = max(0, end - win)
            y = y[start:end]
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
    return torch.stack(chans).unsqueeze(0).numpy()

def main():
    print("SPRSound 標籤診斷啟動...")
    session = ort.InferenceSession(MODEL_PATH)
    json_files = sorted(JSON_DIR.glob('*.json'))

    # 收集每個標籤的預測機率
    label_probs = defaultdict(list)

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
            if not isinstance(ann, dict): continue
            s = float(ann.get('start', 0)) / 1000.0
            e = float(ann.get('end', 0)) / 1000.0
            label = str(ann.get('type', '')).lower().strip()
            if e <= s: continue
                
            seg = y_full[int(s * Config.SAMPLE_RATE):int(e * Config.SAMPLE_RATE)]
            if len(seg) < int(0.1 * Config.SAMPLE_RATE): continue
                
            mel = extract_mel_peak(seg, Config.SAMPLE_RATE)
            w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
            prob = float(1 / (1 + np.exp(-w_logit[0][0])))
            label_probs[label].append(prob)

    print("\n" + "="*70)
    print("SPRSound 標籤級別認知診斷報告 (單窗 Peak-Align)")
    print("="*70)
    print(f"{'標籤 (Label)':<20} | {'數量':>5} | {'平均機率':>8} | {'>0.5 比例':>10} | 模型認知")
    print("-" * 70)
    
    for label in sorted(label_probs.keys()):
        probs = np.array(label_probs[label])
        mean_p = np.mean(probs)
        ratio = np.mean(probs > 0.5)
        
        # 判斷模型的認知
        if label in ['wheeze', 'wheeze+crackle', 'stridor']:
            expected = "高 (應該 >0.5)"
            status = "[OK] 正確" if mean_p > 0.4 else "[MISS] 漏接"
        elif label == 'normal':
            expected = "低 (應該 <0.2)"
            status = "[OK] 正確" if mean_p < 0.3 else "[FP] 幻覺"
        elif label in ['fine crackle', 'coarse crackle']:
            expected = "低 (爆裂音非喘鳴)"
            status = "[OK] 正確" if mean_p < 0.3 else "[WARN] 混淆"
        elif label == 'rhonchi':
            expected = "ICBHI視為1, SPR視為0"
            status = "[CONFLICT] 標籤衝突區"
        else:
            status = "[UNKNOWN] 未知"
            
        print(f"{label:<20} | {len(probs):>5} | {mean_p:>8.3f} | {ratio*100:>9.1f}% | {status}")
        
    print("="*70)
    print("\n結論指南：")
    print("1. 如果 'rhonchi' 和 'coarse crackle' 的平均機率 > 0.4：")
    print("   證明模型非常聰明！它抓到了所有『連續性異常音』，只是 ICBHI 和 SPRSound 的醫學定義不同。")
    print("2. 如果 'normal' 的平均機率 > 0.3：")
    print("   證明模型有 OOD 幻覺，把聽診器底噪或摩擦音當成了喘鳴。")

if __name__ == '__main__':
    main()