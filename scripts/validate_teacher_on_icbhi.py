"""
validate_teacher_icbhi.py
驗證 DSP 教師 (get_teacher_asthma_scores) 在 ICBHI 聽診器資料集上的喘鳴偵測能力。
這是決定「路徑 A+B 知識蒸餾」是否可行的最終判決書。
"""
import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

from step1_extract_features import get_teacher_asthma_scores
from config import Config

# ── 路徑設定 ─────────────────────────────────────────────────
ICBHI_DIR = Path('./data/ICBHI_final_database')
TARGET_SR = Config.SAMPLE_RATE

def parse_icbhi_annotation(txt_path: Path):
    """
    解析 ICBHI 的 .txt 標註檔。
    格式：Start_time End_time Crackle Wheeze
    回傳：list of (start_sec, end_sec, has_crackle, has_wheeze)
    """
    annotations = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                    annotations.append((start, end, crackle, wheeze))
                except ValueError:
                    continue
    return annotations

def main():
    print("="*60)
    print("ICBHI DSP 教師驗證腳本啟動")
    print("="*60)
    
    wav_files = sorted(list(ICBHI_DIR.glob('*.wav')))
    if not wav_files:
        print(f"[Error] 找不到任何 .wav 檔案，請確認路徑：{ICBHI_DIR}")
        return
        
    print(f"找到 {len(wav_files)} 個音檔，開始逐一驗證...")
    
    results = []
    
    for wav_path in tqdm(wav_files, desc="Processing ICBHI"):
        txt_path = wav_path.with_suffix('.txt')
        if not txt_path.exists():
            continue
            
        try:
            # 1. 讀取音檔並強制重採樣到 Config.SAMPLE_RATE (教師預期輸入)
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"\n[Warning] 讀取失敗 {wav_path.name}: {e}")
            continue
            
        # 2. 解析標註
        annotations = parse_icbhi_annotation(txt_path)
        
        # 3. 針對每個呼吸週期切割並送入教師
        for start_sec, end_sec, has_crackle, has_wheeze in annotations:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            
            # 邊界保護
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            segment = y[start_sample:end_sample]
            
            # 忽略過短的片段 (小於 0.1 秒，DSP 無法計算頻譜)
            if len(segment) < int(sr * 0.1):
                continue
                
            # 4. 呼叫 DSP 教師
            # get_teacher_asthma_scores 回傳 [wheezing_score, duration_score]
            scores = get_teacher_asthma_scores(segment, sr=TARGET_SR)
            wheeze_score = float(scores[0])
            
            results.append({
                "file": wav_path.name,
                "start": start_sec,
                "end": end_sec,
                "true_wheeze": has_wheeze,
                "true_crackle": has_crackle,
                "teacher_wheeze_score": wheeze_score
            })
            
    if not results:
        print("[Error] 沒有收集到任何有效的呼吸週期片段，請檢查資料集結構。")
        return

    # ── 5. 計算評估指標 ──────────────────────────────────────
    df = pd.DataFrame(results)
    y_true = df["true_wheeze"].values
    y_score = df["teacher_wheeze_score"].values
    
    print("\n" + "="*60)
    print("驗證結果報告 (DSP Teacher vs ICBHI Ground Truth)")
    print("="*60)
    print(f"總呼吸週期樣本數 : {len(df)}")
    print(f"其中包含喘鳴 (Wheeze) : {int(y_true.sum())} ({y_true.mean()*100:.1f}%)")
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
        print(f"\nAUROC (Area Under ROC) : {auroc:.4f}")
    except ValueError:
        print("\n[Warning] 無法計算 AUROC (可能某一類樣本數為 0)")
        auroc = 0.0
        
    # 尋找最佳 Threshold (基於 F1-score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # 簡單用 Youden's J statistic 找最佳門檻
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    
    y_pred = (y_score >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    print(f"最佳二值化門檻 (Auto-tuned) : {best_thresh:.4f}")
    print(f"   Precision (精確率) : {prec:.4f}")
    print(f"   Recall (召回率)    : {rec:.4f}")
    print(f"   F1-Score           : {f1:.4f}")
    
    print("\n教師給分分佈統計 (Wheeze Score):")
    print(f"   真實為喘鳴 (Wheeze=1) 的平均分 : {y_score[y_true==1].mean():.4f}")
    print(f"   真實為正常 (Wheeze=0) 的平均分 : {y_score[y_true==0].mean():.4f}")
    
    print("\n" + "="*60)
    
    # ── 6. 最終判決建議 ──────────────────────────────────────
    if auroc >= 0.75:
        print("[PASS] 判決：教師物理邏輯極佳！完全可用於路徑 A+B 的知識蒸餾。")
    elif auroc >= 0.60:
        print("[WARN] 判決：教師具備基本鑑別力，但存在雜訊。蒸餾時需調低 lambda_teacher 權重。")
    else:
        print("[FAIL] 判決：教師在聽診器域上失明。請先暫停蒸餾計畫，重新檢討 DSP 頻段設定。")
        
    # 匯出詳細結果供後續除錯
    df.to_csv('icbhi_teacher_validation_report.csv', index=False)
    print(f"\n詳細逐筆預測結果已匯出至: icbhi_teacher_validation_report.csv")

if __name__ == "__main__":
    main()