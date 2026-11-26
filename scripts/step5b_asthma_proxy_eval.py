import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized.*")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.mt_cdann import AsthmaCDANN
import torchaudio

def main():
    print("==========================================================")
    print("啟動臨床假說驗證: 喘鳴聲 (Wheeze) 能否作為氣喘 (Asthma) 的有效代理指標？")
    print("==========================================================")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = Path(Config.PROJECT_ROOT) / "checkpoints" / f"best_{Config.MODEL_NAME}_cdann.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重: {model_path}")

    print("載入聲學模型...")
    model = AsthmaCDANN(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 必須與 step0 完全一致的轉換器
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128,
        f_min=0.0, f_max=Config.SAMPLE_RATE / 2.0, power=2.0, normalized=False, center=True, pad_mode="reflect"
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    mel_transform = torch.nn.Sequential(mel_spec, amp_to_db).to(device)

    print("載入臨床 Metadata...")
    df_part = pd.read_csv('./data/UK_COVID19/participant_metadata.csv')
    df_audio = pd.read_csv('./data/UK_COVID19/audio_metadata.csv')
    
    # 篩選出有明確氣喘標記的病患 (假設欄位名稱為 respiratory_condition_asthma)
    if 'respiratory_condition_asthma' not in df_part.columns:
        raise ValueError("participant_metadata.csv 中找不到 respiratory_condition_asthma 欄位")
        
    df_part['asthma_label'] = df_part['respiratory_condition_asthma'].apply(lambda x: 1.0 if str(x).lower() in ['true', '1', 'yes', '1.0'] else 0.0)
    pid_to_asthma = dict(zip(df_part['participant_identifier'], df_part['asthma_label']))

    audio_dir = Path(Config.UK_AUDIO_DIR)
    
    print("建立全域音檔路徑快取 (Global Path Cache)...")
    all_wavs = list(audio_dir.rglob("*.wav"))
    filename_to_path = {wav.name: wav for wav in all_wavs}
    
    if not filename_to_path:
        raise FileNotFoundError(f"[致命錯誤] 找不到任何 .wav 檔案於 {audio_dir}")
    
    y_true_asthma = []
    y_score_max_wheeze = []

    print("執行病患級別 (Patient-Level) 聲學掃描...")
    # 只掃描有在 audio_metadata 中且有臨床標籤的病患
    valid_pids = [pid for pid in df_audio['participant_identifier'].unique() if pid in pid_to_asthma]
    
    with torch.no_grad():
        for pid in tqdm(valid_pids):
            patient_row = df_audio[df_audio['participant_identifier'] == pid].iloc[0]
            audio_files = [
                str(patient_row.get('exhalation_file_name', '')),
                str(patient_row.get('cough_file_name', '')),
                str(patient_row.get('three_cough_file_name', ''))
            ]
            
            patient_wheeze_probs = []
            
            for fname in audio_files:
                if pd.isna(fname) or str(fname).lower() == 'nan' or not fname: 
                    continue
                    
                # 透過 O(1) 查表獲取真實絕對路徑
                wav_path = filename_to_path.get(fname)
                if not wav_path: 
                    continue
                
                import soundfile as sf
                
                waveform_np, sr = sf.read(str(wav_path), dtype='float32')
                
                if waveform_np.ndim > 1:
                    waveform_np = np.mean(waveform_np, axis=1)
                    
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                
                if sr != Config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)
                    waveform = resampler(waveform)
                
                current_samples = waveform.shape[1]
                target_samples = Config.TARGET_SAMPLES

                if current_samples < target_samples:
                    # 計算需要重複幾次才能覆蓋目標長度
                    repeats = (target_samples // current_samples) + 1
                    waveform = waveform.repeat(1, repeats)[:, :target_samples]
                else:
                    waveform = waveform[:, :target_samples]

                waveform = waveform.to(device)

                # mel_transform 輸出形狀為 [1, 128, T] (Channel, Mels, Time)
                mel = mel_transform(waveform)

                mel = mel.unsqueeze(0)

                # 執行推論
                logits, _ = model(mel, alpha=0.0)
                wheeze_prob = torch.sigmoid(logits[0, 0]).item()
                patient_wheeze_probs.append(wheeze_prob)

            if patient_wheeze_probs:
                # 病患級聚合：取該病患所有音檔中，偵測到喘鳴的最大機率
                max_wheeze = max(patient_wheeze_probs)
                y_score_max_wheeze.append(max_wheeze)
                y_true_asthma.append(pid_to_asthma[pid])

    if len(y_true_asthma) == 0:
        raise ValueError("\n\n[致命錯誤] 成功推論的病患數為 0！\n請檢查: \n1. audio_metadata.csv 的檔名是否有 .wav 副檔名\n2. 該副檔名是否與實際硬碟上的檔案相符。")

    # 計算臨床代理指標
    fpr, tpr, thresholds = roc_curve(y_true_asthma, y_score_max_wheeze)
    roc_auc = auc(fpr, tpr)
    
    print("\n==========================================================")
    print("假說驗證報告: 喘鳴 (Acoustic) -> 氣喘 (Clinical)")
    print("==========================================================")
    print(f"- 參與測試病患總數: {len(y_true_asthma)}")
    print(f"- 真實氣喘病患數: {sum(y_true_asthma)}")
    print(f"- 代理 AUROC: {roc_auc:.4f}")
    
    if roc_auc < 0.70:
        print("\n[Warning] 結論：AUROC 表現不佳。這強烈證實了「非發作期氣喘」與「寂靜胸」的存在。")
        print("[Warning] 僅靠物理聲音無法涵蓋氣喘的複雜臨床樣貌，必須啟動 [選項 B：多模態融合系統]。")
    else:
        print("\n[OK] 結論：喘鳴特徵具備高度代表性。模型表現強勢，可直接考慮封裝上線。")

if __name__ == "__main__":
    main()