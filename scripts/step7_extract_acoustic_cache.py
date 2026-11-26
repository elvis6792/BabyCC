import torch
import torchaudio
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="An output with one or more elements was resized.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.mt_cdann import AsthmaCDANN

def extract_patient_acoustic_feature(pid, patient_row, filename_to_path, model, mel_transform, device):
    audio_files = [
        str(patient_row.get('exhalation_file_name', '')),
        str(patient_row.get('cough_file_name', '')),
        str(patient_row.get('three_cough_file_name', ''))
    ]
    
    patient_logits = []
    
    for fname in audio_files:
        if pd.isna(fname) or str(fname).lower() == 'nan' or not fname: 
            continue
            
        wav_path = filename_to_path.get(fname)
        if not wav_path: 
            continue
        
        import soundfile as sf
        try:
            waveform_np, sr = sf.read(str(wav_path), dtype='float32')
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)
                
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
            if sr != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)
                waveform = resampler(waveform)
                
            current_samples = waveform.shape[1]
            target_samples = Config.TARGET_SAMPLES
            
            if current_samples < target_samples:
                repeats = (target_samples // current_samples) + 1
                waveform = waveform.repeat(1, repeats)[:, :target_samples]
            else:
                waveform = waveform[:, :target_samples]
                
            waveform = waveform.to(device)
            mel = mel_transform(waveform).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = model(mel, alpha=0.0)
                patient_logits.append(logits.squeeze(0).cpu())
        except Exception:
            continue
            
    if not patient_logits:
        return torch.zeros(3, dtype=torch.float32)
        
    stacked_logits = torch.stack(patient_logits, dim=0)
    pooled_feature, _ = torch.max(stacked_logits, dim=0)
    
    return pooled_feature

def main():
    print("==========================================================")
    print("啟動離線特徵聚合快取管線 (Offline Acoustic Feature Caching)")
    print("==========================================================")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = Path(Config.PROJECT_ROOT) / "checkpoints" / f"best_{Config.MODEL_NAME}_cdann.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到聲學模型權重: {model_path}")

    model = AsthmaCDANN(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128,
        f_min=0.0, f_max=Config.SAMPLE_RATE / 2.0, power=2.0, normalized=False, center=True, pad_mode="reflect"
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    mel_transform = torch.nn.Sequential(mel_spec, amp_to_db).to(device)

    data_dir = Path(Config.PROJECT_ROOT) / "processed_fusion_dataset"
    train_dict_path = data_dir / "tabular_train.pt"
    val_dict_path = data_dir / "tabular_val.pt"
    
    if not train_dict_path.exists() or not val_dict_path.exists():
        raise FileNotFoundError("找不到 tabular_train.pt 或 tabular_val.pt，請先執行 step6。")
        
    train_dict = torch.load(train_dict_path, weights_only=True)
    val_dict = torch.load(val_dict_path, weights_only=True)

    df_audio = pd.read_csv('./data/UK_COVID19/audio_metadata.csv')
    audio_dir = Path(Config.UK_AUDIO_DIR)
    all_wavs = list(audio_dir.rglob("*.wav"))
    filename_to_path = {wav.name: wav for wav in all_wavs}

    def process_split(split_dict, split_name):
        print(f"\n處理 {split_name} Split (共 {len(split_dict)} 名病患)...")
        fused_dict = {}
        for pid, data in tqdm(split_dict.items()):
            patient_row_matches = df_audio[df_audio['participant_identifier'] == pid]
            if patient_row_matches.empty:
                continue
                
            patient_row = patient_row_matches.iloc[0]
            acoustic_feat = extract_patient_acoustic_feature(
                pid, patient_row, filename_to_path, model, mel_transform, device
            )
            
            fused_dict[pid] = {
                'tabular': data['tabular'],
                'acoustic': acoustic_feat,
                'label': data['label']
            }
        return fused_dict

    fused_train = process_split(train_dict, "Train")
    fused_val = process_split(val_dict, "Validation")

    torch.save(fused_train, data_dir / "fused_train.pt")
    torch.save(fused_val, data_dir / "fused_val.pt")
    
    print("\n多模態特徵快取完成！")
    print(f"- Fused Train 數量: {len(fused_train)}")
    print(f"- Fused Val 數量: {len(fused_val)}")
    print(f"檔案已綁定並儲存至: {data_dir}")

if __name__ == "__main__":
    main()