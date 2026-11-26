import os
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from scripts.data_pipeline.wheeze_synthesizer import WheezeInjector
from scripts.data_pipeline.metadata_registry import MetadataRegistry

# 每個 Worker 獨立的注入器實例
_imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_worker_injector = None
_worker_registry = None
_worker_mel_transform = None

def _init_worker():
    global _worker_injector, _worker_registry, _worker_mel_transform
    import multiprocessing
    import torch
    import torchaudio
    from scripts.data_pipeline.wheeze_synthesizer import WheezeInjector
    from scripts.data_pipeline.metadata_registry import MetadataRegistry
    from scripts.config import Config

    worker_seed = (42 + multiprocessing.current_process().pid) % (2**32 - 1)
    
    wheeze_dir = Path(Config.PROJECT_ROOT) / "data" / "extracted_wheezes"
    _worker_injector = WheezeInjector(wheeze_dir=wheeze_dir, sample_rate=Config.SAMPLE_RATE, seed=worker_seed)
    
    uk_csv_path = Path('./data/UK_COVID19/participant_metadata.csv')
    uk_audio_csv = Path('./data/UK_COVID19/audio_metadata.csv')
    icbhi_dir = Path('./data/ICBHI_final_database')

    if not uk_csv_path.exists() or not icbhi_dir.exists() or not uk_audio_csv.exists():
        raise FileNotFoundError(f"找不到 Metadata 來源！請確認外接硬碟已掛載且包含 audio_metadata.csv")

    _worker_registry = MetadataRegistry(
        uk_covid_csv_path=str(uk_csv_path),
        icbhi_dir=str(icbhi_dir),
        uk_audio_csv_path=str(uk_audio_csv)
    )

    if not _worker_registry.pid_registry and not _worker_registry.filename_to_pid:
        raise ValueError("MetadataRegistry 初始化失敗：讀取到的參與者表與橋接表數量均為 0。")

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=0.0,
        f_max=Config.SAMPLE_RATE / 2.0,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode="reflect"
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    
    _worker_mel_transform = torch.nn.Sequential(mel_spec, amp_to_db)

def _process_single_audio(wav_path: Path, output_dir: Path):
    global _worker_injector, _worker_registry, _worker_mel_transform
    
    import torch
    import torchaudio
    import numpy as np
    import traceback
    from scripts.config import Config
    
    base_labels = _worker_registry.get_labels(wav_path.name)
    
    if base_labels[2] == -1.0:
        return False
    
    target_clean = base_labels.copy()
    if target_clean[0] == -1.0:
        target_clean[0] = 0.0
        
    target_wheeze = base_labels.copy()
    target_wheeze[0] = 1.0

    tensor_clean_y = torch.tensor(target_clean, dtype=torch.float32)
    tensor_wheeze_y = torch.tensor(target_wheeze, dtype=torch.float32)

    try:
        import soundfile as sf
        
        waveform_np, sr = sf.read(str(wav_path), dtype='float32')
        
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)
            
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        waveform_np = waveform.squeeze().numpy()
        
        target_samples = Config.TARGET_SAMPLES
        current_samples = len(waveform_np)
        
        if current_samples < target_samples:
            pad_len = target_samples - current_samples
            waveform_tiled = np.pad(waveform_np, (0, pad_len), mode='wrap')

            offset = np.random.randint(0, target_samples)
            waveform_np = np.roll(waveform_tiled, offset)

        elif current_samples > target_samples:
            max_start = current_samples - target_samples
            start_idx = np.random.randint(0, max_start + 1)
            waveform_np = waveform_np[start_idx : start_idx + target_samples]

        # 加入微量 Dither (1e-5 級別白噪音)，徹底破壞數位絕對零點
        noise_floor = np.random.normal(0, 1e-5, size=waveform_np.shape)
        waveform_np = waveform_np + noise_floor

        waveform_clean = torch.from_numpy(waveform_np).to(torch.float32).unsqueeze(0)

        mel_clean = _worker_mel_transform(waveform_clean)

        pt_clean_path = output_dir / f"{wav_path.stem}_clean.pt"
        torch.save((mel_clean, tensor_clean_y), pt_clean_path)
        sf.write(output_dir / f"{wav_path.stem}_clean.wav", waveform_np, Config.SAMPLE_RATE)

        if _worker_injector is not None and tensor_wheeze_y[2] == 1.0: 
            snr_db = np.random.uniform(5.0, 15.0)
            injected_np = _worker_injector.inject(waveform_np, snr_db=snr_db)

            if injected_np is not None:
                waveform_wheeze = torch.from_numpy(injected_np).to(torch.float32).unsqueeze(0)
                mel_wheeze = _worker_mel_transform(waveform_wheeze)

                pt_wheeze_path = output_dir / f"{wav_path.stem}_wheeze_v1.pt"
                torch.save((mel_wheeze, tensor_wheeze_y), pt_wheeze_path)
                sf.write(output_dir / f"{wav_path.stem}_wheeze_v1.wav", injected_np, Config.SAMPLE_RATE)
        
        return True
                
    except Exception as e:
        print(f"\n[Error] 處理失敗 {wav_path.name}:\n{traceback.format_exc()}")
        return False


def main():
    print("==========================================================")
    print("啟動離線特徵生成管線 (Offline PT Generation)")
    print("==========================================================")

    raw_audio_dir = Path(Config.UK_AUDIO_DIR)
    pt_output_dir = Path(Config.PROJECT_ROOT) / "processed_pt_dataset"
    pt_output_dir.mkdir(parents=True, exist_ok=True)

    all_wavs = list(raw_audio_dir.rglob("*.wav"))
    if not all_wavs:
        print(f"Error: 找不到任何 .wav 檔案")
        return

    print("執行 Metadata 對齊預檢 (Pre-flight Check)...")
    from scripts.data_pipeline.metadata_registry import MetadataRegistry
    test_registry = MetadataRegistry(
        uk_covid_csv_path='./data/UK_COVID19/participant_metadata.csv',
        icbhi_dir='./data/ICBHI_final_database',
        uk_audio_csv_path='./data/UK_COVID19/audio_metadata.csv'
    )
    
    unmatched_count = 0
    sample_unmatched = []
    check_limit = min(50, len(all_wavs))
    
    for wav in all_wavs[:check_limit]:
        if test_registry.get_labels(wav.name)[2] == -1.0:
            unmatched_count += 1
            sample_unmatched.append(wav.name)

    if unmatched_count == check_limit:
        sample_audio_keys = list(test_registry.filename_to_pid.keys())[:5]
        sample_pid_keys = list(test_registry.pid_registry.keys())[:5]
        raise ValueError(
            f"\n\n[致命錯誤] Metadata 完全無法對齊！\n"
            f"抽查了前 {check_limit} 個音檔，全部找不到對應標籤，被防禦機制攔截。\n"
            f"音檔檔名範例: {sample_unmatched[:3]}\n"
            f"橋接表 (filename_to_pid) Key 範例: {sample_audio_keys}\n"
            f"參與者表 (pid_registry) Key 範例: {sample_pid_keys}\n"
            f"--> 請檢查 audio_metadata.csv 的欄位，或對照上述檔名差異。"
        )
    
    print("預檢通過，開始分配進程...")
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(_process_single_audio, wav, pt_output_dir): wav
            for wav in all_wavs
        }
        
        with tqdm(total=len(all_wavs), desc="Generating .pt files") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    success_count += 1
                pbar.update(1)

    print(f"\n特徵生成完畢！成功處理 {success_count}/{len(all_wavs)} 筆檔案。")
    print(f"輸出目錄: {pt_output_dir}")

if __name__ == "__main__":
    main()