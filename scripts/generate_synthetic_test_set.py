import numpy as np
import soundfile as sf
import librosa
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from scripts.data_pipeline.wheeze_synthesizer import WheezeInjector

def generate_synthetic_coughs(output_dir: Path, uk_covid_audio_dir: Path, num_files: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    wheeze_dir = PROJECT_ROOT / "data" / "extracted_wheezes"
    
    if not wheeze_dir.exists():
        print(f"[Error] 找不到真實喘鳴特徵目錄: {wheeze_dir}")
        sys.exit(1)
        
    if not uk_covid_audio_dir.exists():
        print(f"[Error] 找不到 UK COVID 乾淨音檔目錄: {uk_covid_audio_dir}")
        sys.exit(1)

    # 抓取 UK COVID 資料夾內的 wav 檔案作為底噪 (Base)
    clean_wavs = list(uk_covid_audio_dir.glob("**/*.wav"))
    if not clean_wavs:
        print("[Error] 目錄中沒有找到任何 wav 檔案")
        sys.exit(1)

    injector = WheezeInjector(wheeze_dir=wheeze_dir, sample_rate=Config.SAMPLE_RATE)
    print(f"開始混合 {num_files} 支「真實咳嗽 + 喘鳴病徵」的測試音檔至 {output_dir}...")
    
    for i in range(num_files):
        # 隨機挑選一支乾淨的咳嗽聲
        clean_audio_path = random.choice(clean_wavs)
        try:
            base_signal, sr = librosa.load(str(clean_audio_path), sr=Config.SAMPLE_RATE)
        except Exception as e:
            print(f"無法讀取 {clean_audio_path.name}, 略過。")
            continue
            
        # 將訊號轉為 float32 並確保長度足夠
        base_signal = base_signal.astype(np.float32)
        if len(base_signal) < int(Config.SAMPLE_RATE * 0.5):
            continue

        snr_db = np.random.uniform(5.0, 12.0) # 控制在較真實的混音比例
        
        mixed_audio = injector.inject(base_signal, snr_db=snr_db)
        
        out_path = output_dir / f"real_mixed_wheeze_test_{i+1:02d}.wav"
        sf.write(str(out_path), mixed_audio, Config.SAMPLE_RATE)
        print(f"成功合成: {out_path.name} (Base: {clean_audio_path.name}, SNR: {snr_db:.2f} dB)")

if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "test_audio_batch"
    uk_covid_dir = Config.UK_AUDIO_DIR
    
    generate_synthetic_coughs(out_dir, uk_covid_dir)