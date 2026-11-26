import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config

class ICBHIFeatureExtractor:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_sr = Config.SAMPLE_RATE
        self.min_duration = 0.5  
        self.fade_samples = int(self.target_sr * 0.05)  

    def _apply_zero_phase_bandpass(self, y: np.ndarray) -> np.ndarray:
        nyq = 0.5 * self.target_sr
        low_cut = 150.0 / nyq
        high_cut = 2500.0 / nyq
        
        b, a = butter(4, [low_cut, high_cut], btype='band')
        y_filtered = filtfilt(b, a, y)
        return y_filtered

    def _apply_fade(self, y: np.ndarray) -> np.ndarray:
        if len(y) < self.fade_samples * 2:
            return y
            
        fade_in = np.linspace(0.0, 1.0, self.fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, self.fade_samples, dtype=np.float32)
        
        y_faded = y.copy()
        y_faded[:self.fade_samples] *= fade_in
        y_faded[-self.fade_samples:] *= fade_out
        return y_faded

    def process_database(self):
        txt_files = list(self.input_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No annotation files found in {self.input_dir}")

        extracted_count = 0
        
        for txt_path in tqdm(txt_files, desc="Extracting ICBHI Wheezes"):
            wav_path = txt_path.with_suffix('.wav')
            if not wav_path.exists():
                continue

            annotations = np.loadtxt(txt_path, ndmin=2)
            if annotations.size == 0:
                continue

            try:
                y, sr = librosa.load(str(wav_path), sr=self.target_sr, mono=True)
            except Exception:
                continue

            for idx, row in enumerate(annotations):
                start_time, end_time, crackle, wheeze = row
                
                if wheeze != 1:
                    continue
                    
                duration = end_time - start_time
                if duration < self.min_duration:
                    continue

                start_sample = int(start_time * self.target_sr)
                end_sample = int(end_time * self.target_sr)
                
                if end_sample > len(y):
                    end_sample = len(y)
                    
                segment = y[start_sample:end_sample]
                if len(segment) == 0:
                    continue

                segment_filtered = self._apply_zero_phase_bandpass(segment)
                segment_faded = self._apply_fade(segment_filtered)
                
                max_val = np.max(np.abs(segment_faded))
                if max_val > 0:
                    segment_faded = segment_faded / max_val

                out_filename = f"{txt_path.stem}_wheeze_{idx}.wav"
                out_path = self.output_dir / out_filename
                
                sf.write(str(out_path), segment_faded, self.target_sr)
                extracted_count += 1

        return extracted_count

if __name__ == "__main__":
    print("==========================================================")
    print("啟動 ICBHI 病理特徵萃取管線")
    print("==========================================================")
    
    icbhi_dir = Config.ICBHI_AUDIO_DIR
    extracted_db_dir = Config.PROJECT_ROOT / "data" / "extracted_wheezes"
    
    extractor = ICBHIFeatureExtractor(input_dir=icbhi_dir, output_dir=extracted_db_dir)
    
    try:
        count = extractor.process_database()
        print(f"\n萃取完成！成功提取了 {count} 筆純淨的 Wheeze 特徵。")
        print(f"儲存路徑: {extracted_db_dir}")
    except Exception as e:
        print(f"[Error] 萃取失敗: {str(e)}")