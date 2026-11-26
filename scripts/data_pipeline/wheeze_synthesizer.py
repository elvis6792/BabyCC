import numpy as np
import librosa
from pathlib import Path

class WheezeInjector:
    def __init__(self, wheeze_dir: Path, sample_rate: int = 16000, seed: int = 42):
        self.sr = sample_rate
        self.rng = np.random.default_rng(seed)
        
        self.wheeze_files = list(wheeze_dir.glob("*.wav"))
        if not self.wheeze_files:
            raise FileNotFoundError(f"No valid .wav files found in {wheeze_dir}")

    def _match_length(self, wheeze: np.ndarray, target_len: int) -> np.ndarray:
        w_len = len(wheeze)
        if w_len == target_len:
            return wheeze
        elif w_len > target_len:
            start_idx = self.rng.integers(0, w_len - target_len)
            return wheeze[start_idx : start_idx + target_len]
        else:
            repeats = (target_len // w_len) + 1
            tiled_wheeze = np.tile(wheeze, repeats)
            return tiled_wheeze[:target_len]

    def inject(self, y_original: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
        if len(y_original) == 0:
            return y_original
            
        target_len = len(y_original)
        
        selected_wheeze_path = self.rng.choice(self.wheeze_files)
        raw_wheeze, _ = librosa.load(str(selected_wheeze_path), sr=self.sr, mono=True)
        
        if len(raw_wheeze) == 0:
            return y_original
            
        matched_wheeze = self._match_length(raw_wheeze, target_len)
        
        hop_len = 256
        rms_env = librosa.feature.rms(y=y_original, frame_length=1024, hop_length=hop_len)[0]
        
        t_orig = np.arange(target_len)
        t_rms = np.arange(len(rms_env)) * hop_len
        rms_env_interp = np.interp(t_orig, t_rms, rms_env)
        
        env_max = np.max(rms_env_interp)
        if env_max < 1e-6:
            return y_original
        rms_env_norm = rms_env_interp / env_max
        
        shaped_wheeze = matched_wheeze * rms_env_norm
        
        power_orig = np.mean(y_original ** 2) + 1e-8
        power_wheeze = np.mean(shaped_wheeze ** 2) + 1e-8
        
        target_power_wheeze = power_orig * (10 ** (snr_db / 10))
        scaling_factor = np.sqrt(target_power_wheeze / power_wheeze)
        
        injected_audio = y_original + (shaped_wheeze * scaling_factor)
        
        max_val = np.max(np.abs(injected_audio))
        if max_val > 0.95:
            injected_audio = injected_audio * (0.95 / max_val)
            
        return injected_audio.astype(np.float32)