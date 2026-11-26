# Script Name: scripts/unified_audio_processor.py
# Target Function: unified_acoustic_framing

import numpy as np
import librosa
from scipy.signal import find_peaks

def unified_acoustic_framing(y_raw: np.ndarray, sr: int, target_samples: int = 40000) -> np.ndarray:
    """
    統一聲學特徵擷取框架 (Unified Acoustic Framing)
    
    核心理念：達成 100% Train-Serve Parity。
    無論聲音類別為何，一律使用相同的物理規則進行錨點對齊與擷取，徹底消滅前處理造成的標籤洩漏。
    強迫神經網路學習真實的聲學特徵（如爆發起音），而非學習不同切割演算法留下的邊界痕跡。
    """
    if len(y_raw) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    # 1. 萃取 RMS 能量包絡線
    hop_length = 256
    n_fft = 1024
    rms = librosa.feature.rms(y=y_raw, frame_length=n_fft, hop_length=hop_length)[0]
    
    if np.max(rms) <= 1e-6:
        return np.zeros(target_samples, dtype=np.float32)

    # 2. 尋找全域能量錨點 (Anchor)
    peak_height = max(float(np.mean(rms) + 0.1 * np.std(rms)), float(0.05 * np.max(rms)), 1e-4)
    peak_distance = max(1, int(0.25 * sr / hop_length))
    peaks, _ = find_peaks(rms, height=peak_height, distance=peak_distance)
    
    if len(peaks) == 0:
        # Fallback: 若無明顯突波 (如平緩哈氣)，取全域最大能量點
        main_peak = int(np.argmax(rms))
    else:
        main_peak = int(peaks[np.argmax(rms[peaks])])
        
    peak_sample = main_peak * hop_length
    
    # 3. 錨點對齊 (將最大能量點固定於視窗 30% 處)
    peak_left = int(target_samples * 0.30)
    start_sample = peak_sample - peak_left
    end_sample = start_sample + target_samples
    
    # 4. 視窗擷取與動態邊界處理
    if start_sample < 0:
        # 若峰值太靠前，將視窗貼齊音檔起始點
        end_sample = min(len(y_raw), end_sample - start_sample)
        start_sample = 0
    if end_sample > len(y_raw):
        # 若峰值太靠後，將視窗貼齊音檔結束點
        start_sample = max(0, start_sample - (end_sample - len(y_raw)))
        end_sample = len(y_raw)
        
    y_framed = y_raw[start_sample:end_sample]
    
    # 5. 環狀填充 (Circular Padding)
    # 捨棄絕對零填充以避免特徵洩漏。使用 wrap 保持自然底噪與頻率分佈的連續性。
    if len(y_framed) < target_samples:
        deficit = target_samples - len(y_framed)
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        y_framed = np.pad(y_framed, (pad_left, pad_right), mode='wrap')
    elif len(y_framed) > target_samples:
        y_framed = y_framed[:target_samples]
        
    return y_framed.astype(np.float32)