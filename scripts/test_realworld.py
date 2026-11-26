"""
test_realworld.py
端到端真實音檔測試 (兩階段預測流程)

此腳本採用兩階段預測流程，不再依賴外部 metadata 分類：
1. 初步切割與分類：對音檔進行通用切割，送入模型以判斷聲音類型（咳嗽/哈氣）。
2. 優化切割與分析：根據模型判斷的類型，採用最優的切割演算法重新處理音檔，再次送入模型以獲得最精準的 7 維特徵分數。
"""

import os
import sys
from pathlib import Path

# 將專案根目錄 (scripts/ 的上一層) 加入到 Python 的搜尋路徑中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import logging
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec

from config import Config
from step0_preprocess import cut_cough, cut_exhale
from step1_extract_features import get_teacher_6d_scores
from models.hpss_multitask_network import HPSSMultiTaskEdgeNet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── 設定 ─────────────────────────────────────────────────────
TEST_FILES = [
    "./data/88.wav",
    "./data/WheezingCough.mp3",
]

DIM_NAMES = ['Wheezing', 'Whooping', 'Duration', 'Dry', 'Bovine', 'Explosive', 'SoundType']
DIM_DESC = {
    'Wheezing'  : '連續喘鳴音（氣道狹窄）',
    'Whooping'  : '頻率上升的吸氣聲',
    'Duration'  : '有效音長',
    'Dry'       : '乾咳衰減特徵',
    'Bovine'    : '低頻能量佔比',
    'Explosive' : '爆發性（起音速度）',
    'SoundType' : '聲音類型（1=咳嗽 0=哈氣）',
}

# ─────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────

def load_model(device: torch.device) -> HPSSMultiTaskEdgeNet:
    """載入新版 HPSSMultiTaskEdgeNet 權重，並具備架構血統驗證"""
    checkpoint_path = Path(Config.CHECKPOINT_DIR) / f"best_{Config.MODEL_NAME}_v2.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型權重檔案：{checkpoint_path}")
        
    logger.info("載入模型權重檔案：%s", str(checkpoint_path))
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    # 架構血統驗證 (Architecture Validation)
    # 防範載入到舊版 MobileNetV2 或未完成訓練的殘缺權重
    saved_arch = ckpt.get("architecture", "Unknown_Old_Architecture")
    expected_arch = "HPSSMultiTaskEdgeNet"
    
    if saved_arch != expected_arch:
        logger.error(f"架構不匹配！預期: {expected_arch}，但檔案內是: {saved_arch}")
        logger.error("這代表您目前的 checkpoint 其實是舊版架構的權重。")
        logger.error("解決方案：請先確保 python scripts/step2_train_mobilenet.py 已完整執行並產出新權重。")
        sys.exit(1)
        
    model = HPSSMultiTaskEdgeNet(
        backbone_name=Config.MODEL_NAME,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    
    # 嚴格對齊權重
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model

def extract_features(y: np.ndarray, sr: int):
    """從 step1_extract_features.py 移植，確保特徵提取邏輯 100% 一致"""
    teacher_scores = get_teacher_6d_scores(y, sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(3.0, 3.0))
    
    mel_full = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    mel_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=Config.N_MELS, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    mel_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=Config.N_MELS, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    
    ref_max = float(np.max(mel_full)) + 1e-8
    
    log_mel_full = librosa.power_to_db(mel_full, ref=ref_max)
    log_mel_harmonic = librosa.power_to_db(mel_harmonic, ref=ref_max)
    log_mel_percussive = librosa.power_to_db(mel_percussive, ref=ref_max)
    log_mel_for_viz = librosa.power_to_db(mel_full, ref=np.max(mel_full))
    
    pad_value = float(np.min(log_mel_full)) if log_mel_full.size else -100.0
    pad_value = min(pad_value, -80.0)
    
    from step1_extract_features import _pad_or_trim_mel
    log_mel_full_padded = _pad_or_trim_mel(log_mel_full, Config.TARGET_FRAMES, pad_value=pad_value)
    log_mel_harmonic_padded = _pad_or_trim_mel(log_mel_harmonic, Config.TARGET_FRAMES, pad_value=pad_value)
    log_mel_percussive_padded = _pad_or_trim_mel(log_mel_percussive, Config.TARGET_FRAMES, pad_value=pad_value)
    
    tensor_mel = torch.stack([
        torch.from_numpy(log_mel_full_padded).float(),
        torch.from_numpy(log_mel_harmonic_padded).float(),
        torch.from_numpy(log_mel_percussive_padded).float()
    ])
    
    return tensor_mel, teacher_scores, log_mel_for_viz

def predict(model: nn.Module, tensor_mel: torch.Tensor, device: torch.device) -> np.ndarray:
    """與 step2_train_mobilenet.py 完全對齊的預處理與推論流程"""
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        inp = tensor_mel.unsqueeze(0).to(device)
        
        inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        
        inp = (inp + 100.0) / 100.0
        inp = inp.clamp(0.0, 1.0)
        
        inp = (inp - imagenet_mean) / imagenet_std

        logits = model(inp)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    return pred

def align_peak_30(y, sr):
    """
    與 step1_extract_features.py 完全同步的音訊對齊與填充邏輯
    """
    rms = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0]
    if len(rms) == 0 or np.max(rms) <= 1e-6:
        return np.zeros(Config.TARGET_SAMPLES, dtype=np.float32)
        
    peak_height = max(float(np.mean(rms) + 0.1 * np.std(rms)), float(0.05 * np.max(rms)), 1e-4)
    peak_distance = max(1, int(0.25 * sr / Config.HOP_LENGTH))
    peaks, _ = find_peaks(rms, height=peak_height, distance=peak_distance)
    
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(rms))])
        
    main_peak = int(peaks[np.argmax(rms[peaks])])
    peak_sample = main_peak * Config.HOP_LENGTH
    
    peak_left = int(Config.TARGET_SAMPLES * 0.30)
    start_sample = peak_sample - peak_left
    end_sample = start_sample + Config.TARGET_SAMPLES
    
    if start_sample < 0:
        end_sample = min(len(y), end_sample - start_sample)
        start_sample = 0
    if end_sample > len(y):
        start_sample = max(0, start_sample - (end_sample - len(y)))
        end_sample = len(y)
        
    y_out = y[start_sample:end_sample]
    
    if len(y_out) < Config.TARGET_SAMPLES:
        deficit = Config.TARGET_SAMPLES - len(y_out)
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        y_out = np.pad(y_out, (pad_left, pad_right), mode='reflect')
    elif len(y_out) > Config.TARGET_SAMPLES:
        y_out = y_out[:Config.TARGET_SAMPLES]
        
    return y_out

def run_two_stage_prediction(model, y_raw, sr, device):
    logger.info("[Stage 1] 使用包容性 VAD (Inclusive Probing) 進行類型判定...")

    probe_y = cut_cough(y_raw, sr)
    prelim_aligned = align_peak_30(probe_y, sr)
    
    prelim_mel, prelim_dsp, prelim_viz = extract_features(prelim_aligned, sr)
    prelim_pred = predict(model, prelim_mel, device)
    
    model_sound_type = 'cough' if prelim_pred[6] > 0.5 else 'exhale'
    logger.info(f"[Stage 1] 模型判定類型: '{model_sound_type}' (Score: {prelim_pred[6]:.3f})")
    
    if model_sound_type == 'cough':
        logger.info("[Stage 2] 確認為咳嗽，保留高保真長尾特徵...")
        return prelim_pred, prelim_dsp, prelim_viz, model_sound_type, prelim_aligned
    else:
        logger.info("[Stage 2] 確認為哈氣，切換為哈氣專用切割以排除雜音...")
        final_cut = cut_exhale(y_raw, sr)
        final_segment = align_peak_30(final_cut, sr)
        final_mel, dsp_scores, log_mel_for_viz = extract_features(final_segment, sr)
        final_pred = predict(model, final_mel, device)
        return final_pred, dsp_scores, log_mel_for_viz, model_sound_type, final_segment

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_result(wav_path, y, sr, log_mel_viz, dsp_scores, model_pred, model_sound_type, out_path):
    duration = len(y) / sr
    n_dims = len(DIM_NAMES)
    fname = os.path.basename(wav_path)

    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 2, 2], hspace=0.45, wspace=0.35)

    ax_wave = fig.add_subplot(gs[0, :])
    times = np.linspace(0, duration, len(y))
    ax_wave.plot(times, y, color='#4fc3f7', linewidth=0.6, alpha=0.9)
    ax_wave.set_xlim(0, duration)
    ax_wave.set_facecolor('#0d0d1a')
    ax_wave.set_ylabel('Amplitude', color='white', fontsize=9)
    ax_wave.tick_params(colors='white', labelsize=8)
    title = f'{fname}  |  Model-Predicted Type: {"咳嗽" if model_sound_type=="cough" else "哈氣"}  |  時長: {duration:.2f}s'
    ax_wave.set_title(title, color='white', fontsize=11, fontweight='bold')
    for spine in ax_wave.spines.values():
        spine.set_edgecolor('#333')

    ax_mel = fig.add_subplot(gs[1, :])
    import librosa.display
    img = librosa.display.specshow(log_mel_viz, sr=sr, hop_length=Config.HOP_LENGTH, x_axis='time', y_axis='mel', fmin=20, fmax=8000, ax=ax_mel, cmap='magma', shading='gouraud')
    ax_mel.set_facecolor('#0d0d1a')
    ax_mel.set_xlabel('Time (s)', color='white', fontsize=9)
    ax_mel.set_ylabel('Frequency (Hz)', color='white', fontsize=9)
    ax_mel.tick_params(colors='white', labelsize=8)
    cbar = fig.colorbar(img, ax=ax_mel, format='%+2.0f dB', pad=0.01)
    cbar.ax.tick_params(colors='white', labelsize=7)
    for spine in ax_mel.spines.values():
        spine.set_edgecolor('#333')

    ax_dsp = fig.add_subplot(gs[2, 0])
    dims = DIM_NAMES[:n_dims]
    colors_dsp = ['#ff6b6b' if s > 0.5 else '#4fc3f7' for s in dsp_scores[:n_dims]]
    bars = ax_dsp.barh(dims, dsp_scores[:n_dims], color=colors_dsp, alpha=0.85)
    ax_dsp.set_xlim(0, 1)
    ax_dsp.axvline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_dsp.set_facecolor('#0d0d1a')
    ax_dsp.set_xlabel('Score', color='white', fontsize=9)
    ax_dsp.set_title('DSP 教師分數 (for reference)', color='#aaaaaa', fontsize=10)
    ax_dsp.tick_params(colors='white', labelsize=8)
    for spine in ax_dsp.spines.values():
        spine.set_edgecolor('#333')
    for i, bar in enumerate(bars):
        val = dsp_scores[i]
        text_val = f'{val:.2f}'
        diff = val - model_pred[i]
        if abs(diff) > 0.25:
            text_val += " (!)"
        ax_dsp.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2, text_val, va='center', color='white', fontsize=8)

    ax_pred = fig.add_subplot(gs[2, 1])
    colors_pred = ['#ff6b6b' if s > 0.5 else '#4fc3f7' for s in model_pred[:n_dims]]
    bars2 = ax_pred.barh(dims, model_pred[:n_dims], color=colors_pred, alpha=0.85)
    ax_pred.set_xlim(0, 1)
    ax_pred.axvline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_pred.set_facecolor('#0d0d1a')
    ax_pred.set_xlabel('Score', color='white', fontsize=9)
    ax_pred.set_title('模型預測分數 (Final)', color='#aaaaaa', fontsize=10)
    ax_pred.tick_params(colors='white', labelsize=8)
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('#333')
    for i, bar in enumerate(bars2):
        val = model_pred[i]
        text_val = f'{val:.2f}'
        diff = val - dsp_scores[i]
        if abs(diff) > 0.25:
            text_val += " (!)"
        ax_pred.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2, text_val, va='center', color='white', fontsize=8)

    fig.patch.set_facecolor('#1a1a2e')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   報告已存至：{out_path}")

# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────

def main():
    results_dir = Path("./test_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    print(f"使用設備：{device}")
    
    try:
        model = load_model(device)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print(f"   請確認 '{Path(Config.CHECKPOINT_DIR) / f'best_{Config.MODEL_NAME}_v2.pt'}' 檔案是否存在，且您已執行過 step2_train_mobilenet.py。")
        return
    except Exception as e:
        print(f"[Error] 載入模型時發生未預期的錯誤: {e}")
        return

    for wav_path_str in TEST_FILES:
        wav_path = Path(wav_path_str)
        if not wav_path.exists():
            print(f"[Error] 找不到檔案：{str(wav_path)}")
            continue

        fname = wav_path.name
        print(f"\n處理檔案：{fname}")

        try:
            y, sr = librosa.load(wav_path_str, sr=Config.SAMPLE_RATE, mono=True)
            if len(y) < 100:
                print("   [Warning] 音檔過短，略過。")
                continue
        except Exception as e:
            print(f"   [Error] 讀取音檔失敗: {e}")
            continue

        # 執行兩階段預測流程
        final_pred, dsp_scores, log_mel_for_viz, model_sound_type, final_segment = run_two_stage_prediction(model, y, sr, device)
        
        # 補上 SoundType 維度給 DSP 分數 (僅供參考比較)
        type_score_dsp = 1.0 if model_sound_type == 'cough' else 0.0
        dsp_scores = np.append(dsp_scores, type_score_dsp)


        print(f"\n   --- 最終預測結果 (類型: {model_sound_type}) ---")
        print(f"   {'維度':<12} {'DSP分數 (參考)':>15} {'模型預測':>10} {'差異':>8}")
        print(f"   {'-'*55}")
        for i, name in enumerate(DIM_NAMES):
            dsp = dsp_scores[i]
            pred = final_pred[i]
            diff = pred - dsp
            diff_str = f"{diff:+.3f}"
            print(f"   {name:<12} {dsp:>15.3f} {pred:>10.3f} {diff_str:>8}")

        out_path = results_dir / f"{wav_path.stem}_result_2stage.png"
        plot_result(str(wav_path), final_segment, sr, log_mel_for_viz, dsp_scores, final_pred, model_sound_type, str(out_path))

    print(f"\n測試完成！結果圖表已妥善儲存至：{str(results_dir)}")

if __name__ == "__main__":
    main()
