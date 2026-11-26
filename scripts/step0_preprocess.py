"""
step0_preprocess.py
咳嗽 / 哈氣智慧預處理腳本

修正重點：
1. 修復 QUALITY dict key 空格問題
2. strict_quality_check 改成 sound_type aware
3. cough 切割保留更多 peak 後段，避免 whoop 被切掉
4. cough 不使用 fade-in，避免破壞 explosive attack
5. exhale 保留 fade-in，避免截斷爆音
"""

import os
import shutil
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from config import Config
from tqdm import tqdm
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


# ── 路徑設定 ─────────────────────────────────────────────────
AUDIO_DIR = './data/UK_COVID19/audio'
OUTPUT_DIR = './data/UK_audio_preprocessed'
GARBAGE_DIR = './data/UK_COVID19/audio_preprocess_garbage'

SAMPLE_RATE = Config.SAMPLE_RATE


# ── 切割參數 ─────────────────────────────────────────────────
# 註：以下參數在 unified_acoustic_framing 中已不再使用，但保留供參考
CUT = {
    "cough_pre_peak_sec": 0.05,
    "cough_decay_db": 30.0,
    "cough_min_post_sec": 0.35,
    "cough_max_post_sec": 1.20,
    "exhale_onset_db": 20.0,
    "exhale_offset_db": 25.0,
    "min_segment_sec": 0.15,
    "smooth_ms": 20,
    "target_peak_ratio": 0.30, # 錨點對齊位置
}


# ── 品質閾值 ─────────────────────────────────────────────────
QUALITY = {
    "min_duration_sec": 0.2,
    "clipping_amplitude": 0.95,  # 嚴格化：振幅超過 0.95 即視為噴麥邊緣
    "clipping_min_samples": 10,  # 嚴格化：只要有 10 個 sample 觸發 clipping 直接丟棄 (原 100)
    "harmonic_ratio_max": 0.30,
    "abs_silence_rms": 0.003,
    "near_silence_mean_rms": 0.002,
    "crest_factor_min": 4.0,     # 嚴格化：要求更高的峰值因數，剔除被過度動態壓縮的持續性雜音 (原 3.0)
    "active_ratio_min": 0.05,
    "spectral_flat_min": 0.001,
    "mfcc_dog_distance": 9999,
}


# ── 新增：物理防護閾值 ─────────────────────────────────────────
# ── 更新：物理防護閾值 ─────────────────────────────────────────
ACOUSTIC_GUARD = {
    "sub_bass_hz": 80.0,
    "max_sub_bass_ratio": 0.75,
    "min_zcr_for_vocal": 0.015,
    "max_centroid_hz": 3500.0,
    "plosive_crest_factor": 12.0,
    "plosive_lf_hz": 300.0,          # 噴麥的低頻能量檢測上限
    "max_plosive_lf_ratio": 0.85,    # 若 Crest Factor 異常，且低於 300Hz 的能量超過 85%，則判定為噴麥
}


MFCC_PROTOTYPE = np.array([
    -470.3, 32.7, -19.4, -4.1, -9.3, -1.4, -12.7, -8.1,
    -5.1, -3.1, -5.8, 0.1, -4.6, -3.0, -5.3, -2.8,
    -3.5, -2.3, -3.1, -1.1
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════
def get_rms_curve(y, sr, smooth_ms=20):
    """取得平滑後的 RMS 能量曲線"""
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    smooth_frames = max(1, int(smooth_ms * sr / 1000 / hop))
    return median_filter(rms, size=smooth_frames), hop


def frames_to_samples(frame_idx, hop):
    return frame_idx * hop


def fit_segment(y, start_sample, end_sample, target):
    """
    將切出的片段調整為 target 長度
    超過 → 頭尾各截一半
    不足 → 頭尾各補一半 padding
    """
    seg = y[max(0, start_sample):min(len(y), end_sample)]
    current = len(seg)

    if current >= target:
        excess = current - target
        cut_head = excess // 2
        seg = seg[cut_head:cut_head + target]
    else:
        deficit = target - current
        pad_head = deficit // 2
        pad_tail = deficit - pad_head
        seg = np.pad(seg, (pad_head, pad_tail), mode='constant')

    return seg


def fit_segment_peak(y, start_sample, end_sample, peak_sample, target, peak_left_ratio=0.25):
    """
    以 peak 為參考點，保留 peak 前與 peak 後段落。
    對 cough / whooping 較安全，因為 peak 後段需要保留。
    """
    seg_start = max(0, int(start_sample))
    seg_end = min(len(y), int(end_sample))
    peak_sample = int(peak_sample)

    if seg_end <= seg_start:
        return np.zeros(target, dtype=np.float32)

    current = seg_end - seg_start

    if current >= target:
        desired_peak = int(target * peak_left_ratio)
        new_start = peak_sample - desired_peak
        new_end = new_start + target

        if new_start < seg_start:
            new_start = seg_start
            new_end = new_start + target

        if new_end > seg_end:
            new_end = seg_end
            new_start = new_end - target

        new_start = max(seg_start, new_start)
        new_end = min(seg_end, new_start + target)

        seg = y[new_start:new_end]

    else:
        seg = y[seg_start:seg_end]
        deficit = target - len(seg)

        peak_in_seg = peak_sample - seg_start
        desired_peak = int(target * peak_left_ratio)

        pad_left = desired_peak - peak_in_seg
        pad_left = max(0, min(deficit, pad_left))

        pad_right = deficit - pad_left

        seg = np.pad(seg, (pad_left, pad_right), mode='constant')

    if len(seg) < target:
        seg = np.pad(seg, (0, target - len(seg)), mode='constant')

    return seg[:target]


def crossfade_edges(y, sr, fade_ms=10, fade_in=True):
    """
    頭尾 fade。
    cough 建議 fade_in=False，避免破壞 attack。
    """
    fade_samples = int(fade_ms * sr / 1000)

    if fade_samples <= 0 or len(y) < fade_samples * 2:
        return y

    y = y.copy()

    if fade_in:
        y[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)

    y[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    return y


# ═══════════════════════════════════════════════════════════
# Metadata 分類：咳嗽 vs 哈氣
# ═══════════════════════════════════════════════════════════
METADATA_PATH = './data/UK_COVID19/audio_metadata.csv'


def _build_type_map(metadata_path):
    """從 metadata CSV 建立檔名→類型的對照表"""
    type_map = {}

    if not os.path.exists(metadata_path):
        print(f"[Warning] 找不到 metadata 檔案：{metadata_path}")
        return type_map

    df = pd.read_csv(metadata_path)

    for _, row in df.iterrows():
        if pd.notna(row.get('exhalation_file_name')):
            fname = os.path.splitext(row['exhalation_file_name'])[0]
            type_map[fname] = 'exhale'

        if pd.notna(row.get('cough_file_name')):
            fname = os.path.splitext(row['cough_file_name'])[0]
            type_map[fname] = 'cough'

        if pd.notna(row.get('three_cough_file_name')):
            fname = os.path.splitext(row['three_cough_file_name'])[0]
            type_map[fname] = 'cough'

    return type_map


_TYPE_MAP = _build_type_map(METADATA_PATH)

def classify_sound_type(y, sr, filename=None):
    if filename is not None:
        fname = os.path.splitext(os.path.basename(filename))[0]
        if fname in _TYPE_MAP:
            return _TYPE_MAP[fname]

    return 'unknown'



# ═══════════════════════════════════════════════════════════
# 切割：咳嗽與哈氣 (去除破壞性特徵與強制對齊的乾淨版本)
# ═══════════════════════════════════════════════════════════
def cut_cough(y, sr):
    rms, hop = get_rms_curve(y, sr, CUT["smooth_ms"])
    max_rms = np.max(rms) + 1e-8
    peak_frame = int(np.argmax(rms))
    peak_sample = int(peak_frame * hop)

    pre_samples = int(CUT["cough_pre_peak_sec"] * sr)
    start_sample = max(0, peak_sample - pre_samples)

    decay_threshold = max_rms * 10 ** (-CUT["cough_decay_db"] / 20)
    end_frame = peak_frame
    for i in range(peak_frame, len(rms)):
        if rms[i] > decay_threshold:
            end_frame = i
        else:
            if i - end_frame > 5:
                break

    decay_end_sample = min(len(y), frames_to_samples(end_frame + 1, hop))

    min_end_sample = min(
        len(y),
        peak_sample + int(CUT["cough_min_post_sec"] * sr)
    )

    max_end_sample = min(
        len(y),
        peak_sample + int(CUT["cough_max_post_sec"] * sr)
    )

    end_sample = max(decay_end_sample, min_end_sample)
    end_sample = min(end_sample, max_end_sample)

    return y[start_sample:end_sample]

def cut_exhale(y, sr):
    rms, hop = get_rms_curve(y, sr, CUT["smooth_ms"])
    max_rms = np.max(rms) + 1e-8

    onset_threshold = max_rms * 10 ** (-CUT["exhale_onset_db"] / 20)
    offset_threshold = max_rms * 10 ** (-CUT["exhale_offset_db"] / 20)

    onset_frames = np.where(rms > onset_threshold)[0]

    if len(onset_frames) == 0:
        return y

    start_frame = max(0, onset_frames[0] - 3)

    offset_frames = np.where(rms > offset_threshold)[0]
    end_frame = offset_frames[-1] + 5 if len(offset_frames) > 0 else len(rms) - 1
    end_frame = min(end_frame, len(rms) - 1)

    start_sample = frames_to_samples(start_frame, hop)
    end_sample = frames_to_samples(end_frame + 1, hop)

    return y[start_sample:end_sample]

# ═══════════════════════════════════════════════════════════
# 品質清洗
# ═══════════════════════════════════════════════════════════
def detect_mechanical_and_prank(y: np.ndarray, sr: int, sound_type: str) -> tuple[bool, str]:
    """
    純 DSP 物理防護網：攔截撞麥、噴麥與惡作劇音檔。
    利用低頻能量佔比 (Low-Frequency Ratio) 作為噴麥的二次物理驗證，避免誤殺哈氣。
    """
    g = ACOUSTIC_GUARD

    # 1. 計算頻譜與能量分佈
    stft = np.abs(librosa.stft(y, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=Config.N_FFT)
    
    total_energy = np.sum(stft) + 1e-8
    sub_bass_mask = freqs <= g["sub_bass_hz"]
    sub_bass_energy = np.sum(stft[sub_bass_mask, :])
    sub_bass_ratio = float(sub_bass_energy / total_energy)

    # 2. 計算過零率 (Zero-Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0]
    mean_zcr = float(np.mean(zcr))

    # 聯集判定 A：撞麥 (Mic Bumps)
    # 物理特徵：極高比例的超低頻能量 + 缺乏氣流摩擦的高頻過零率
    if sub_bass_ratio > g["max_sub_bass_ratio"] and mean_zcr < g["min_zcr_for_vocal"]:
        return False, "Mic_Bump_Mechanical"

    # 3. 計算頻譜質心 (Spectral Centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)[0]
    mean_centroid = float(np.mean(centroid))

    # 聯集判定 B：惡作劇 / 尖叫 / 電子干擾音 (Pranks)
    if mean_centroid > g["max_centroid_hz"]:
        return False, "Prank_High_Centroid"

    # 4. 噴麥 (Plosives / Wind Noise) 檢測
    # 結合 Crest Factor (時間軸爆發) 與 低頻能量佔比 (頻域物理特徵)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    mean_rms = float(np.mean(rms)) + 1e-6
    peak_amp = float(np.max(np.abs(y)))
    crest_factor = peak_amp / mean_rms

    if crest_factor > g["plosive_crest_factor"]:
        plosive_lf_mask = freqs <= g["plosive_lf_hz"]
        plosive_lf_energy = np.sum(stft[plosive_lf_mask, :])
        plosive_lf_ratio = float(plosive_lf_energy / total_energy)

        # 唯有「瞬間爆發」且「能量極度集中在低頻」，才是真實的風切與噴麥
        # 這能完美放行高 Crest Factor 但寬頻摩擦的正常「哈氣 (exhale)」
        if plosive_lf_ratio > g["max_plosive_lf_ratio"]:
            return False, "Impulse_Plosive"

    return True, "Clean"

def strict_quality_check(y, sr, sound_type='cough'):
    """
    嚴格品質清洗。

    修正：
    - sound_type aware
    - exhale 不使用 spectral_flat_min 過濾 tonal
      否則 wheeze / whoop 會被丟掉
    - harmonic ratio 放寬，避免 voiced cough / wheeze 被丟掉
    """
    t = QUALITY

    if len(y) / sr < t["min_duration_sec"]:
        return False, "Too_Short"

    if np.max(np.abs(y)) >= t["clipping_amplitude"]:
        if np.sum(np.abs(y) >= t["clipping_amplitude"]) > t["clipping_min_samples"]:
            return False, "Clipping"
            
    # ── 插入：物理聲學聯合防護網 ──
    is_safe, mech_reason = detect_mechanical_and_prank(y, sr, sound_type)
    if not is_safe:
        return False, mech_reason
    # ────────────────────────────

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]

    if np.max(rms) < t["abs_silence_rms"]:
        return False, "Absolute_Silence"

    if np.mean(rms) < t["near_silence_mean_rms"]:
        return False, "Near_Silence"

    mean_rms = np.mean(rms)
    crest = np.max(np.abs(y)) / (mean_rms + 1e-6)

    active_ratio = np.sum(rms > np.max(rms) * 0.1) / (len(rms) + 1e-6)
    if active_ratio < t["active_ratio_min"]:
        return False, "Incomplete_Segment"

    if crest < t["crest_factor_min"]:
        return False, "Compressed"

    harmonic_y = librosa.effects.harmonic(y, margin=3.0)
    harmonic_ratio = np.sum(harmonic_y ** 2) / (np.sum(y ** 2) + 1e-6)

    harmonic_limit = 0.45 if sound_type == 'exhale' else 0.30
    if harmonic_ratio > harmonic_limit:
        return False, "Voiced_Ah"

    flatness = float(np.mean(
        librosa.feature.spectral_flatness(
            y=y,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )[0]
    ))

    if sound_type == 'cough' and flatness < t["spectral_flat_min"]:
        return False, "Tonal_Interference"

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    dist = float(np.linalg.norm(
        np.mean(mfcc, axis=1).astype(np.float32) - MFCC_PROTOTYPE
    ))

    if dist > t["mfcc_dog_distance"]:
        return False, f"NonHuman_MFCC_{dist:.0f}"

    return True, "Clean"


# ═══════════════════════════════════════════════════════════
# 主處理函數
# ═══════════════════════════════════════════════════════════
def process_file(file_path): 
    """
    回傳:
        (is_clean, reason, sound_type, processed_audio)
    """
    sound_type = 'unknown'
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        if len(y) / sr < CUT["min_segment_sec"]:
            return False, "Too_Short_Raw", sound_type, None

        intervals = librosa.effects.split(y, top_db=35)
        if len(intervals) > 0:
            y = y[intervals[0][0]:intervals[-1][1]]

        sound_type = classify_sound_type(y, sr, filename=file_path)

        if sound_type == 'unknown':
            return False, "Unknown_Type", sound_type, None

        if sound_type == 'cough':
            seg = cut_cough(y, sr)
        else:
            seg = cut_exhale(y, sr)

        is_clean, reason = strict_quality_check(seg, sr, sound_type=sound_type)

        if not is_clean:
            return False, reason, sound_type, None

        return True, "Clean", sound_type, seg

    except Exception as e:
        error_type = type(e).__name__ 
        error_msg = str(e).replace(" ", "_").replace(",", "") 
        # 回傳崩潰前最後得知的 sound_type
        return False, f"Err_{error_type}_{error_msg}"[:60], sound_type, None 

# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GARBAGE_DIR, exist_ok=True)

    audio_files = sorted([
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith('.wav')
    ])

    print(f"開始預處理 {len(audio_files)} 筆音檔...")
    print(f"   輸入：{AUDIO_DIR}")
    print(f"   輸出：{OUTPUT_DIR}\n")

    report_data = []

    cough_count = 0
    exhale_count = 0

    for fname in tqdm(audio_files):
        src_path = os.path.join(AUDIO_DIR, fname)

        is_clean, reason, sound_type, seg = process_file(src_path)

        report_data.append({
            "FileName": fname,
            "Status": "Clean" if is_clean else "Garbage",
            "SoundType": sound_type,
            "Reason": reason,
        })

        if is_clean and seg is not None:
            type_dir = os.path.join(OUTPUT_DIR, sound_type)
            os.makedirs(type_dir, exist_ok=True)

            out_path = os.path.join(
                type_dir,
                os.path.splitext(fname)[0] + '.wav'
            )

            sf.write(out_path, seg, SAMPLE_RATE)

            if sound_type == 'cough':
                cough_count += 1
            else:
                exhale_count += 1

        else:
            try:
                shutil.copy(
                    src_path,
                    os.path.join(GARBAGE_DIR, f"[{reason}]_{fname}")
                )
            except Exception:
                pass

    df = pd.DataFrame(report_data)
    df.to_csv('./preprocess_report.csv', index=False)

    total = len(df)
    n_clean = int((df["Status"] == "Clean").sum())

    print("\n預處理完成！")

    if total > 0:
        print(f"   乾淨：{n_clean}/{total} ({n_clean / total * 100:.1f}%)")
    else:
        print("   乾淨：0/0")

    print(f"   咳嗽：{cough_count} 筆")
    print(f"   哈氣：{exhale_count} 筆")

    print("\n各垃圾類型統計：")
    print(df[df["Status"] == "Garbage"]["Reason"].value_counts().to_string())

    print("\n分類統計：")
    print(df["SoundType"].value_counts().to_string())


if __name__ == "__main__":
    main()