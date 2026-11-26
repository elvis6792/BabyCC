"""
synthesize_phone_wheeze_v3.py
終極物理聲學合成腳本 v3
結合 Dialogue Match 概念與醫療聲學先驗，將 ICBHI 聽診器喘鳴物理轉換為手機麥克風域信號，
並與 UK 哈氣背景進行物理正確的混合 (Overlap-Add)。
"""
import os
import numpy as np
import hashlib
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.ndimage import median_filter
from tqdm import tqdm
from config import Config

# ── 路徑設定（新增咳嗽路徑）──────────────────────────────────
ICBHI_DIR = Path('./data/ICBHI_final_database')
UK_EXHALE_DIR = Path('./data/UK_audio_preprocessed/exhale')
UK_COUGH_DIR = Path('./data/UK_audio_preprocessed/cough')
ASTHMA_MAP_PATH = './data/asthma_map.csv'
OUTPUT_SYNTH_DIR = Path('./data/synthetic_wheeze_v3')
TARGET_SR = Config.SAMPLE_RATE

# ==========================================
# 1. 物理聲學模擬器 (PhoneMicSimulator v3 - Nyquist 修正版)
# ==========================================
class PhoneMicSimulatorV4:
    def __init__(self, sr=TARGET_SR):
        self.sr = sr
        # 每次處理都重新採樣，模擬不同手機/環境
        self.safe_nyquist = sr / 2.0 - 100.0 
        
        # 手機麥克風基礎頻響 (DC block & anti-alias)
        self.sos_mic_hp = butter(4, 100, btype='high', fs=sr, output='sos')
        self.sos_mic_lp = butter(4, min(7500, self.safe_nyquist), btype='low', fs=sr, output='sos')

    def _randomize_params(self):
        # 每次處理都重新採樣，模擬不同手機/環境
        self.dist_m = np.random.uniform(0.10, 0.50)
        self.rt60 = np.random.uniform(0.08, 0.50)
        self.off_axis_cm = np.random.uniform(10.0, 50.0)
        self.mic_hp = np.random.uniform(60.0, 180.0)
        self.mic_lp = np.random.uniform(5500.0, 8500.0)
        self.tissue_cutoff = np.random.uniform(120.0, 200.0)
        self.air_absorb_factor = np.random.uniform(8000, 14000)

        # Nyquist 保護
        self.mic_lp = min(self.mic_lp, self.safe_nyquist)
        self.mic_hp = min(self.mic_hp, self.safe_nyquist - 200)
        if self.mic_hp >= self.mic_lp:
            self.mic_hp = self.mic_lp * 0.1

    def tissue_to_air(self, y):
        """組織->空氣 輻射濾波 (Cut < 150Hz)"""
        sos = butter(3, self.tissue_cutoff, btype='high', fs=self.sr, output='sos')
        return sosfilt(sos, y).astype(np.float32)

    def air_absorption(self, y):
        """空氣吸收 (高頻衰減)"""
        cutoff = max(1500, int(self.air_absorb_factor - self.dist_m * self.air_absorb_factor))
        cutoff = min(cutoff, self.safe_nyquist)
        sos = butter(2, cutoff, btype='low', fs=self.sr, output='sos')
        return sosfilt(sos, y).astype(np.float32)

    def room_convolution(self, y):
        """房間 RIR 卷積 (Reverb Match)"""
        ir_len = int(self.rt60 * self.sr)
        if ir_len < 1: return y
        t = np.arange(ir_len, dtype=np.float64) / self.sr
        decay = np.exp(-6.9 * t / self.rt60)
        ir = np.zeros(ir_len, dtype=np.float64)
        ir[0] = 1.0
        # Early reflections
        for delay_ms, gain in zip([5.0, 12.0, 19.0], [0.6, 0.4, 0.25]):
            idx = int(np.random.uniform(3.0, 8.0) * self.sr / 1000)
            if idx < ir_len: ir[idx] = gain
        # Late reverb (用長度作為 seed 確保多樣性)
        rng = np.random.default_rng(seed=hash(str(len(y))) % (2**31))
        late_start = int(0.025 * self.sr)
        if late_start < ir_len:
            ir[late_start:] += rng.normal(0, 1, ir_len - late_start) * decay[late_start:] * np.random.uniform(0.2, 0.4)
        ir = ir / (np.max(np.abs(ir)) + 1e-10)
        y_conv = fftconvolve(y, ir.astype(np.float32), mode='full')[:len(y)]
        return y_conv.astype(np.float32)

    def apply_off_axis_roll_off(self, y):
        """離軸距離模擬"""
        hf_attenuation_db = max(0, (self.off_axis_cm / 10.0) * 1.5)
        cutoff = 6000 - int(self.off_axis_cm * 50)
        cutoff = max(3000, min(cutoff, 7500))
        cutoff = min(cutoff, self.safe_nyquist)
        sos = butter(2, cutoff, btype='low', fs=self.sr, output='sos')
        y_filtered = sosfilt(sos, y)
        wet_ratio = min(0.7, hf_attenuation_db / 10.0)
        return (y * (1 - wet_ratio) + y_filtered * wet_ratio).astype(np.float32)

    def mic_response(self, y):
        """手機麥克風頻響"""
        sos_hp = butter(4, self.mic_hp, btype='high', fs=self.sr, output='sos')
        sos_lp = butter(4, self.mic_lp, btype='low', fs=self.sr, output='sos')
        y = sosfilt(sos_hp, y)
        y = sosfilt(sos_lp, y)
        return y.astype(np.float32)

    def airflow_envelope_transfer(self, wheeze_phone, bg_exhale, wheeze_band=(200, 1000)):
        """
        修正模組 B：氣流包絡遷移 (EQ Match 的終極進化)
        """
        n_fft = 2048
        hop = 512
        
        if len(bg_exhale) < n_fft or len(wheeze_phone) < n_fft:
            return wheeze_phone

        S_w = librosa.stft(wheeze_phone, n_fft=n_fft, hop_length=hop)
        S_bg = librosa.stft(bg_exhale, n_fft=n_fft, hop_length=hop)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        wheeze_mask = (freqs >= wheeze_band[0]) & (freqs <= wheeze_band[1])
        non_wheeze_mask = ~wheeze_mask
        
        # 計算全局頻譜包絡 (沿時間軸平均並平滑)
        env_bg = median_filter(np.mean(np.abs(S_bg), axis=1), size=15)
        env_w = median_filter(np.mean(np.abs(S_w), axis=1), size=15)
        
        correction = np.ones_like(S_w, dtype=np.float32)
        transfer_ratio = env_bg / (env_w + 1e-10)
        transfer_ratio = np.clip(transfer_ratio, 0.3, 3.0)  # 限制極端增益
        
        # 只在非喘鳴頻帶應用 correction
        correction[non_wheeze_mask, :] = transfer_ratio[non_wheeze_mask, np.newaxis]
        
        S_matched = S_w * correction
        y_matched = librosa.istft(S_matched, hop_length=hop, length=len(wheeze_phone))
        return y_matched.astype(np.float32)

    def process_base(self, y_stethoscope, randomize=True):
        """基礎鏈路：不含依賴背景的包絡遷移"""
        if randomize:
            self._randomize_params()
        y = self.tissue_to_air(y_stethoscope)
        y = self.air_absorption(y)
        y = self.room_convolution(y)
        y = self.apply_off_axis_roll_off(y)
        y = self.mic_response(y)
        peak = np.max(np.abs(y)) + 1e-10
        y = y / peak * 0.6
        return y.astype(np.float32)

# ==========================================
# 2. 終極混音：保留嘯喘相位
# ==========================================
def physical_mix(bg, wheeze_track, sr, snr_db=0.0, wheeze_region=None):
    min_len = min(len(bg), len(wheeze_track))
    bg = bg[:min_len].astype(np.float32)
    w = wheeze_track[:min_len].astype(np.float32)

    # 定位嘯喘活躍區
    if wheeze_region is not None:
        w_start = max(0, min(wheeze_region[0], min_len))
        w_end = max(0, min(wheeze_region[1], min_len))
        w_active = w[w_start:w_end]
    else:
        active_idx = np.where(np.abs(w) > 1e-6)[0]
        if len(active_idx) == 0:
            return bg
        w_start, w_end = active_idx[0], active_idx[-1] + 1
        w_active = w[w_start:w_end]

    if len(w_active) == 0:
        return bg.astype(np.float32)

    # 能量匹配
    w_rms = np.sqrt(np.mean(w_active ** 2)) + 1e-8
    bg_rms = np.sqrt(np.mean(bg ** 2)) + 1e-8
    target_w_rms = bg_rms * (10 ** (snr_db / 20.0))
    scale = target_w_rms / w_rms
    w_scaled = w * scale

    # 時域直接相加（保留諧波相位完整性）
    mixed = bg + w_scaled

    # # 輕微房間梳狀濾波調幅（±3–8%，不破壞諧波結構）
    # active_len = w_end - w_start
    # if active_len > int(0.05 * sr):
    #     t = np.arange(active_len) / sr
    #     comb_freq = np.random.uniform(30.0, 250.0)
    #     depth = np.random.uniform(0.03, 0.08)
    #     modulation = 1.0 + depth * np.sin(2 * np.pi * comb_freq * t)
    #     mixed[w_start:w_end] *= modulation

    # Soft clip
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.95
    # mixed = np.tanh(mixed * 1.2) / np.tanh(1.2)

    return mixed.astype(np.float32)


# ==========================================
# 3. 主流程
# ==========================================
def extract_pure_wheezes_from_icbhi(simulator):
    """從 ICBHI 中提取純喘鳴，並預先跑完基礎物理鏈路"""
    pure_wheezes = []
    wav_files = list(ICBHI_DIR.glob('*.wav'))
    
    for wav_path in tqdm(wav_files, desc="Mining & Converting ICBHI Wheezes"):
        txt_path = wav_path.with_suffix('.txt')
        if not txt_path.exists(): continue

        # val 檔不參與合成（與 train_final 的 _keep_file 同一把 hash）
        if int(hashlib.md5(wav_path.name.encode()).hexdigest(), 16) % 100 >= 80:
            continue

        try:
            y_full, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception:
            continue
            
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        start, end, crackle, wheeze = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
                    except ValueError:
                        continue
                    if wheeze == 1 and (end - start) > 0.3:
                        s_idx = int(start * sr)
                        e_idx = int(end * sr)
                        seg = y_full[s_idx:e_idx]
                        # 預先轉換為基礎手機域 (不含包絡遷移)
                        seg_phone_base = simulator.process_base(seg)
                        pure_wheezes.append(seg_phone_base)
                        
    print(f"成功提取並轉換了 {len(pure_wheezes)} 段『基礎手機域嘯喘』信號！")
    return pure_wheezes


def _xfade(a, b, n):
    """將 a 的尾部與 b 的頭部做 n 個 sample 的 crossfade"""
    n = int(n)
    if n <= 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b]).astype(np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]]).astype(np.float32)


def _estimate_noise_floor(x, sr):
    """用 RMS 的 10th percentile 估計噪底"""
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=128)[0]
    if len(rms) == 0:
        return 1e-5
    return max(float(np.percentile(rms, 10)), 1e-5)


def _synthesize_gap_noise(nf, gap_len, sr):
    """合成匹配噪底的房間環境音（低通濾波，不是白噪）"""
    rng = np.random.default_rng()
    noise = rng.normal(0.0, nf, gap_len).astype(np.float32)
    sos = butter(2, 2000, btype='low', fs=sr, output='sos')
    noise = sosfilt(sos, noise).astype(np.float32)
    cur = np.sqrt(np.mean(noise ** 2)) + 1e-8
    return (noise * (nf / cur)).astype(np.float32)


def build_participant_cough_map(cough_files, asthma_map_path):
    """
    按受試者分組咳嗽檔。
    同一受試者 = 同一房間、同一支手機、同一噪底 → 拼接後紋理一致。
    """
    if not Path(asthma_map_path).exists():
        return None
    df = pd.read_csv(asthma_map_path)
    df = df[df['sound_type'] == 'cough']
    stem_to_pid = dict(zip(df['file_stem'].astype(str), df['participant_id'].astype(str)))
    pid_map = {}
    for p in cough_files:
        pid = stem_to_pid.get(p.stem)
        if pid is not None:
            pid_map.setdefault(pid, []).append(p)
    pid_map = {k: v for k, v in pid_map.items() if len(v) >= 3}
    print(f"同受試者咳嗽組：{len(pid_map)} 組")
    return pid_map if len(pid_map) > 0 else None


def find_cough_decay_region(y: np.ndarray, sr: int) -> tuple:
    """找到咳嗽的衰減區（post-tussive region）。"""
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]

    if len(rms) == 0:
        return -1, -1

    peak_frame = int(np.argmax(rms))
    peak_rms = float(rms[peak_frame]) + 1e-8

    decay_threshold = peak_rms * 0.30

    # 從 peak 往後找，找到 RMS 首次降到 threshold 以下的位置
    decay_start_frame = peak_frame
    for i in range(peak_frame, len(rms)):
        if rms[i] < decay_threshold:
            decay_start_frame = i
            break

    decay_start_sample = int(decay_start_frame * hop)

    # 衰減區終點：信號結尾
    decay_end_sample = len(y)

    if decay_end_sample - decay_start_sample < int(0.12 * sr):
        return -1, -1

    return decay_start_sample, decay_end_sample


def build_cough_background(cough_files, sr, target_min_len_sec=1.5,
                           participant_map=None):
    """
    v2：噪底匹配填縫 + 15ms crossfade + 同受試者優先。
    消滅數位靜音縫與硬切邊緣。
    """
    target_min_samples = int(target_min_len_sec * sr)
    gap_len = int(0.08 * sr)
    xfade_len = int(0.015 * sr)

    # 優先使用同一受試者的咳嗽（噪底/房間聲學一致）
    candidates = cough_files
    if participant_map is not None:
        keys = list(participant_map.keys())
        pool = participant_map[keys[np.random.randint(len(keys))]]
        if len(pool) >= 2:
            candidates = pool

    segments = []
    total_len = 0
    attempts = 0
    while total_len < target_min_samples and attempts < 8:
        attempts += 1
        path = candidates[np.random.randint(len(candidates))]
        try:
            seg, _ = librosa.load(str(path), sr=sr, mono=True)
        except Exception:
            continue
        if len(seg) < int(0.1 * sr):
            continue
        segments.append(seg)
        total_len += len(seg) + gap_len

    if len(segments) == 0:
        return None

    # 拼接：噪底匹配的環境音填縫 + crossfade 接合
    result = segments[0]
    for seg in segments[1:]:
        nf = min(_estimate_noise_floor(result, sr), _estimate_noise_floor(seg, sr))
        gap = _synthesize_gap_noise(nf, gap_len + 2 * xfade_len, sr)
        result = _xfade(result, gap, xfade_len)
        result = _xfade(result, seg, xfade_len)

    return result.astype(np.float32)


def loop_fill_wheeze(w_base: np.ndarray, target_len: int, sr: int,
                     crossfade_ms: int = 40) -> np.ndarray:
    """
    將喘鳴循環填充到目標長度。
    使用 crossfade 避免循環點的 click。

    物理依據：真實喘鳴在整個呼氣過程中持續存在，
    頻率軌跡會有微小漂移但不會中斷。
    """
    if len(w_base) >= target_len:
        return w_base[:target_len]

    if len(w_base) < int(0.1 * sr):
        # 喘鳴太短，無法循環，直接補零
        return np.pad(w_base, (0, target_len - len(w_base)), mode='constant')

    crossfade_len = int(crossfade_ms * sr / 1000)
    crossfade_len = min(crossfade_len, len(w_base) // 3)

    if crossfade_len < 4:
        # 太短無法 crossfade，直接 tile
        repeats = (target_len // len(w_base)) + 1
        result = np.tile(w_base, repeats)[:target_len]
        return result.astype(np.float32)

    result = w_base.copy()

    while len(result) < target_len:
        remaining = target_len - len(result)
        next_len = min(len(w_base), remaining + crossfade_len)
        next_chunk = w_base[:next_len]

        if len(next_chunk) <= crossfade_len:
            result = np.concatenate([result, next_chunk])
            break

        # Crossfade 拼接
        fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)

        overlap = result[-crossfade_len:] * fade_out + next_chunk[:crossfade_len] * fade_in
        result = np.concatenate([
            result[:-crossfade_len],
            overlap,
            next_chunk[crossfade_len:]
        ])

    return result[:target_len].astype(np.float32)


def synthesize_dataset_v2(wheezes_phone, bg_exhale_dir, bg_cough_dir,
                           out_dir, simulator, num_synths=15000):
    """
    雙背景合成 v4：
    - 咳嗽：拼接多個短片段構建長背景
    - 哈氣：喘鳴循環填充，填滿整個活躍區
    - 零填充軌道 + 整體相位鎖定混合（無拼貼感）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    exhale_files = list(bg_exhale_dir.glob('*.wav'))
    cough_files = list(bg_cough_dir.glob('*.wav'))
    participant_map = build_participant_cough_map(cough_files, ASTHMA_MAP_PATH)

    if not exhale_files or not cough_files:
        print(f"[Error] 找不到背景音檔！哈氣：{len(exhale_files)}，咳嗽：{len(cough_files)}")
        return

    print(f"開始雙背景合成 {num_synths} 筆 (v4)...")
    print(f"   哈氣背景池：{len(exhale_files)} 筆")
    print(f"   咳嗽背景池：{len(cough_files)} 筆")

    cough_count = 0
    exhale_count = 0
    skip_count = 0

    for i in tqdm(range(num_synths)):
        # 1. 隨機抽一個基礎嘯喘
        w_base = wheezes_phone[np.random.randint(len(wheezes_phone))]

        # 2. 決定背景類型：55% 哈氣，45% 咳嗽
        use_cough = np.random.random() < 0.45

        if use_cough:
            background_type = "cough"
            # 拼接多個短咳嗽構建長背景
            bg = build_cough_background(cough_files, TARGET_SR, target_min_len_sec=1.5,
                                        participant_map=participant_map)
            if bg is None or len(bg) < int(1.0 * TARGET_SR):
                skip_count += 1
                continue
        else:
            background_type = "exhale"
            bg_path = exhale_files[np.random.randint(len(exhale_files))]
            try:
                bg, _ = librosa.load(str(bg_path), sr=TARGET_SR, mono=True)
            except Exception:
                skip_count += 1
                continue

        if len(bg) < int(0.5 * TARGET_SR):
            skip_count += 1
            continue

        # 3. 找到合法的插入區域
        if background_type == "cough":
            # 咳嗽：在最後一個 peak 的衰減尾部插入
            rms = librosa.feature.rms(y=bg, frame_length=512, hop_length=128)[0]
            if len(rms) == 0:
                skip_count += 1
                continue
            # 找最後一個高能量幀（最後一次咳嗽）
            threshold = np.max(rms) * 0.3
            active_frames = np.where(rms > threshold)[0]
            if len(active_frames) < 3:
                skip_count += 1
                continue
            # 插入區：最後一個活躍幀之後到結尾
            insert_region_start = int(active_frames[-1] * 128)
            insert_region_end = len(bg)
            available_space = insert_region_end - insert_region_start

        else:
            # 哈氣：在氣流活躍區插入
            rms = librosa.feature.rms(y=bg, frame_length=512, hop_length=128)[0]
            if len(rms) == 0:
                skip_count += 1
                continue
            active_threshold = np.max(rms) * 0.25
            active_frames = np.where(rms > active_threshold)[0]
            if len(active_frames) < 5:
                skip_count += 1
                continue
            insert_region_start = int(active_frames[0] * 128)
            insert_region_end = int(active_frames[-1] * 128)
            available_space = insert_region_end - insert_region_start

        if available_space < int(0.15 * TARGET_SR):
            skip_count += 1
            continue

        # 4. 循環填充喘鳴，填滿整個可用區域
        w_filled = loop_fill_wheeze(w_base, available_space, TARGET_SR)

        # 位置抖動
        jitter = int(np.random.uniform(-0.05, 0.05) * available_space)
        insert_start = max(0, min(insert_region_start + jitter, len(bg) - len(w_filled)))
        w_len = min(len(w_filled), len(bg) - insert_start)

        if w_len < int(0.1 * TARGET_SR):
            skip_count += 1
            continue

        w_final = w_filled[:w_len]

        # 5. 對喘鳴頭尾做漸變（消滅降噪音檔拼接點）
        fade_len = int(0.03 * TARGET_SR)
        if w_len > fade_len * 3:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            w_final = w_final.copy()
            w_final[:fade_len] *= fade_in
            w_final[-fade_len:] *= fade_out

        # 6. 局部包絡遷移
        bg_local = bg[insert_start:insert_start + w_len]
        if len(bg_local) >= 2048 and len(w_final) >= 2048:
            w_matched = simulator.airflow_envelope_transfer(w_final, bg_local)
        else:
            w_matched = w_final

        # 7. 構建零填充軌道
        wheeze_track = np.zeros(len(bg), dtype=np.float32)
        wheeze_track[insert_start:insert_start + w_len] = w_matched[:w_len]

        # 8. 整體相位鎖定混合
        if background_type == "cough":
            snr_db = np.random.uniform(-8.0, 2.0)
        else:
            snr_db = np.random.uniform(-5.0, 3.0)

        mixed = physical_mix(
            bg, wheeze_track, TARGET_SR, snr_db=snr_db,
            wheeze_region=(insert_start, insert_start + w_len)
        )

        # 9. 儲存
        out_path = out_dir / f"synth_wheeze_{background_type}_{i:05d}.wav"
        sf.write(str(out_path), mixed, TARGET_SR)

        if background_type == "cough":
            cough_count += 1
        else:
            exhale_count += 1

    print(f"\n合成完成！")
    print(f"   哈氣背景：{exhale_count} 筆")
    print(f"   咳嗽背景：{cough_count} 筆")
    print(f"   跳過：{skip_count} 筆")

def main():
    print("啟動跨域物理聲學合成管線 v3 (雙背景版)...")
    sim = PhoneMicSimulatorV4(TARGET_SR)
    
    wheezes = extract_pure_wheezes_from_icbhi(sim)
    if len(wheezes) > 10:
        synthesize_dataset_v2(
            wheezes_phone=wheezes,
            bg_exhale_dir=UK_EXHALE_DIR,   # ./data/audio_preprocessed/exhale
            bg_cough_dir=UK_COUGH_DIR,     # ./data/audio_preprocessed/cough
            out_dir=OUTPUT_SYNTH_DIR,
            simulator=sim,
            num_synths=18000,
        )
        print(f"\n合成完成！請查看 {OUTPUT_SYNTH_DIR}")
    else:
        print("[Warning] 提取到的嘯喘片段太少，無法進行有效合成。")

if __name__ == "__main__":
    main()