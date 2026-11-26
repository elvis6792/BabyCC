# dataset_uk_v2.py
"""
UK 資料集 v2：整合氣喘先驗的弱監督 Dataset。

核心邏輯：
- 從 asthma_map.csv 讀取每個音檔的氣喘狀態
- DSP 教師分數 + 氣喘狀態 → 動態 sample weight
- 氣喘患者的高分 → 高信心 positive
- 非氣喘患者的高分 → 極可能是雜訊，降低權重
"""
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from config import Config


class UKDatasetV2(Dataset):
    """
    UK 資料集，整合 participant-level 氣喘先驗。

    每個樣本包含：
    - mel: [3, N_MELS, T] 3 通道 HPSS Mel
    - teacher_scores: [3] = [wheezing, duration, sound_type]
    - asthma_status: 0.0 或 1.0
    - sample_weight: 動態計算的損失權重
    - pseudo_wheeze_label: DSP 教師產生的偽標籤
    """

    def __init__(
        self,
        cache_dir: str,
        asthma_map_path: str,
        split: str = "train",
        split_ratio: float = 0.8,
        wheeze_score_pos_thresh: float = 0.20,
        wheeze_score_neg_thresh: float = 0.10,
    ):
        self.cache_dir = Path(cache_dir)
        self.wheeze_pos_thresh = wheeze_score_pos_thresh
        self.wheeze_neg_thresh = wheeze_score_neg_thresh

        # ── 讀取氣喘映射表 ──
        self.asthma_map = {}
        if Path(asthma_map_path).exists():
            map_df = pd.read_csv(asthma_map_path)
            for _, row in map_df.iterrows():
                self.asthma_map[row['file_stem']] = {
                    'asthma': float(row['asthma']),
                    'shortness_of_breath': float(row['shortness_of_breath']),
                    'sound_type': row['sound_type'],
                }
            print(f"已載入氣喘映射表：{len(self.asthma_map)} 筆")
        else:
            print(f"[Warning] 找不到氣喘映射表：{asthma_map_path}")
            print(f"   請先執行 scripts/build_asthma_map.py")

        # ── 掃描 .pt 快取 ──
        self.samples = []
        for pt_file in sorted(self.cache_dir.glob("*.pt")):
            self.samples.append(str(pt_file))

        # ── hash split ──
        import hashlib
        if split == "train":
            self.samples = [
                s for s in self.samples
                if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100
                   < int(split_ratio * 100)
            ]
        else:
            self.samples = [
                s for s in self.samples
                if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100
                   >= int(split_ratio * 100)
            ]

        print(f"UK {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _compute_sample_weight(self, wheeze_score: float, asthma: float, sob: float) -> float:
        """
        終極版：四象限臨床先驗加權邏輯。
        
        核心目標：
        1. 極大化象限 A (氣喘+呼吸困難) 的正樣本影響力。
        2. 強制模型學習象限 C (COVID/非氣喘呼吸困難) 的雜音不是喘鳴。
        3. 告訴模型象限 B (氣喘緩解期) 沒有喘鳴信號。
        """
        is_positive = wheeze_score > self.wheeze_pos_thresh  # 預設 > 0.20
        is_negative = wheeze_score < self.wheeze_neg_thresh  # 預設 < 0.10
        
        has_asthma = asthma > 0.5
        has_sob = sob > 0.5

        if has_asthma and has_sob:
            # 【象限 A】黃金正樣本 (氣喘發作中)
            if is_positive:
                weight = 1.0   # 極高信心：臨床症狀 + DSP 物理特徵 雙重確認
            else:
                weight = 0.3   # 可能是錄音品質不佳沒錄到，保留一點權重
            
        elif has_asthma and not has_sob:
            # 【象限 B】寂靜胸 / 緩解期 (有氣喘但當下沒發作)
            if is_positive:
                weight = 0.1   # 極大概率是 DSP 教師把環境音/喉嚨音誤判為喘鳴
            else:
                weight = 0.5   # 可靠負樣本：沒發作就該是正常呼吸音
                
        elif not has_asthma and has_sob:
            # 【象限 C】Hard Negative (如 COVID 肺炎，呼吸困難但無氣喘)
            if is_positive:
                weight = 0.05  # 極度懲罰：DSP 教師被 COVID 的粗糙痰音/肺炎雜音騙了！
            else:
                weight = 0.6   # 加重負樣本權重：強制模型學會「這只是肺炎，不是氣喘」
                
        else:
            # 【象限 D】健康對照組 (無氣喘、無呼吸困難)
            if is_positive:
                weight = 0.05  # 懲罰：健康人不該有喘鳴，DSP 誤判
            else:
                weight = 0.5   # 極可靠的基線負樣本

        return weight

    def __getitem__(self, idx):
        pt_path = self.samples[idx]
        data = torch.load(pt_path, weights_only=True)

        mel = data["mel"]           # [3, N_MELS, T]
        scores = data["scores"]     # [3] = [wheezing, duration, sound_type]

        wheeze_score = float(scores[0])
        type_score = float(scores[2])

        # ── 查找氣喘狀態 ──
        file_stem = Path(pt_path).stem
        meta = self.asthma_map.get(file_stem, {
            'asthma': 0.0,
            'shortness_of_breath': 0.0,
            'sound_type': 'unknown',
        })

        asthma = meta['asthma']
        sob = meta['shortness_of_breath']

        # ── 計算 pseudo label ──
        pseudo_wheeze = 1.0 if wheeze_score > self.wheeze_pos_thresh else 0.0

        # ── 計算動態權重 ──
        weight = self._compute_sample_weight(wheeze_score, asthma, sob)

        # ── 咳嗽標籤 ──
        cough_label = 1.0 if type_score > 0.5 else 0.0

        return {
            "mel": mel,
            "wheeze_label": torch.tensor(
                [[pseudo_wheeze]], dtype=torch.float32
            ),
            "cough_label": torch.tensor(
                [[cough_label]], dtype=torch.float32
            ),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "asthma_status": torch.tensor(asthma, dtype=torch.float32),
            "source": "uk",
        }