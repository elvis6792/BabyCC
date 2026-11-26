import pandas as pd
import numpy as np
from pathlib import Path

class MetadataRegistry:
    def __init__(self, uk_covid_csv_path: str, icbhi_dir: str, uk_audio_csv_path: str = None):
        self.uk_covid_csv = Path(uk_covid_csv_path)
        self.uk_audio_csv = Path(uk_audio_csv_path) if uk_audio_csv_path else None
        self.icbhi_dir = Path(icbhi_dir)
        
        # pid_registry 存放: PID -> 標籤
        self.pid_registry = {}
        # filename_to_pid 存放: 音檔名/UUID -> PID (Bridge Table)
        self.filename_to_pid = {}
        
        self._build_registry()

    def _build_registry(self):
        # 1. 載入 UK COVID 參與者資料
        if self.uk_covid_csv.exists():
            df = pd.read_csv(self.uk_covid_csv)
            if 'participant_identifier' in df.columns:
                for _, row in df.iterrows():
                    pid = str(row['participant_identifier']).strip()
                    if pid and pid.lower() != 'nan':
                        self.pid_registry[pid] = {
                            "wheeze": -1.0,
                            "duration": -1.0,
                            "sound_type": 1.0,
                            "domain": 1.0 
                        }
            else:
                raise ValueError(f"[致命錯誤] 參與者表缺少 participant_identifier 欄位。")

        # 2. 載入 UK COVID 音檔橋接資料
        if self.uk_audio_csv and self.uk_audio_csv.exists():
            df_audio = pd.read_csv(self.uk_audio_csv)
            
            pid_col = 'participant_identifier'
            audio_cols = ['exhalation_file_name', 'cough_file_name', 'three_cough_file_name']
            
            if pid_col in df_audio.columns:
                for _, row in df_audio.iterrows():
                    pid = str(row[pid_col]).strip()
                    if pd.isna(row[pid_col]) or pid.lower() == 'nan': 
                        continue
                    
                    # 將同一個 PID 底下的三個音檔都註冊進橋接表
                    for col in audio_cols:
                        if col in df_audio.columns:
                            fname = str(row[col]).strip()
                            if fname and fname.lower() != 'nan':
                                self.filename_to_pid[fname] = pid
                                # 雙保險：順便存一份不含副檔名的 key，應對切檔名的極端狀況
                                stem = Path(fname).stem
                                self.filename_to_pid[stem] = pid
            else:
                raise ValueError(
                    f"\n\n[致命錯誤] 無法在 audio_metadata.csv 中建立橋接關係！\n"
                    f"找不到主鍵 {pid_col}。\n"
                    f"實際擁有的欄位: {df_audio.columns.tolist()}"
                )

        # 3. 載入 ICBHI 標籤
        if self.icbhi_dir.exists():
            for txt_path in self.icbhi_dir.glob("*.txt"):
                stem = txt_path.stem
                try:
                    annotations = np.loadtxt(txt_path, ndmin=2)
                    if annotations.size > 0:
                        has_wheeze = 1.0 if np.any(annotations[:, 3] == 1) else 0.0
                        self.pid_registry[stem] = {
                            "wheeze": has_wheeze,
                            "duration": -1.0,
                            "sound_type": 0.0,
                            "domain": 0.0
                        }
                except Exception:
                    pass

    def get_labels(self, filename: str) -> list:
        stem = Path(filename).stem
        
        # 查詢路徑 A: 直接比對 (ICBHI 模式)
        if stem in self.pid_registry:
            labels = self.pid_registry[stem]
            return [labels["wheeze"], labels["duration"], labels["sound_type"], labels["domain"]]
            
        # 查詢路徑 B: 透過橋接表比對 (UK COVID 模式)
        pid = self.filename_to_pid.get(filename) or self.filename_to_pid.get(stem)
        
        # 容錯比對：如果 CSV 裡的檔名被截斷
        if not pid:
            for k, v in self.filename_to_pid.items():
                if k in filename or filename in k:
                    pid = v
                    break
                    
        if pid and pid in self.pid_registry:
            labels = self.pid_registry[pid]
            return [labels["wheeze"], labels["duration"], labels["sound_type"], labels["domain"]]
            
        return [-1.0, -1.0, -1.0, 1.0]