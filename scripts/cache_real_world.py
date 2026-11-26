"""
scripts/cache_real_world.py (v2)
自動掃描真實錄音目錄，批量建特徵快取。
命名約定（RealWorldDataset 依關鍵字讀標籤）：
  real_cough_XX        → wheeze=0, cough=1
  real_cough_wheeze_XX → wheeze=1, cough=1
  real_exhale_XX       → wheeze=0, cough=0
  triple（連咳）是驗收治具，放進來也會被自動跳過
"""
import glob
from pathlib import Path
from step1_extract_features import process_and_save

REAL_DIR = './data/real_world_recordings'
REAL_CACHE = './cache/real_world_cache'


def main():
    Path(REAL_CACHE).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(REAL_DIR + '/*.wav') + glob.glob(REAL_DIR + '/*.mp3'))
    if not files:
        print(f"[Error] 在 {REAL_DIR} 找不到 wav/mp3 錄音")
        return

    ok_count = 0
    for f in files:
        name = Path(f).name.lower()
        if 'triple' in name:
            print(f"[Skip] 跳過連咳治具：{name}")
            continue
        ok = process_and_save(f, REAL_CACHE)
        print(f"{name}  {'[OK]' if ok else '[FAIL] 被品質閘門擋掉（SNR/peak）'}")
        ok_count += 1 if ok else 0

    print(f"\n[Done] 真實錄音快取完成：{ok_count}/{len(files)} -> {REAL_CACHE}")


if __name__ == '__main__':
    main()