"""
scripts/infer_real_world.py
真實世界驗收腳本：用 ONNX 模型預測真實錄音。
"""
import argparse
import shutil
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from step1_extract_features import process_and_save
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model", type=str, default="asthma_monitor_final.onnx")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[Error] 找不到檔案：{file_path}")
        return

    tmp_cache = Path("./tmp_real_world_cache")
    tmp_cache.mkdir(exist_ok=True)
    for f in tmp_cache.glob("*"):
        f.unlink()

    name_lower = file_path.name.lower()
    if ('cough' in name_lower or 'exhale' in name_lower
            or 'exhalation' in name_lower):
        work_path = file_path
        copied = None
    else:
        copied = tmp_cache / f"real_cough_{file_path.name}"
        shutil.copy(str(file_path), str(copied))
        work_path = copied
        print(f"檔名無類型關鍵字，以臨時檔名提取特徵：{copied.name}")

    print(f"處理音檔：{file_path.name}")
    ok = process_and_save(str(work_path), str(tmp_cache))

    if copied is not None:
        copied.unlink()

    if not ok:
        print("[Error] 特徵提取失敗（可能被 SNR 或品質閘門擋掉）。")
        return

    pt_files = list(tmp_cache.glob("*.pt"))
    if not pt_files:
        print("[Error] 沒有產生 .pt 快取。")
        return

    data = torch.load(pt_files[0], weights_only=True)
    mel = data["mel"].unsqueeze(0).numpy()

    print(f"載入模型：{args.model}")
    session = ort.InferenceSession(args.model)
    outputs = session.run(["wheeze_logit", "cough_logit"], {"mel": mel})

    wheeze_prob = 1.0 / (1.0 + np.exp(-outputs[0][0][0]))
    cough_prob = 1.0 / (1.0 + np.exp(-outputs[1][0][0]))

    print("\n" + "=" * 40)
    print(f"預測結果：{file_path.name}")
    print("=" * 40)
    print(f"喘鳴機率 (Wheeze) : {wheeze_prob:.4f}  "
          f"{'[ALARM] 偵測到喘鳴！' if wheeze_prob > 0.5 else '[NORMAL] 正常'}")
    print(f"咳嗽機率 (Cough)  : {cough_prob:.4f}")
    print("=" * 40)

    for f in tmp_cache.glob("*"):
        f.unlink()


if __name__ == "__main__":
    main()