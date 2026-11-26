import sys
import io
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from scripts.inference_engine import AsthmaInferencePipeline

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/test_inference_direct.py <wav檔案路徑>")
        sys.exit(1)

    wav_path = Path(sys.argv[1]).resolve()
    if not wav_path.exists():
        print(f"[Error] 找不到檔案: {wav_path}")
        sys.exit(1)

    model_path = Config.CHECKPOINT_DIR / f"{Config.MODEL_NAME}_3d_production.onnx"
    if not model_path.exists():
        print(f"[Error] 找不到 ONNX 模型: {model_path}")
        sys.exit(1)

    print(f"載入推論引擎: {model_path.name}")
    engine = AsthmaInferencePipeline(str(model_path))

    print(f"讀取並預處理音檔: {wav_path.name}")
    try:
        y, sr = sf.read(str(wav_path))
        
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        if sr != Config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
            sr = Config.SAMPLE_RATE
            
        target_samples = Config.TARGET_SAMPLES
        current_samples = len(y)
        
        if current_samples < target_samples:
            y = np.pad(y, (0, target_samples - current_samples), mode='constant')
        elif current_samples > target_samples:
            y = y[:target_samples]
            
        y = y.astype(np.float32)
        
        print("執行推論...")
        result = engine.predict(y, sr)
        
        print("\n[OK] 推論成功，回傳結果：")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n[Error] 推論過程發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()