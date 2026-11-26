import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from models.hpss_multitask_network import HPSSMultiTaskEdgeNet

def main():
    print("==========================================================")
    print("啟動 3D 氣喘專用模型 ONNX 導出與驗證管線")
    print("==========================================================")

    checkpoint_path = Path(Config.CHECKPOINT_DIR) / f"best_{Config.MODEL_NAME}_v2.pt"
    onnx_output_path = Path(Config.CHECKPOINT_DIR) / f"{Config.MODEL_NAME}_3d_production.onnx"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型權重檔: {checkpoint_path}")

    device = torch.device("cpu")
    model = HPSSMultiTaskEdgeNet(
        backbone_name=Config.MODEL_NAME,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES
    )
    
    print("載入 PyTorch 權重...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, Config.IN_CHANNELS, Config.N_MELS, Config.TARGET_FRAMES, dtype=torch.float32)

    print(f"執行 ONNX 導出 (Opset: {Config.ONNX_OPSET})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=Config.ONNX_OPSET,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'time_frames'},
            'output': {0: 'batch_size'}
        }
    )

    print("\n驗證 ONNX 模型完整性...")
    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX 結構檢查通過。")

    print("\n執行 PyTorch vs ONNX 精度比對...")
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(str(onnx_output_path), providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"最大絕對誤差 (Max Absolute Error): {max_diff:.8f}")

    if max_diff < 1e-4:
        print(f"[PASS] 精度驗證通過！ONNX 模型已準備就緒，可投入 Production 環境。")
        print(f"導出路徑: {onnx_output_path}")
    else:
        print("[FAIL] 精度驗證失敗：誤差超出安全範圍。")

if __name__ == "__main__":
    main()