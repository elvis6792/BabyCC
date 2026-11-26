import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.mt_cdann import AsthmaCDANN

class MobileNetGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=0):
        self.model.eval()
        self.model.zero_grad()
        
        clinical_logits, _ = self.model(x, alpha=0.0)
        
        # 針對指定類別（Wheeze 為 0）進行反向傳播
        target = clinical_logits[0, class_idx]
        target.backward(retain_graph=True)
        
        # 計算特徵通道權重 (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 加權組合並經過 ReLU 濾除負相關特徵
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 正規化並上採樣回原始頻譜尺寸
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach().cpu().numpy(), clinical_logits.detach().cpu().numpy()

def _get_target_layer_for_cam(model: torch.nn.Module) -> torch.nn.Module:
    """
    動態尋找適合掛載 Grad-CAM 的特徵層。
    避開最後破壞空間解析度的 1x1 Point-wise Conv，
    尋找最後一個具備真實感受野 (Kernel Size > 1) 的卷積層 (通常是 3x3 或 5x5 的 Depthwise Conv)。
    """
    target_layer = None
    for module in model.modules():
        # 尋找具備空間維度操作的 Conv2d
        if isinstance(module, torch.nn.Conv2d):
            if module.kernel_size[0] > 1:
                target_layer = module
            
    if target_layer is None:
        raise ValueError("[致命錯誤] 模型中找不到任何 Kernel Size > 1 的 Conv2d 層，無法掛載高解析度 Grad-CAM。")
    return target_layer

def main():
    print("==========================================================")
    print("啟動 XAI 模型可解釋性分析 (高解析度 Grad-CAM)")
    print("==========================================================")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = Path(Config.PROJECT_ROOT) / "checkpoints" / f"best_{Config.MODEL_NAME}_cdann.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重: {model_path}")

    model = AsthmaCDANN(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    target_layer = _get_target_layer_for_cam(model)
    cam_extractor = MobileNetGradCAM(model, target_layer)

    pt_dir = Path(Config.PROJECT_ROOT) / "processed_pt_dataset"
    all_pt_files = list(pt_dir.glob("*_wheeze_v1.pt"))
    if not all_pt_files:
        raise ValueError("找不到包含 wheeze 標籤的 .pt 檔案")
        
    sample_path = random.choice(all_pt_files)
    data = torch.load(sample_path, map_location=device, weights_only=True)
    
    if isinstance(data, (tuple, list)):
        mel_spec = data[0]
        labels = data[1]
    else:
        raise TypeError(f"未知的資料格式: {type(data)}")
        
    input_tensor = mel_spec.unsqueeze(0).to(device)
    
    print(f"分析樣本: {sample_path.name}")
    print(f"真實標籤: {labels.cpu().numpy()}")
    
    heatmap, logits = cam_extractor(input_tensor, class_idx=0)
    pred_prob = torch.sigmoid(torch.tensor(logits[0, 0])).item()
    print(f"模型預測機率 (Wheeze): {pred_prob:.4f}")

    mel_numpy = input_tensor.squeeze().cpu().numpy()
    if mel_numpy.ndim == 3:
        mel_numpy = mel_numpy[0] 

    output_dir = Path(Config.PROJECT_ROOT) / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(mel_numpy, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Original Log-Mel Spectrogram")
    plt.ylabel("Mel Filterbank")
    plt.xlabel("Time Frames")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mel_numpy, aspect='auto', origin='lower', cmap='gray')
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM (Wheeze Prob: {pred_prob:.2f})")
    plt.xlabel("Time Frames")
    
    save_path = output_dir / f"gradcam_{sample_path.stem}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"XAI 分析完成，熱力圖已儲存至: {save_path}")

if __name__ == "__main__":
    main()