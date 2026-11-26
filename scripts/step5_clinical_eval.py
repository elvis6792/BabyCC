import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import sys
import subprocess
import torch.nn.functional as F
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.mt_cdann import AsthmaCDANN
from torch.utils.data import DataLoader
from collections import defaultdict

class EvalPtDataset(Dataset):
    def __init__(self, pt_files):
        self.pt_files = pt_files

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        mel, target = torch.load(pt_path, weights_only=True)
        return mel, target, str(pt_path)

def eval_collate_fn(batch):
    mels = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    paths = [item[2] for item in batch]

    max_time = max(m.shape[2] for m in mels)
    padded_mels = []
    for m in mels:
        pad_size = max_time - m.shape[2]
        if pad_size > 0:
            m = F.pad(m, (0, pad_size), "constant", 0)
        padded_mels.append(m)

    batch_mels = torch.stack(padded_mels, dim=0)
    batch_targets = torch.stack(targets, dim=0)
    return batch_mels, batch_targets, paths

def interactive_audio_audit(all_paths, y_true, y_scores, clinical_threshold):
    """
    Mac M1 專屬底層音訊稽核工具。
    過濾出 False Positive 與 False Negative 供人類驗證。
    """
    print("\n==========================================================")
    print("進入臨床聽覺稽核模式 (Interactive Audit Mode)")
    print("==========================================================")
    
    audit_list = []
    for path, true_label, score in zip(all_paths, y_true, y_scores):
        pred_label = 1 if score >= clinical_threshold else 0
        if true_label != pred_label:
            error_type = "False Positive (沒病猜有病)" if pred_label == 1 else "False Negative (有病被漏診)"
            wav_path = path.replace(".pt", ".wav")
            audit_list.append((wav_path, error_type, score, true_label))
            
    if not audit_list:
        print("模型沒有任何誤判！")
        return
        
    print(f"[Warning] 發現 {len(audit_list)} 筆誤判樣本。")
    print("操作提示：按 [Enter] 播放下一首，輸入 [q] 結束稽核。")
    
    for wav_path, err_type, score, true_label in audit_list:
        user_input = input(f"\n[{err_type}] 預測機率: {score:.4f} | 實際: {true_label} | 檔案: {Path(wav_path).name} \n> (Enter 播放 / q 離開): ")
        if user_input.lower() == 'q':
            break
            
        if Path(wav_path).exists():
            # 使用 macOS 原生底層播放器，無痛且不阻塞進程
            subprocess.run(["afplay", wav_path])
        else:
            print(f"[Error] 找不到音檔: {wav_path} (請確認 Step0 有正確儲存 .wav)")

def main():
    print("==========================================================")
    print("啟動醫療級模型統計驗證管線 (Clinical Evaluation)")
    print("==========================================================")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = Path(Config.PROJECT_ROOT) / "checkpoints" / f"best_{Config.MODEL_NAME}_cdann.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重: {model_path}")

    print("載入模型...")
    model = AsthmaCDANN(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("準備 Validation Data (嚴格 Group-Level 隔離)...")
    pt_dir = Path(Config.PROJECT_ROOT) / "processed_pt_dataset"
    all_pt_files = list(pt_dir.glob("*.pt"))
    
    import random
    random.seed(42)
    pt_groups = defaultdict(list)
    for pt_file in all_pt_files:
        original_stem = pt_file.name.replace("_clean.pt", "").replace("_wheeze_v1.pt", "")
        pt_groups[original_stem].append(pt_file)
        
    unique_stems = list(pt_groups.keys())
    random.shuffle(unique_stems)
    split_idx = int(len(unique_stems) * 0.8)
    val_stems = unique_stems[split_idx:]
    
    val_files = []
    for stem in val_stems:
        val_files.extend(pt_groups[stem])

    val_dataset = EvalPtDataset(val_files)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_collate_fn
    )

    all_targets = []
    all_probs = []
    all_paths = []

    print("執行大規模推論評估...")
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            inputs, targets, paths = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]
            clinical_logits, _ = model(inputs, alpha=0.0)
            
            probs = torch.sigmoid(clinical_logits[:, 0])
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets[:, 0].cpu().numpy())
            all_paths.extend(paths)

    y_true = np.array(all_targets)
    y_scores = np.array(all_probs)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # 尋找能滿足 95% Sensitivity 的最低門檻值 (臨床排查需求)
    target_sensitivity = 0.95
    idx_95_sens = np.where(tpr >= target_sensitivity)[0][0]
    clinical_threshold = roc_thresholds[idx_95_sens]
    clinical_specificity = 1 - fpr[idx_95_sens]

    y_pred_clinical = (y_scores >= clinical_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_clinical).ravel()

    print("\n==========================================================")
    print("臨床評估報告 (Clinical Metrics Report)")
    print("==========================================================")
    print(f"- AUROC (整體分辨力): {roc_auc:.4f}")
    print(f"- AUPRC (精準召回率): {pr_auc:.4f}")
    print(f"\n以 [Sensitivity >= 95%] 為目標，反推最佳閾值:")
    print(f"- 臨床建議閾值 (Threshold): {clinical_threshold:.4f}")
    print(f"- 敏感度 (Sensitivity/Recall): {tpr[idx_95_sens]:.4f} (漏診率: {fn/(fn+tp):.4%})")
    print(f"- 特異度 (Specificity): {clinical_specificity:.4f} (誤診率: {fp/(fp+tn):.4%})")
    print(f"- 混淆矩陣: TP={tp}, FN={fn}, TN={tn}, FP={fp}")

    output_dir = Path(Config.PROJECT_ROOT) / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([fpr[idx_95_sens]], [tpr[idx_95_sens]], color='red', s=50, label='95% Sensitivity Target')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    save_path = output_dir / "clinical_metrics.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n圖表已儲存至: {save_path}")

    # 在 main() 的最後一行，呼叫稽核工具
    interactive_audio_audit(all_paths, y_true, y_scores, clinical_threshold)

if __name__ == "__main__":
    main()