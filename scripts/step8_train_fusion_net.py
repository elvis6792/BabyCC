import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.fusion_net import AsthmaDiagnosticFusionNet

class MultimodalDataset(Dataset):
    def __init__(self, data_dict):
        self.data_list = []
        num_positives = 0
        
        for pid, data in data_dict.items():
            t_feat = data['tabular'].squeeze(0) if data['tabular'].ndim > 1 else data['tabular']
            a_feat = data['acoustic']
            label = data['label']
            
            self.data_list.append((t_feat, a_feat, label))
            if label.item() == 1.0:
                num_positives += 1
                
        self.num_positives = num_positives
        self.num_negatives = len(self.data_list) - num_positives

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class FocalLoss(nn.Module):
    """
    專治極度不平衡與困難樣本的 Focal Loss (RetinaNet 提出)。
    alpha 處理正負樣本不平衡，gamma 處理難易樣本不平衡。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

def main():
    print("==========================================================")
    print("啟動多模態融合網路訓練管線 (Train Fusion Net)")
    print("==========================================================")

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    data_dir = Path(Config.PROJECT_ROOT) / "processed_fusion_dataset"
    train_path = data_dir / "fused_train.pt"
    val_path = data_dir / "fused_val.pt"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("找不到快取特徵，請先執行 step7。")

    print("載入離線特徵快取並攤平記憶體...")
    train_dict = torch.load(train_path, weights_only=True)
    val_dict = torch.load(val_path, weights_only=True)
    
    train_dataset = MultimodalDataset(train_dict)
    val_dataset = MultimodalDataset(val_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    tabular_dim = train_dataset[0][0].shape[0]
    acoustic_dim = train_dataset[0][1].shape[0]
    
    print(f"- Tabular Dim: {tabular_dim} | Acoustic Dim: {acoustic_dim}")
    print(f"- 訓練集比例 (Pos/Neg): {train_dataset.num_positives}/{train_dataset.num_negatives}")

    model = AsthmaDiagnosticFusionNet(tabular_dim=tabular_dim, acoustic_dim=acoustic_dim).to(device)
    
    # alpha 設為正樣本的期望佔比 (約 0.88 給負樣本，0.12 給正樣本的對稱調平)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    epochs = 30
    best_auroc = 0.0
    checkpoint_dir = Path(Config.PROJECT_ROOT) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / "best_fusion_net.pt"

    print("\n開始訓練...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for t_feat, a_feat, label in train_loader:
            t_feat, a_feat, label = t_feat.to(device), a_feat.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(t_feat, a_feat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for t_feat, a_feat, label in val_loader:
                t_feat, a_feat, label = t_feat.to(device), a_feat.to(device), label.to(device)
                
                logits = model(t_feat, a_feat)
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                
        val_auroc = roc_auc_score(all_labels, all_preds)
        val_auprc = average_precision_score(all_labels, all_preds)
        
        scheduler.step(val_auroc)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), save_path)

    print("\n訓練結束！")
    print(f"最佳 Val AUROC: {best_auroc:.4f}")
    print(f"模型已儲存至: {save_path}")

if __name__ == "__main__":
    main()