import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import math
import random
import sys
from collections import defaultdict
import torchaudio.transforms as T

# 確保能讀取到 config 與 models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from models.mt_cdann import AsthmaCDANN, ConditionalMultiTaskLoss

class CoughPtDataset(Dataset):
    def __init__(self, pt_files, is_train=False):
        self.pt_files = pt_files
        self.is_train = is_train
        
        if self.is_train:
            # 頻譜為 128 維，最多遮蔽 20 個連續頻帶
            self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
            # 時間軸為動態，最多遮蔽 40 個連續時間幀
            self.time_mask = T.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        mel, target = torch.load(pt_path, weights_only=True)
        
        # 確保維度是 [1, Mels, Time]
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
            
        if self.is_train:
            # 1. 隨機頻率與時間遮蔽 (打馬賽克)
            if torch.rand(1).item() > 0.3:
                mel = self.freq_mask(mel)
            if torch.rand(1).item() > 0.3:
                mel = self.time_mask(mel)
                
            # 2. 頻譜域微量白噪音 (灑沙子)，抹除合成器留下的能量與相位指紋
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(mel) * 1.5
                mel = mel + noise
                
        return mel, target

def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)

    mels = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 檢查通道與 Mel 數量是否一致 (若不一致，代表混到了舊架構的資料)
    ref_shape = mels[0].shape
    for i, m in enumerate(mels):
        if m.shape[0] != ref_shape[0] or m.shape[1] != ref_shape[1]:
            raise ValueError(f"形狀嚴重不符！第一筆為 {ref_shape}，但第 {i} 筆為 {m.shape}。你的資料夾裡混到了舊檔案！")

    # 找尋這個 Batch 裡最大的時間維度
    max_time = max(m.shape[2] for m in mels)
    
    # 將所有 Mel 頻譜補零到最大時間維度
    padded_mels = []
    for m in mels:
        pad_size = max_time - m.shape[2]
        if pad_size > 0:
            m = F.pad(m, (0, pad_size), "constant", 0)
        padded_mels.append(m)

    batch_mels = torch.stack(padded_mels, dim=0)
    batch_targets = torch.stack(targets, dim=0)
    
    return batch_mels, batch_targets

def train_model(train_loader, val_loader, num_epochs=30, device='mps'):
    print("==========================================================")
    print("啟動 CDANN 對抗網路 PyTorch 訓練管線 (MPS / CUDA)")
    print("==========================================================")
    print(f"使用硬體加速設備: {device}")

    model = AsthmaCDANN(pretrained=True).to(device)
    criterion = ConditionalMultiTaskLoss(ignore_index=-1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    total_steps = num_epochs * len(train_loader)
    current_step = 0

    best_val_loss = float('inf')
    best_model_path = Path(Config.PROJECT_ROOT) / "checkpoints" / f"best_{Config.MODEL_NAME}_cdann.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_wheeze = 0.0
        train_loss_domain = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for batch_data in train_loader:                
                inputs, targets = batch_data[0], batch_data[1]
                    
                if not isinstance(inputs, torch.Tensor):
                    raise TypeError("致命錯誤：輸入不是 Tensor。")
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Alpha 退火 (GRL 權重)
                p = current_step / total_steps
                alpha = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0
                current_step += 1
                
                optimizer.zero_grad()
                
                clinical_logits, domain_logits = model(inputs, alpha=alpha)
                domain_targets = targets[:, 3]
                
                loss, l_wheeze, l_sound, l_domain = criterion(
                    clinical_logits, domain_logits, targets, domain_targets
                )
                
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item()
                train_loss_wheeze += l_wheeze.item()
                train_loss_domain += l_domain.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update(1)
                
        # Validation Loop
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for batch_data in val_loader:
                    inputs, targets = batch_data[0].to(device), batch_data[1].to(device)
                    domain_targets = targets[:, 3]
                    
                    clinical_logits, domain_logits = model(inputs, alpha=0.0)
                    
                    loss, _, _, _ = criterion(
                        clinical_logits, domain_logits, targets, domain_targets
                    )
                    val_loss_total += loss.item()
                    pbar.update(1)
                    
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} 結算:")
        print(f"  Train -> Total: {avg_train_loss:.4f} | Wheeze: {train_loss_wheeze/len(train_loader):.4f} | Domain: {train_loss_domain/len(train_loader):.4f} | Alpha: {alpha:.4f}")
        print(f"  Val   -> Total: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, best_model_path)
            print(f"發現新低 Val Loss，模型已儲存至 {best_model_path}")

def main():
    pt_dir = Path(Config.PROJECT_ROOT) / "processed_pt_dataset"
    if not pt_dir.exists():
        raise FileNotFoundError(f"找不到特徵目錄: {pt_dir}")

    all_pt_files = list(pt_dir.glob("*.pt"))
    if not all_pt_files:
        raise ValueError(f"目錄 {pt_dir} 中沒有找到任何 .pt 檔案")

    pt_groups = defaultdict(list)
    for pt_file in all_pt_files:
        original_stem = pt_file.name.replace("_clean.pt", "").replace("_wheeze_v1.pt", "")
        pt_groups[original_stem].append(pt_file)
        
    unique_stems = list(pt_groups.keys())
    random.seed(42)
    random.shuffle(unique_stems)
    
    split_idx = int(len(unique_stems) * 0.8)
    train_stems = unique_stems[:split_idx]
    val_stems = unique_stems[split_idx:]
    
    train_files = []
    for stem in train_stems:
        train_files.extend(pt_groups[stem])
        
    val_files = []
    for stem in val_stems:
        val_files.extend(pt_groups[stem])

    # 必須標明 is_train=True 啟動強干擾
    train_dataset = CoughPtDataset(train_files, is_train=True)
    val_dataset = CoughPtDataset(val_files, is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        collate_fn=safe_collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        collate_fn=safe_collate_fn
    )

    print(f"資料集準備完成: Train={len(train_dataset)} 筆, Val={len(val_dataset)} 筆")
    
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.EPOCHS,
        device=Config.DEVICE
    )

if __name__ == "__main__":
    main()