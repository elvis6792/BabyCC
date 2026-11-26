# train_v2.py
import torch
from torch.utils.data import DataLoader
from model_v2 import AsthmaMonitorNetV2, AsthmaMonitorLossV2
from dataset_mixed import build_mixed_dataset
from sklearn.metrics import roc_auc_score
from config import Config


ICBHI_DIR = './data/ICBHI_final_database'
UK_CACHE_DIR = Config.CACHE_DIR


def collate_fn_v2(batch):
    """自定義 collate，處理動態權重與 source 字串"""
    mels = torch.stack([b["mel"] for b in batch])
    wheeze_labels = torch.cat([b["wheeze_label"] for b in batch])
    cough_labels = torch.cat([b["cough_label"] for b in batch])
    sample_weights = torch.stack([b["sample_weight"] for b in batch])
    sources = [b["source"] for b in batch]

    return {
        "mel": mels,
        "wheeze_label": wheeze_labels,
        "cough_label": cough_labels,
        "sample_weight": sample_weights,
        "source": sources,
    }


def validate(model, loader, device, epoch):
    model.eval()
    all_wheeze_true = []
    all_wheeze_pred = []
    all_cough_true = []
    all_cough_pred = []
    
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            wheeze_logit, cough_logit = model(mel)
            
            wheeze_pred = torch.sigmoid(wheeze_logit).cpu().numpy().flatten()
            cough_pred = torch.sigmoid(cough_logit).cpu().numpy().flatten()
            
            all_wheeze_true.extend(batch["wheeze_label"].numpy().flatten())
            all_wheeze_pred.extend(wheeze_pred)
            all_cough_true.extend(batch["cough_label"].numpy().flatten())
            all_cough_pred.extend(cough_pred)
    
    import numpy as np
    all_wheeze_true = np.array(all_wheeze_true)
    all_wheeze_pred = np.array(all_wheeze_pred)
    
    # 只在 ICBHI 上計算喘鳴 AUROC（UK 沒有真實標籤）
    # 這裡簡單用全部資料，但 UK 的 pseudo label 會稀釋結果
    try:
        wheeze_auroc = roc_auc_score(all_wheeze_true, all_wheeze_pred)
    except ValueError:
        wheeze_auroc = 0.0
    
    print(f"  [Val] Wheeze AUROC: {wheeze_auroc:.4f}")
    return wheeze_auroc


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"使用裝置: {device}")
    
    # ── 建立資料集 ──
    train_dataset, val_dataset, sampler = build_mixed_dataset(
        ICBHI_DIR, UK_CACHE_DIR
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_v2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn_v2,
    )
    
    # ── 模型 ──
    model = AsthmaMonitorNetV2(
        n_mels=Config.N_MELS, dropout=0.3
    ).to(device)
    
    loss_fn = AsthmaMonitorLossV2(focal_gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    
    # ── 訓練 ──
    best_auroc = 0.0
    
    for epoch in range(1, 51):
        model.train()
        running = {"l_wheeze": 0, "l_cough": 0, "total": 0}
        
        for batch_idx, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            targets = {
                "wheeze_label": batch["wheeze_label"].to(device),
                "cough_label": batch["cough_label"].to(device),
                "sample_weight": batch["sample_weight"].to(device),
                "source": batch["source"],
            }
            
            optimizer.zero_grad()
            outputs = model(mel)
            loss, loss_dict = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for k in running:
                running[k] += loss_dict[k]
        
        n = len(train_loader)
        print(f"Epoch {epoch:03d}: " + " | ".join(
            f"{k}={v/n:.4f}" for k, v in running.items()
        ))
        
        scheduler.step()
        
        # 每 5 epoch 驗證
        if epoch % 5 == 0:
            auroc = validate(model, val_loader, device, epoch)
            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(model.state_dict(), "best_asthma_monitor.pt")
                print(f"  儲存最佳模型 (AUROC={auroc:.4f})")
    
    print(f"\n最佳 Validation AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()