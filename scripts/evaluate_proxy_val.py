import glob, random
import numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from train_final import AsthmaMonitorNetV2
from config import Config

PROXY_CACHE = './data/UK_COVID19/proxy_val_cache'

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AsthmaMonitorNetV2().to(device)
    model.load_state_dict(torch.load("best_asthma_monitor.pt", map_location=device))
    model.eval()

    # 正樣本：留檔 val 喘鳴 → 手機域合成
    pos = [torch.load(p, weights_only=True)["mel"]
           for p in sorted(glob.glob(PROXY_CACHE + "/*.pt"))]

    # 負樣本：真實 UK、教師分數 < 0.05 的可靠負樣本（排除合成檔）
    random.seed(42)
    uk = sorted(glob.glob(str(Config.CACHE_DIR) + "/*.pt"))
    random.shuffle(uk)
    neg = []
    for p in uk:
        if Path(p).stem.startswith("synth_wheeze"):
            continue
        d = torch.load(p, weights_only=True)
        if float(d["scores"][0]) < 0.05:
            neg.append(d["mel"])
        if len(neg) >= len(pos):
            break

    mels = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    preds = []
    with torch.no_grad():
        for i in range(0, len(mels), 32):
            w, _ = model(torch.stack(mels[i:i + 32]).to(device))
            preds.append(torch.sigmoid(w).cpu().numpy().flatten())

    print(f"pos={len(pos)}, neg={len(neg)}")
    print(f"Proxy Phone-Val AUROC: {roc_auc_score(y, np.concatenate(preds)):.4f}")

if __name__ == "__main__":
    main()