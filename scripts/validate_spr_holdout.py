"""
scripts/validate_spr_holdout.py
SPR holdout (30%) 驗證：納入 SPR 訓練後的「新期末考」。
"""
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

HOLDOUT_DIR = Path('./cache/spr_cache/holdout')
MODEL_PATH = 'asthma_monitor_final.onnx'

def main():
    session = ort.InferenceSession(MODEL_PATH)
    probs, trues = [], []
    for pt in sorted(HOLDOUT_DIR.glob('*.pt')):
        d = torch.load(pt, weights_only=True)
        mel = d['mel'].unsqueeze(0).numpy()
        w_logit, _ = session.run(['wheeze_logit', 'cough_logit'], {'mel': mel})
        probs.append(float(1 / (1 + np.exp(-w_logit[0][0]))))
        trues.append(float(d['wheeze']))

    probs, trues = np.array(probs), np.array(trues)
    print(f"SPR holdout：n={len(probs)}，wheeze={int(trues.sum())}")
    if len(np.unique(trues)) < 2:
        print("[Warning] 只有一個類別")
        return
    auroc = roc_auc_score(trues, probs)
    fpr, tpr, th = roc_curve(trues, probs)
    i = min(int(np.argmax(tpr - fpr)), len(th) - 1)
    pred = (probs >= th[i]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(trues, pred, average='binary', zero_division=0)
    print(f"  AUROC={auroc:.4f} | Prec={p:.4f} Rec={r:.4f} F1={f1:.4f} @thr={th[i]:.3f}")
    print(f"  wheeze mean={probs[trues == 1].mean():.3f}，neg mean={probs[trues == 0].mean():.3f}")

if __name__ == '__main__':
    main()