# loss_v2.py
"""
弱監督損失函數 v2。
結合 ICBHI 真實標籤 + UK 氣喘先驗加權 pseudo label。
"""
import torch
import torch.nn as nn


class WeaklySupervisedLoss(nn.Module):
    """
    Loss = Σ_i w_i · FocalBCE(wheeze_logit_i, wheeze_label_i)
         + Σ_i FocalBCE(cough_logit_i, cough_label_i)

    其中 w_i 是動態權重：
    - ICBHI 樣本：w_i = 1.0（真實標籤）
    - UK 樣本：w_i = sample_weight（由氣喘先驗決定）
    """

    def __init__(self, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_gamma = focal_gamma

    def focal_bce_elementwise(self, logits, targets, gamma=None):
        """逐元素 Focal Loss（不做 mean，讓 caller 加權）"""
        if gamma is None:
            gamma = self.focal_gamma
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        return focal_weight * bce

    def forward(self, outputs, targets):
        """
        outputs: (wheeze_logit, cough_logit)
        targets: dict with:
            'wheeze_label': [B, 1]
            'cough_label': [B, 1]
            'sample_weight': [B]
            'source': list of str
        """
        wheeze_logit, cough_logit = outputs

        # ── Wheeze Loss（加權）──
        wheeze_focal = self.focal_bce_elementwise(
            wheeze_logit, targets["wheeze_label"]
        )  # [B, 1]

        # 組合權重：ICBHI = 1.0, UK = sample_weight
        batch_size = wheeze_focal.shape[0]
        final_weights = torch.ones(batch_size, 1, device=wheeze_focal.device)

        for i, src in enumerate(targets["source"]):
            if src == "uk":
                final_weights[i] = targets["sample_weight"][i]
            else:
                final_weights[i] = 1.0  # ICBHI 真實標籤

        l_wheeze = (wheeze_focal * final_weights).mean()

        # ── Cough Loss（只有 UK 有咳嗽標籤）──
        cough_focal = self.focal_bce_elementwise(
            cough_logit, targets["cough_label"]
        )

        cough_mask = torch.zeros(batch_size, 1, device=cough_focal.device)
        for i, src in enumerate(targets["source"]):
            if src == "uk":
                cough_mask[i] = 1.0

        l_cough = (cough_focal * cough_mask).mean()

        total = l_wheeze + l_cough

        return total, {
            "l_wheeze": l_wheeze.item(),
            "l_cough": l_cough.item(),
            "total": total.item(),
        }