"""
Sparsity-Aware Loss Function
==============================

Combines standard cross-entropy classification loss with an L1 penalty on the
gate values of all ``PrunableLinear`` layers.

    Total Loss = CE(logits, targets) + λ · Σ sigmoid(gate_scores)

**Why this works:**
* ``sigmoid(gate_scores)`` maps each gate to [0, 1].
* The L1 sum penalises gates that are far from 0.
* During training the optimizer pushes ``gate_scores`` toward large negative
  values (making ``sigmoid → 0``), which effectively zeros out the
  corresponding weights.
* The classification loss counter-balances: gates critical to accuracy are
  kept open while redundant ones are driven to zero.
* The trade-off is controlled by ``lambda_sparse`` (λ).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.prunable_net import SelfPruningNetwork


class SparsityAwareLoss(nn.Module):
    """Cross-entropy + L1 gate penalty.

    Parameters
    ----------
    lambda_sparse : float
        Weight of the sparsity penalty term.
    """

    def __init__(self, lambda_sparse: float = 1e-4) -> None:
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: SelfPruningNetwork,
    ) -> dict:
        """Compute combined loss.

        Returns
        -------
        dict
            ``"total"``: combined scalar loss (for ``.backward()``).
            ``"classification"``: CE loss value.
            ``"sparsity"``: raw L1 gate sum.
            ``"sparsity_weighted"``: λ × L1 gate sum.
        """
        cls_loss = self.ce_loss(logits, targets)
        gate_l1 = model.get_total_gate_l1()
        sparsity_loss = self.lambda_sparse * gate_l1
        total_loss = cls_loss + sparsity_loss

        return {
            "total": total_loss,
            "classification": cls_loss.detach(),
            "sparsity": gate_l1.detach(),
            "sparsity_weighted": sparsity_loss.detach(),
        }
