"""
Evaluation Metrics
===================

Provides functions to assess a trained self-pruning network:

* **Accuracy** (top-1) on any data loader.
* **Sparsity** — percentage of gate values below a configurable threshold.
* **evaluate_model** — convenience wrapper that returns both.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.models.prunable_net import SelfPruningNetwork


@torch.no_grad()
def compute_accuracy(
    model: SelfPruningNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Top-1 accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_sparsity(
    model: SelfPruningNetwork,
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """Per-layer and overall sparsity (fraction of gates < threshold).

    Returns
    -------
    dict
        ``"overall"`` : float — network-wide sparsity.
        ``"layer_<i>"`` : float — sparsity for i-th PrunableLinear.
    """
    from src.models.prunable_net import PrunableLinear

    result: Dict[str, float] = {}
    total_pruned = 0
    total_params = 0
    layer_idx = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            sp = module.get_sparsity(threshold)
            n = module.weight.numel()
            result[f"layer_{layer_idx}"] = sp
            total_pruned += sp * n
            total_params += n
            layer_idx += 1

    result["overall"] = total_pruned / total_params if total_params > 0 else 0.0
    return result


def evaluate_model(
    model: SelfPruningNetwork,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """Compute accuracy + sparsity in one call.

    Returns
    -------
    dict
        ``"accuracy"`` : float
        ``"sparsity_overall"`` : float
        ``"sparsity_layer_<i>"`` : float per layer
    """
    accuracy = compute_accuracy(model, test_loader, device)
    sparsity = compute_sparsity(model, threshold)

    metrics = {"accuracy": accuracy}
    for key, val in sparsity.items():
        metrics[f"sparsity_{key}"] = val
    return metrics
