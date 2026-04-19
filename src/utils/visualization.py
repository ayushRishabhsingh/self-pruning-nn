"""
Visualisation Utilities
========================

Plotting functions for gate-value histograms and accuracy–sparsity trade-off
curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.prunable_net import SelfPruningNetwork, PrunableLinear


def plot_gate_histogram(
    model: SelfPruningNetwork,
    save_path: str,
    title: str = "Distribution of Gate Values",
    bins: int = 100,
) -> str:
    """Plot a histogram of all gate values across the network.

    A successful pruning run shows a strong spike near 0 (pruned weights)
    and possibly a smaller peak near 1 (retained weights).

    Parameters
    ----------
    model : SelfPruningNetwork
        Trained model.
    save_path : str
        File path to save the figure (e.g. ``reports/gate_hist.png``).
    title : str
        Plot title.
    bins : int
        Number of histogram bins.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    all_gates: List[np.ndarray] = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(module.get_gate_values().numpy().flatten())

    gate_values = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gate_values, bins=bins, color="#6366f1", edgecolor="#312e81", alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axvline(x=0.01, color="#ef4444", linestyle="--", linewidth=1.5, label="Pruning threshold (0.01)")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(Path(save_path).resolve())


def plot_accuracy_sparsity_tradeoff(
    results: List[Dict],
    save_path: str,
    title: str = "Accuracy vs Sparsity Trade-off",
) -> str:
    """Bar chart comparing accuracy and sparsity across lambda settings.

    Parameters
    ----------
    results : list of dict
        Each dict must contain ``"name"``, ``"accuracy"``, ``"sparsity"``,
        and ``"lambda_sparse"``.
    save_path : str
        Output file path.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(x - width / 2, accuracies, width, label="Accuracy (%)",
                    color="#6366f1", edgecolor="#312e81", alpha=0.85)
    ax1.set_ylabel("Accuracy (%)", fontsize=13, color="#6366f1")
    ax1.tick_params(axis="y", labelcolor="#6366f1")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, sparsities, width, label="Sparsity (%)",
                    color="#f59e0b", edgecolor="#92400e", alpha=0.85)
    ax2.set_ylabel("Sparsity (%)", fontsize=13, color="#f59e0b")
    ax2.tick_params(axis="y", labelcolor="#f59e0b")
    ax2.set_ylim(0, 100)

    ax1.set_xlabel("Experiment (λ)", fontsize=13)
    ax1.set_xticks(x)
    lambda_labels = [f"{r['name']}\n(λ={r['lambda_sparse']})" for r in results]
    ax1.set_xticklabels(lambda_labels, fontsize=10)
    ax1.set_title(title, fontsize=15, fontweight="bold")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=11)

    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(Path(save_path).resolve())
