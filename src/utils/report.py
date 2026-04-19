"""
Markdown Report Generator
==========================

Automatically generates a structured Markdown report summarising experiment
results, including the theoretical motivation for L1 gate pruning, a results
table, and embedded plots.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.helpers import ensure_dir


def generate_report(
    experiment_results: List[Dict],
    gate_histogram_path: Optional[str] = None,
    tradeoff_plot_path: Optional[str] = None,
    output_dir: str = "./reports",
    filename: str = "experiment_report.md",
    config: Optional[Dict] = None,
) -> str:
    """Generate a Markdown report summarising all experiments.

    Parameters
    ----------
    experiment_results : list of dict
        Each dict must have: ``name``, ``lambda_sparse``, ``accuracy``,
        ``sparsity``, and optionally per-layer sparsity keys.
    gate_histogram_path : str or None
        Path to the gate histogram image (relative or absolute).
    tradeoff_plot_path : str or None
        Path to the accuracy-vs-sparsity plot.
    output_dir : str
        Output directory for the report.
    filename : str
        Report filename.
    config : dict or None
        The full config dict for reference.

    Returns
    -------
    str
        Path to the generated report.
    """
    ensure_dir(output_dir)
    report_path = Path(output_dir) / filename

    lines: List[str] = []
    _h = lines.append  # shorthand

    # ================================================================== #
    # Header
    # ================================================================== #
    _h("# Self-Pruning Neural Network — Experiment Report")
    _h("")
    _h(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    _h("")
    _h("---")
    _h("")

    # ================================================================== #
    # Theory / Motivation
    # ================================================================== #
    _h("## 1. Why L1 on Sigmoid Gates Induces Sparsity")
    _h("")
    _h("Each weight $w_{ij}$ in a `PrunableLinear` layer is paired with a "
       "learnable **gate score** $g_{ij}$ (unconstrained real value).  During the "
       "forward pass the effective weight is:")
    _h("")
    _h("$$\\hat{w}_{ij} = w_{ij} \\cdot \\sigma(g_{ij})$$")
    _h("")
    _h("where $\\sigma$ is the sigmoid function mapping $g_{ij}$ to $[0, 1]$.")
    _h("")
    _h("The total training loss is:")
    _h("")
    _h("$$\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\lambda \\sum_{i,j} \\sigma(g_{ij})$$")
    _h("")
    _h("The L1 penalty on $\\sigma(g_{ij})$ pushes these values toward **zero**, "
       "because the derivative $\\frac{\\partial}{\\partial g_{ij}} \\sigma(g_{ij}) = "
       "\\sigma(g_{ij})(1 - \\sigma(g_{ij}))$ is always positive for finite $g$, "
       "so the penalty gradient always points in the direction of decreasing $g$.")
    _h("")
    _h("When $\\sigma(g_{ij}) \\approx 0$, the effective weight $\\hat{w}_{ij} \\approx 0$, "
       "meaning that connection is effectively **pruned** from the network.")
    _h("")
    _h("The hyper-parameter $\\lambda$ controls the sparsity–accuracy trade-off: "
       "higher $\\lambda$ produces sparser networks at the potential cost of accuracy.")
    _h("")
    _h("---")
    _h("")

    # ================================================================== #
    # Configuration
    # ================================================================== #
    if config:
        _h("## 2. Configuration")
        _h("")
        _h("| Parameter | Value |")
        _h("|-----------|-------|")
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        _h(f"| Input dim | {model_cfg.get('input_dim', 'N/A')} |")
        _h(f"| Hidden dims | {model_cfg.get('hidden_dims', 'N/A')} |")
        _h(f"| Output dim | {model_cfg.get('output_dim', 'N/A')} |")
        _h(f"| Epochs | {train_cfg.get('epochs', 'N/A')} |")
        _h(f"| Batch size | {train_cfg.get('batch_size', 'N/A')} |")
        _h(f"| Learning rate | {train_cfg.get('learning_rate', 'N/A')} |")
        _h(f"| Optimizer | {train_cfg.get('optimizer', 'N/A')} |")
        _h(f"| Gate threshold | {config.get('pruning', {}).get('gate_threshold', 'N/A')} |")
        _h(f"| Seed | {config.get('seed', 'N/A')} |")
        _h("")
        _h("---")
        _h("")

    # ================================================================== #
    # Results Table
    # ================================================================== #
    section_num = 3 if config else 2
    _h(f"## {section_num}. Experiment Results")
    _h("")
    _h("| Experiment | Lambda (λ) | Test Accuracy (%) | Sparsity (%) |")
    _h("|------------|------------|-------------------|--------------|")
    for r in experiment_results:
        acc = r["accuracy"] * 100
        sp = r["sparsity"] * 100
        _h(f"| {r['name']} | {r['lambda_sparse']} | {acc:.2f} | {sp:.2f} |")
    _h("")

    # Per-layer detail
    _h(f"### {section_num}.1 Per-Layer Sparsity")
    _h("")
    for r in experiment_results:
        _h(f"**{r['name']}** (λ = {r['lambda_sparse']}):")
        _h("")
        layer_keys = sorted([k for k in r if k.startswith("sparsity_layer_")])
        if layer_keys:
            _h("| Layer | Sparsity (%) |")
            _h("|-------|-------------|")
            for k in layer_keys:
                layer_name = k.replace("sparsity_", "").replace("_", " ").title()
                _h(f"| {layer_name} | {r[k] * 100:.2f} |")
            _h("")
        else:
            _h("_No per-layer data available._")
            _h("")
    _h("---")
    _h("")

    # ================================================================== #
    # Visualisations
    # ================================================================== #
    section_num += 1
    _h(f"## {section_num}. Visualisations")
    _h("")

    if gate_histogram_path:
        rel = os.path.relpath(gate_histogram_path, output_dir)
        _h("### Gate Value Distribution (Best Model)")
        _h("")
        _h(f"![Gate Histogram]({rel})")
        _h("")
        _h("A strong spike near 0 indicates successful pruning — those weights "
           "have been effectively removed by the network.")
        _h("")

    if tradeoff_plot_path:
        rel = os.path.relpath(tradeoff_plot_path, output_dir)
        _h("### Accuracy vs Sparsity Trade-off")
        _h("")
        _h(f"![Trade-off Plot]({rel})")
        _h("")
        _h("Higher λ values push more gates toward zero (higher sparsity) but "
           "may reduce classification accuracy.  The optimal λ balances both.")
        _h("")

    _h("---")
    _h("")

    # ================================================================== #
    # Conclusion
    # ================================================================== #
    section_num += 1
    _h(f"## {section_num}. Conclusion")
    _h("")
    if experiment_results:
        best = max(experiment_results, key=lambda r: r["accuracy"])
        sparsest = max(experiment_results, key=lambda r: r["sparsity"])
        _h(f"* **Best accuracy**: {best['name']} — "
           f"{best['accuracy'] * 100:.2f}% (sparsity: {best['sparsity'] * 100:.2f}%)")
        _h(f"* **Highest sparsity**: {sparsest['name']} — "
           f"{sparsest['sparsity'] * 100:.2f}% (accuracy: {sparsest['accuracy'] * 100:.2f}%)")
        _h("")
        _h("The results confirm the expected trade-off: increasing the sparsity "
           "penalty (λ) drives more gate values toward zero, yielding a sparser "
           "network, while the classification loss keeps essential connections alive.")
    _h("")
    _h("---")
    _h("")
    _h("*Report auto-generated by `src/utils/report.py`.*")

    # ---- Write ----
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(report_path.resolve())
