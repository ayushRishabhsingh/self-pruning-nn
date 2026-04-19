#!/usr/bin/env python3
"""
Self-Pruning Neural Network — Main Entry Point
================================================

CLI-driven experiment runner that:

1. Loads configuration from ``config/config.yaml`` (or a custom path).
2. Iterates over experiment definitions (different λ values).
3. Trains a fresh ``SelfPruningNetwork`` for each experiment.
4. Evaluates test accuracy and sparsity.
5. Generates plots and a comprehensive Markdown report.

Usage
-----
Run all experiments defined in config::

    python main.py

Run with a custom config file::

    python main.py --config path/to/config.yaml

Run a single experiment by name::

    python main.py --experiment low_lambda

Override epochs and batch size from CLI::

    python main.py --epochs 50 --batch-size 256
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List

import torch

from src.models import SelfPruningNetwork
from src.training import Trainer
from src.data import get_cifar10_loaders
from src.evaluation import evaluate_model
from src.utils import (
    set_seed,
    get_device,
    load_config,
    ensure_dir,
    plot_gate_histogram,
    plot_accuracy_sparsity_tradeoff,
    generate_report,
)


# ====================================================================== #
# Argument parser
# ====================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — CIFAR-10 Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Run only the named experiment (must match a name in config).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override the learning rate.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the random seed.",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic random data instead of real CIFAR-10 (for pipeline testing).",
    )
    return parser


# ====================================================================== #
# Logging setup
# ====================================================================== #

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ====================================================================== #
# Single experiment runner
# ====================================================================== #

def run_experiment(
    experiment_cfg: Dict[str, Any],
    config: Dict[str, Any],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> Dict[str, Any]:
    """Train and evaluate a single experiment.

    Returns a dict with experiment name, lambda, accuracy, sparsity, etc.
    """
    exp_name = experiment_cfg["name"]
    lambda_sparse = experiment_cfg["lambda_sparse"]
    logger = logging.getLogger(f"experiment.{exp_name}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT: %s  |  lambda = %.2e", exp_name, lambda_sparse)
    logger.info("=" * 70)

    # Build fresh model
    model_cfg = config.get("model", {})
    model = SelfPruningNetwork(
        input_dim=model_cfg.get("input_dim", 3072),
        hidden_dims=model_cfg.get("hidden_dims", [1024, 512, 256]),
        output_dim=model_cfg.get("output_dim", 10),
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        lambda_sparse=lambda_sparse,
        experiment_name=exp_name,
    )
    history = trainer.fit()

    # Evaluate on test set
    threshold = config.get("pruning", {}).get("gate_threshold", 1e-2)
    metrics = evaluate_model(model, test_loader, device, threshold=threshold)

    logger.info(
        "RESULTS — Test Accuracy: %.2f%%, Sparsity: %.2f%%",
        metrics["accuracy"] * 100,
        metrics["sparsity_overall"] * 100,
    )

    # Compile result record
    result = {
        "name": exp_name,
        "lambda_sparse": lambda_sparse,
        "accuracy": metrics["accuracy"],
        "sparsity": metrics["sparsity_overall"],
        "history": history,
        "model": model,
    }
    # Include per-layer sparsity
    for key, val in metrics.items():
        if key.startswith("sparsity_layer_"):
            result[key] = val

    return result


# ====================================================================== #
# Main
# ====================================================================== #

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---- Load config ----
    config = load_config(args.config)

    # ---- CLI overrides ----
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_tensorboard:
        config.setdefault("logging", {})["tensorboard"] = False

    # ---- Setup ----
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger("main")

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device()

    # ---- Data ----
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})

    use_synthetic = getattr(args, "synthetic", False)
    logger.info("Loading %s dataset...", "SYNTHETIC" if use_synthetic else "CIFAR-10")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        root=dataset_cfg.get("root", "./data"),
        batch_size=train_cfg.get("batch_size", 128),
        num_workers=dataset_cfg.get("num_workers", 2),
        pin_memory=dataset_cfg.get("pin_memory", True),
        seed=seed,
        use_synthetic=use_synthetic,
    )
    logger.info(
        "Data loaded — Train: %d batches, Val: %d batches, Test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ---- Filter experiments ----
    experiments = config.get("experiments", [])
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            logger.error("Experiment '%s' not found in config.", args.experiment)
            sys.exit(1)

    if not experiments:
        logger.error("No experiments defined in config.")
        sys.exit(1)

    logger.info("Running %d experiment(s): %s",
                len(experiments), [e["name"] for e in experiments])

    # ---- Run experiments ----
    all_results: List[Dict[str, Any]] = []
    total_start = time.time()

    for exp_cfg in experiments:
        # Reset seed for each experiment for fair comparison
        set_seed(seed)
        result = run_experiment(
            exp_cfg, config, train_loader, val_loader, test_loader, device,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start
    logger.info("All experiments completed in %.1f seconds.", total_elapsed)

    # ---- Visualisation ----
    report_cfg = config.get("report", {})
    output_dir = report_cfg.get("output_dir", "./reports")
    ensure_dir(output_dir)

    # Gate histogram for the best model (by accuracy)
    best_result = max(all_results, key=lambda r: r["accuracy"])
    gate_hist_path = plot_gate_histogram(
        model=best_result["model"],
        save_path=f"{output_dir}/gate_histogram_{best_result['name']}.png",
        title=f"Gate Distribution — {best_result['name']} (λ={best_result['lambda_sparse']})",
    )
    logger.info("Gate histogram saved -> %s", gate_hist_path)

    # Trade-off plot
    tradeoff_path = plot_accuracy_sparsity_tradeoff(
        results=all_results,
        save_path=f"{output_dir}/accuracy_sparsity_tradeoff.png",
    )
    logger.info("Trade-off plot saved -> %s", tradeoff_path)

    # ---- Generate Report ----
    report_path = generate_report(
        experiment_results=all_results,
        gate_histogram_path=gate_hist_path,
        tradeoff_plot_path=tradeoff_path,
        output_dir=output_dir,
        filename=report_cfg.get("filename", "experiment_report.md"),
        config=config,
    )
    logger.info("Report generated -> %s", report_path)

    # ---- Final Summary ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("%-20s %-12s %-15s %-15s", "Experiment", "Lambda", "Accuracy (%)", "Sparsity (%)")
    logger.info("-" * 62)
    for r in all_results:
        logger.info(
            "%-20s %-12.2e %-15.2f %-15.2f",
            r["name"], r["lambda_sparse"],
            r["accuracy"] * 100, r["sparsity"] * 100,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
