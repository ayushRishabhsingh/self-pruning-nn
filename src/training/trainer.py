"""
Training Pipeline
==================

The ``Trainer`` class encapsulates the full training loop for the
self-pruning network:

* Forward pass → gated linear layers
* Loss computation (classification + sparsity)
* Backward pass & optimiser step
* Validation loop with early stopping
* Checkpoint saving
* TensorBoard logging
* Per-epoch metric tracking

Usage::

    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.fit()
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.prunable_net import SelfPruningNetwork
from src.training.loss import SparsityAwareLoss
from src.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitor a metric and stop training when it stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Complete training pipeline for the self-pruning network.

    Parameters
    ----------
    model : SelfPruningNetwork
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : dict
        Full configuration dictionary (from ``config.yaml``).
    device : torch.device
        Compute device.
    lambda_sparse : float
        Sparsity penalty weight (λ).
    experiment_name : str
        Identifier for this experiment run.
    """

    def __init__(
        self,
        model: SelfPruningNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        lambda_sparse: float = 1e-4,
        experiment_name: str = "default",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 0.0)
        self.log_interval = config.get("logging", {}).get("log_interval", 50)

        # Loss & optimiser
        self.criterion = SparsityAwareLoss(lambda_sparse=lambda_sparse)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Early stopping
        es_cfg = config.get("early_stopping", {})
        self.early_stopping: Optional[EarlyStopping] = None
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 7),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode="min",
            )

        # Checkpointing
        ckpt_cfg = config.get("checkpoint", {})
        self.ckpt_dir = Path(ckpt_cfg.get("dir", "./checkpoints")) / experiment_name
        self.save_best = ckpt_cfg.get("save_best", True)
        self.save_every = ckpt_cfg.get("save_every", 0)
        ensure_dir(str(self.ckpt_dir))

        # TensorBoard
        self.tb_writer = None
        log_cfg = config.get("logging", {})
        if log_cfg.get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(log_cfg.get("tensorboard_dir", "./runs")) / experiment_name
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info("TensorBoard logging -> %s", tb_dir)
            except ImportError:
                logger.warning("TensorBoard not available - skipping.")

        # State tracking
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict] = None
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self) -> List[Dict[str, float]]:
        """Run the full training loop.

        Returns
        -------
        list of dict
            Per-epoch metrics history.
        """
        logger.info(
            "Starting training: %d epochs, lambda=%.2e, device=%s",
            self.epochs,
            self.criterion.lambda_sparse,
            self.device,
        )
        param_info = self.model.count_parameters()
        logger.info(
            "Model parameters - total: %s, trainable: %s",
            f"{param_info['total']:,}",
            f"{param_info['trainable']:,}",
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # -- Train --
            train_metrics = self._train_one_epoch(epoch)

            # -- Validate --
            val_metrics = self._validate(epoch)

            elapsed = time.time() - t0
            sparsity = self.model.get_overall_sparsity(
                self.config.get("pruning", {}).get("gate_threshold", 1e-2)
            )

            # -- Compile epoch summary --
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_sparsity_loss": train_metrics["sparsity_loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_cls_loss": val_metrics["cls_loss"],
                "val_acc": val_metrics["accuracy"],
                "sparsity": sparsity,
                "elapsed": elapsed,
            }
            self.history.append(epoch_record)

            # -- Logging --
            logger.info(
                "Epoch %02d/%02d | "
                "Train Loss: %.4f (CE: %.4f, Sp: %.4f) Acc: %.2f%% | "
                "Val Loss: %.4f Acc: %.2f%% | "
                "Sparsity: %.2f%% | "
                "Time: %.1fs",
                epoch, self.epochs,
                train_metrics["loss"], train_metrics["cls_loss"],
                train_metrics["sparsity_loss"],
                train_metrics["accuracy"] * 100,
                val_metrics["loss"],
                val_metrics["accuracy"] * 100,
                sparsity * 100,
                elapsed,
            )

            # -- TensorBoard --
            if self.tb_writer:
                step = epoch
                self.tb_writer.add_scalar("Loss/train_total", train_metrics["loss"], step)
                self.tb_writer.add_scalar("Loss/train_cls", train_metrics["cls_loss"], step)
                self.tb_writer.add_scalar("Loss/train_sparsity", train_metrics["sparsity_loss"], step)
                self.tb_writer.add_scalar("Loss/val_total", val_metrics["loss"], step)
                self.tb_writer.add_scalar("Loss/val_cls", val_metrics["cls_loss"], step)
                self.tb_writer.add_scalar("Accuracy/train", train_metrics["accuracy"], step)
                self.tb_writer.add_scalar("Accuracy/val", val_metrics["accuracy"], step)
                self.tb_writer.add_scalar("Sparsity/overall", sparsity, step)

            # -- Checkpointing (best model) --
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.save_best:
                    self._save_checkpoint(epoch, is_best=True)

            # -- Periodic checkpoint --
            if self.save_every > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # -- Early stopping --
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["loss"]):
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d)",
                        epoch, self.early_stopping.patience,
                    )
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model (val loss: %.4f)", self.best_val_loss)

        if self.tb_writer:
            self.tb_writer.close()

        return self.history

    # ------------------------------------------------------------------ #
    # Single epoch
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch, return averaged metrics."""
        self.model.train()

        running_loss = 0.0
        running_cls = 0.0
        running_sp = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward
            logits = self.model(images)
            loss_dict = self.criterion(logits, labels, self.model)

            # Backward
            self.optimizer.zero_grad()
            loss_dict["total"].backward()
            self.optimizer.step()

            # Track metrics
            batch_size = labels.size(0)
            running_loss += loss_dict["total"].item() * batch_size
            running_cls += loss_dict["classification"].item() * batch_size
            running_sp += loss_dict["sparsity_weighted"].item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            # Batch-level logging
            if (batch_idx + 1) % self.log_interval == 0:
                logger.debug(
                    "  Epoch %02d Batch %04d/%04d - Loss: %.4f",
                    epoch, batch_idx + 1, len(self.train_loader),
                    loss_dict["total"].item(),
                )

        n = max(total, 1)
        return {
            "loss": running_loss / n,
            "cls_loss": running_cls / n,
            "sparsity_loss": running_sp / n,
            "accuracy": correct / n,
        }

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop, return averaged metrics."""
        self.model.eval()

        running_loss = 0.0
        running_cls = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss_dict = self.criterion(logits, labels, self.model)

            batch_size = labels.size(0)
            running_loss += loss_dict["total"].item() * batch_size
            running_cls += loss_dict["classification"].item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        n = max(total, 1)
        return {
            "loss": running_loss / n,
            "cls_loss": running_cls / n,
            "accuracy": correct / n,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint to disk."""
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        path = self.ckpt_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
            "config": self.config,
            "lambda_sparse": self.criterion.lambda_sparse,
            "experiment_name": self.experiment_name,
        }
        torch.save(checkpoint, path)
        tag = "(best)" if is_best else ""
        logger.debug("Checkpoint saved -> %s %s", path, tag)
