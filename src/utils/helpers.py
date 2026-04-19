"""
Utility Helpers
================

Common utilities: seed setting, device detection, config loading, and
directory creation.
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using device: %s", device)
    return device


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", path)
    return config


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
