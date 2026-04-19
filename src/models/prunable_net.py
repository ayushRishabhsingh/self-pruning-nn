"""
Self-Pruning Neural Network — Model Definitions
=================================================

This module implements the core building blocks for a self-pruning network:

1.  **PrunableLinear**: A drop-in replacement for ``nn.Linear`` that associates
    every weight with a *learnable gate* parameter.  During the forward pass the
    gate values are squashed through a sigmoid and element-wise multiplied into
    the weight matrix.  An L1 penalty on the sigmoid outputs drives gates
    toward 0, effectively pruning the corresponding weights.

2.  **SelfPruningNetwork**: A feed-forward classifier composed of
    ``PrunableLinear`` layers with ReLU activations, designed for CIFAR-10.

Design rationale
----------------
* ``gate_scores`` live in *logit* space (unconstrained reals).  The sigmoid
  maps them to [0, 1], ensuring numerical stability and smooth gradients.
* Weight initialisation follows Kaiming uniform (suitable for ReLU) while
  gate scores are initialised to +2.0 so that ``sigmoid(2) ≈ 0.88``, meaning
  all weights start "alive" and pruning is a *learned* decision.
* No ``torch.nn.Linear`` is used internally — weights and biases are raw
  ``nn.Parameter`` objects, guaranteeing full transparency in gradient flow.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PrunableLinear Layer
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """Fully-connected layer with per-weight learnable gates.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default True
        If ``True``, the layer includes a learnable bias.
    gate_init : float, default 2.0
        Initial value for ``gate_scores`` (in logit space).  A positive value
        ensures gates start near 1 after the sigmoid, keeping all weights
        active at the beginning of training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ---- Learnable parameters (no nn.Linear used) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), gate_init))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialise weights (Kaiming uniform, fan_in mode for ReLU)
        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _reset_parameters(self) -> None:
        """Kaiming uniform init for weights; uniform init for bias."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # gate_scores are already initialised to `gate_init` in __init__

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gated linear transformation.

        1. ``gates = sigmoid(gate_scores)``
        2. ``pruned_weight = weight * gates``
        3. ``output = x @ pruned_weight^T + bias``

        Gradients flow through **both** ``weight`` and ``gate_scores``.
        """
        gates = torch.sigmoid(self.gate_scores)           # (out, in) ∈ [0, 1]
        pruned_weight = self.weight * gates                # element-wise masking
        return F.linear(x, pruned_weight, self.bias)       # standard linear op

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    def get_gate_values(self) -> torch.Tensor:
        """Return current gate values (detached, on CPU)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu()

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below *threshold* (i.e. effectively pruned)."""
        gates = self.get_gate_values()
        return float((gates < threshold).sum().item() / gates.numel())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Full Network
# ---------------------------------------------------------------------------

class SelfPruningNetwork(nn.Module):
    """Feed-forward classifier built from :class:`PrunableLinear` layers.

    Architecture::

        Input → PrunableLinear → ReLU → … → PrunableLinear → Logits

    Parameters
    ----------
    input_dim : int
        Flattened input dimension (3072 for CIFAR-10).
    hidden_dims : list of int
        Widths of hidden layers.
    output_dim : int
        Number of classes (10 for CIFAR-10).
    gate_init : float, default 2.0
        Passed to each ``PrunableLinear`` layer.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 10,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Build layer stack
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(PrunableLinear(prev_dim, h_dim, gate_init=gate_init))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim

        # Output layer (no activation — CrossEntropyLoss expects raw logits)
        layers.append(PrunableLinear(prev_dim, output_dim, gate_init=gate_init))
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten input and pass through the layer stack."""
        x = x.view(x.size(0), -1)   # (B, C, H, W) → (B, C*H*W)
        return self.network(x)

    # ------------------------------------------------------------------ #
    # Gate aggregation (used by the loss function)
    # ------------------------------------------------------------------ #

    def get_all_gate_values(self) -> List[torch.Tensor]:
        """Collect sigmoid-activated gate values from every PrunableLinear."""
        gate_vals: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gate_vals.append(torch.sigmoid(module.gate_scores))
        return gate_vals

    def get_total_gate_l1(self) -> torch.Tensor:
        """Sum of *all* gate values across every layer (differentiable)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for gate in self.get_all_gate_values():
            total = total + gate.sum()
        return total

    def get_overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below *threshold* across the entire network."""
        pruned = 0
        total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()
                pruned += (gates < threshold).sum().item()
                total += gates.numel()
        return pruned / total if total > 0 else 0.0

    def count_parameters(self) -> dict:
        """Return parameter counts for reporting."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
