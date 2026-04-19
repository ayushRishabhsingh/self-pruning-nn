# Self-Pruning Neural Network for CIFAR-10

A production-quality PyTorch implementation of a **self-pruning feed-forward neural network** that learns to identify and remove unnecessary weights during training.

## Key Idea

Each weight in the network is paired with a **learnable gate** parameter. During training:

1. Gate scores pass through a **sigmoid** to produce values in `[0, 1]`.
2. Each weight is **multiplied by its gate value** — gates near 0 effectively prune the weight.
3. An **L1 penalty** on gate values encourages the network to push unnecessary gates toward zero.
4. The result is a **sparse network** that maintains competitive accuracy.

## Project Structure

```
newproj/
├── config/
│   └── config.yaml              # All hyperparameters & experiment definitions
├── src/
│   ├── models/
│   │   └── prunable_net.py      # PrunableLinear layer + SelfPruningNetwork
│   ├── training/
│   │   ├── loss.py              # SparsityAwareLoss (CE + λ·L1)
│   │   └── trainer.py           # Full training loop with early stopping
│   ├── data/
│   │   └── dataset.py           # CIFAR-10 loaders with augmentation
│   ├── evaluation/
│   │   └── metrics.py           # Accuracy & sparsity computation
│   └── utils/
│       ├── helpers.py           # Seed, device, config utilities
│       ├── visualization.py     # Gate histograms & trade-off plots
│       └── report.py            # Auto-generated Markdown report
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments

```bash
python main.py
```

This will train three models with different λ (sparsity penalty) values as defined in `config/config.yaml`, then generate plots and a report in the `reports/` directory.

### Run a single experiment

```bash
python main.py --experiment low_lambda
```

### CLI options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (default: `config/config.yaml`) |
| `--experiment NAME` | Run only the named experiment |
| `--epochs N` | Override number of training epochs |
| `--batch-size N` | Override batch size |
| `--lr FLOAT` | Override learning rate |
| `--seed INT` | Override random seed |
| `--no-tensorboard` | Disable TensorBoard logging |

### Monitor training with TensorBoard

```bash
tensorboard --logdir runs/
```

## Configuration

All hyperparameters are in `config/config.yaml`:

- **Model**: Input/hidden/output dimensions
- **Training**: Epochs, batch size, learning rate, optimizer
- **Pruning**: Gate threshold for sparsity measurement
- **Early stopping**: Patience and minimum delta
- **Experiments**: List of λ values to sweep

## How It Works

### PrunableLinear Layer

```
gates = sigmoid(gate_scores)          # [0, 1] per weight
pruned_weight = weight * gates        # Element-wise masking
output = input @ pruned_weight.T + bias
```

- `weight` and `gate_scores` are both `nn.Parameter` — gradients flow through both.
- No `torch.nn.Linear` is used internally.

### Loss Function

```
Total Loss = CrossEntropy(logits, targets) + λ × Σ sigmoid(gate_scores)
```

The L1 penalty on sigmoid outputs drives gate values toward 0, effectively pruning weights. The classification loss counterbalances, keeping essential connections alive.

### Sparsity Measurement

A weight is considered **pruned** if its gate value `sigmoid(gate_score) < 0.01`.

## Outputs

After training completes:

| Output | Location |
|--------|----------|
| Model checkpoints | `checkpoints/<experiment_name>/` |
| Gate histogram plot | `reports/gate_histogram_*.png` |
| Accuracy vs sparsity plot | `reports/accuracy_sparsity_tradeoff.png` |
| Full experiment report | `reports/experiment_report.md` |
| TensorBoard logs | `runs/<experiment_name>/` |

## Expected Results

| Lambda | Expected Accuracy | Expected Sparsity |
|--------|------------------|--------------------|
| 1e-5 (low) | ~52-55% | Low (~5-15%) |
| 1e-4 (medium) | ~48-52% | Medium (~30-60%) |
| 1e-3 (high) | ~35-45% | High (~70-95%) |

> **Note**: Exact numbers depend on the run. The key insight is the clear **trade-off** between accuracy and sparsity as λ increases.

## Technical Details

- **Reproducibility**: All random seeds (Python, NumPy, PyTorch, CUDA) are fixed.
- **GPU support**: Automatically uses CUDA if available.
- **Early stopping**: Monitors validation loss with configurable patience.
- **Checkpointing**: Best model (by val loss) is saved and restored after training.

## License

MIT
