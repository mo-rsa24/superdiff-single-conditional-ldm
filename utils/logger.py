import os
from datetime import datetime
from typing import Any
import numpy as np
import wandb
from rich.console import Console
from rich.table import Table

try:
    _console = Console(log_time=False, log_path=False)
    _RICH = True
except Exception:
    _RICH = False
    _console = None

_WANDB = 'wandb' in globals()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_wandb(args, run_name: str, project: str, tags: str = ""):
    """Initializes W&B and logs all run arguments and SLURM env vars."""
    if not _WANDB or not args.wandb:
        print("W&B logging is disabled.")
        return None

    # Parse tags
    tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else None

    # Log SLURM environment variables for reproducibility
    slurm_config = {k: os.environ[k] for k in os.environ if k.startswith('SLURM_')}

    # Initialize run
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "slurm": slurm_config,
            **vars(args)  # Log all argparse arguments
        },
        tags=tags_list,
        resume='allow'
    )
    print(f"✅ W&B initialized. Run Name: {run.name}")
    return run


def open_block(kind: str, step: int = None, epoch: int = None, note: str = ""):
    """Pretty-print a block header for structured logging."""
    tag = f"{kind.upper()} | step={step} epoch={epoch} time={_now()}"
    print(f"\n<<<{kind.upper()}_BEGIN step={step} epoch={epoch}>>>", flush=True)
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
        if note:
            _console.print(note, style="dim")
    else:
        print("=" * 100)
        print(tag)
        if note: print(note)
        print("-" * 100)


def close_block(kind: str, step: int = None):
    """Pretty-print a block footer."""
    tag = f"END {kind.upper()} | step={step} time={_now()}"
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
    else:
        print("-" * 100)
    print(f"<<<{kind.upper()}_END step={step}>>>", flush=True)


def pretty_table(title: str, metrics: dict):
    """Prints a clean metric table to console."""
    if _RICH:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        for k, v in metrics.items():
            # Handle JAX/Numpy array input if it slips through
            v = np.asarray(v).item() if isinstance(v, (np.ndarray, list)) and len(np.asarray(v).shape) == 0 else v
            table.add_row(k, f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v))
        _console.print(table)
    else:
        # Fallback to simple print
        keys = list(metrics.keys())
        w = max(len(k) for k in keys) if keys else 10
        print(f"\n{title}")
        for k in keys:
            v = metrics[k]
            v = np.asarray(v).item() if isinstance(v, (np.ndarray, list)) and len(np.asarray(v).shape) == 0 else v
            v = f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v)
            print(f"{k:<{w}} : {v}")


def log_sample_diversity(samples_np: np.ndarray, step: int, epoch: int, wandb_run: Any):
    """Calculates and logs pairwise MSE for sample collapse detection."""
    if samples_np.ndim != 4 or samples_np.shape[0] <= 1:
        print("[diversity] Batch size is 1 or shape is wrong, skipping diversity check.")
        return

    batch_size = samples_np.shape[0]
    flattened_samples = samples_np.reshape(batch_size, -1)

    # Calculate pairwise Mean Squared Error (MSE)
    sum_sq = np.sum(flattened_samples ** 2, axis=1, keepdims=True)
    dot_prod = flattened_samples @ flattened_samples.T
    mse_matrix = (sum_sq + sum_sq.T - 2 * dot_prod) / flattened_samples.shape[1]

    indices = np.triu_indices(batch_size, k=1)
    pairwise_mse_vals = mse_matrix[indices]

    metrics = {
        "pairwise_mse_mean": float(np.mean(pairwise_mse_vals)),
        "pairwise_mse_std": float(np.std(pairwise_mse_vals)),
        "pairwise_mse_min": float(np.min(pairwise_mse_vals)),
    }

    open_block("diversity", step=step, epoch=epoch, note="Pairwise MSE between generated samples in the batch")
    pretty_table("sample_diversity/metrics", metrics)
    close_block("diversity", step=step)

    if wandb_run is not None:
        # Log to W&B with clearer metric names
        wandb_run.log({
            "sample_diversity/pairwise_mse_mean": metrics["pairwise_mse_mean"],
            "sample_diversity/pairwise_mse_std": metrics["pairwise_mse_std"],
            "epoch": epoch,
            "step": step
        })