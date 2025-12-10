from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import argparse
from typing import Any, Tuple
import torch.utils.data
from utils import logger
from utils.model_utils import TrainStateWithEMA


def run_training_epoch(
        ep: int,
        ldm_state: TrainStateWithEMA,
        ae_params: Any,
        pmapped_train_step: Any,
        loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
        rng: jax.random.PRNGKey,
        wandb_run: Any
) -> Tuple[TrainStateWithEMA, jax.random.PRNGKey, int]:
    """Runs a single epoch of training, logging metrics at specified intervals."""

    global_step = int(ldm_state.step[0])
    progress_bar = tqdm(loader, desc=f"Epoch {ep + 1}/{args.epochs}", leave=False)
    epoch_metrics = defaultdict(list)
    for batch in progress_bar:
        x, y = batch
        x = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
        # Normalize to [0,1] for VAE encoding (if AE was trained on [0,1])
        x = (x + 1.0) / 2.0
        x_sharded = x.reshape((jax.local_device_count(), -1) + x.shape[1:])
        y = jnp.asarray(y.numpy())
        y_sharded = y.reshape((jax.local_device_count(), -1))
        rng, step_rng = jax.random.split(rng)
        rng_sharded = jax.random.split(step_rng, jax.local_device_count())
        ldm_state, loss, aux = pmapped_train_step(rng_sharded, ldm_state, ae_params, x_sharded, y_sharded)
        loss_val = float(np.asarray(loss[0]))
        epoch_metrics["loss/total_loss"].append(loss_val)
        aux_host = jax.tree_map(lambda x: float(np.asarray(x[0])), aux)
        epoch_metrics["optimization/grad_norm"].append(aux_host["grad_norm"])
        epoch_metrics["diffusion/t_mean"].append(aux_host["t_mean"])
        epoch_metrics["diffusion/cos_eps_mean"].append(aux_host["cos_eps"])
        epoch_metrics["latent/z_std"].append(aux_host["z_std"])
        global_step += 1

    avgs = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
    metrics_to_log = {
        "step": global_step,
        "epoch": ep + 1,
        **avgs
    }

    logger.open_block("train", step=global_step, epoch=ep + 1)
    logger.pretty_table("train/epoch_metrics", metrics_to_log)
    logger.close_block("train", step=global_step)

    if wandb_run:
        wandb_run.log(metrics_to_log)

    return ldm_state, rng, global_step