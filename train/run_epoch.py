# train/run_epoch.py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import argparse
from typing import Any, Tuple
import torch.utils.data

# Local Imports
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

    for batch in progress_bar:
        x, y = batch

        # --- Data Preparation for pmap ---
        # Convert Torch tensor to JAX array, swap axes (NCHW -> NHWC)
        x = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
        # Normalize to [0,1] for VAE encoding (if AE was trained on [0,1])
        x = (x + 1.0) / 2.0

        # Shard data and RNG for pmap
        x_sharded = x.reshape((jax.local_device_count(), -1) + x.shape[1:])
        y = jnp.asarray(y.numpy())
        y_sharded = y.reshape((jax.local_device_count(), -1))

        rng, step_rng = jax.random.split(rng)
        rng_sharded = jax.random.split(step_rng, jax.local_device_count())

        # --- Execute Training Step ---
        ldm_state, loss, aux = pmapped_train_step(rng_sharded, ldm_state, ae_params, x_sharded, y_sharded)

        # --- Logging ---
        if global_step % args.log_every == 0:
            loss_val = float(np.asarray(loss[0]))
            # Get host metrics from device 0, converted to native Python floats
            aux_host = jax.tree_map(lambda x: float(np.asarray(x[0])), aux)

            metrics = {
                "step": global_step,
                "loss/total_loss": loss_val,
                "optimization/grad_norm": aux_host["grad_norm"],
                "diffusion/t_mean": aux_host["t_mean"],
                "diffusion/cos_eps_mean": aux_host["cos_eps"],
                "latent/z_std": aux_host["z_std"],
            }
            logger.open_block("train", step=global_step, epoch=ep + 1)
            logger.pretty_table("train/metrics", metrics)
            logger.close_block("train", step=global_step)

            if wandb_run:
                wandb_run.log({"train/loss": loss_val, "train/step": global_step, **metrics})

        global_step += 1

    return ldm_state, rng, global_step