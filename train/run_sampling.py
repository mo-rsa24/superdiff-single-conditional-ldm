import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import tensorflow as tf
from flax.serialization import to_bytes
from torchvision.utils import save_image
from typing import Any, Tuple

# Local Imports
from utils import logger
from utils.model_utils import TrainStateWithEMA
from diffusion.sampling import Euler_Maruyama_sampler
from diffusion.sde_vp import marginal_prob_std_fn, diffusion_coeff_fn
from models.ldm_unet import ScoreNet
from models.ae_kl import AutoencoderKL

try:
    import wandb
except ImportError:
    wandb = None


def run_sampling_and_checkpoint(
        ep: int,
        global_step: int,
        ldm_state: TrainStateWithEMA,
        ae_model: AutoencoderKL,
        ae_params: Any,
        ldm_model: ScoreNet,
        args: argparse.Namespace,
        rng: jax.random.PRNGKey,
        latent_size: int,
        z_channels: int,
        run_setup: Any,  # RunSetup NamedTuple
        wandb_run: Any
) -> jax.random.PRNGKey:
    """
    Runs image generation (sampling), logs diversity metrics, and saves the checkpoint.
    """

    logger.open_block("sample", step=global_step, epoch=ep + 1)

    # --- 1. Select Parameters for Inference ---
    # Unreplicate params for inference (always unrep the base params first)
    unrep_ldm_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.params))

    # Determine whether to use EMA or standard trained parameters
    if args.use_ema and ldm_state.ema_params is not None:
        print("[sampling] Using EMA parameters.")
        sampling_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.ema_params))
    else:
        print("[sampling] Using trained parameters.")
        sampling_params = unrep_ldm_params

    # Get unreplicated AE params
    unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))

    class_names = {0: "Normal", 1: "TB"}

    rng, sample_rng_base = jax.random.split(rng)

    # --- 2. Run Sampler for Each Class ---
    for class_idx in range(args.num_classes):
        bs = args.sample_batch_size
        y_sample = jnp.full((bs,), class_idx, dtype=jnp.int32)

        sample_rng_base, step_rng = jax.random.split(sample_rng_base)

        # Sampler Call (using Euler-Maruyama with CFG)
        samples_grid, final_latent = Euler_Maruyama_sampler(
            rng=step_rng,
            ldm_model=ldm_model,
            ldm_params=sampling_params,
            ae_model=ae_model,
            ae_params=unrep_ae_params,
            marginal_prob_std_fn=marginal_prob_std_fn,
            diffusion_coeff_fn=diffusion_coeff_fn,
            latent_size=latent_size,
            batch_size=bs,
            z_channels=z_channels,
            z_std=args.latent_scale_factor,
            y=y_sample,
            guidance_scale=args.guidance_scale,
            num_classes=args.num_classes,
            num_steps=args.num_sampling_steps
        )

        # --- 3. Log Sampling Metrics ---
        final_latent_np = np.asarray(final_latent)
        logger.log_sample_diversity(final_latent_np, step=global_step, epoch=ep + 1, wandb_run=wandb_run)

        # --- 4. Save and Log Images ---
        class_name_str = class_names.get(class_idx, f"Class_{class_idx}")
        out_path = os.path.join(run_setup.samples_dir, f"sample_ep{ep + 1:04d}_{class_name_str}.png")
        save_image(samples_grid, out_path)

        if wandb_run:
            wandb_run.log({
                f"samples/{class_name_str}": wandb_run.Image(out_path, caption=f"Epoch {ep + 1} - {class_name_str}"),
                "epoch": ep + 1,
                "step": global_step
            })

    logger.close_block("sample", step=global_step)

    # --- 5. Save Checkpoint ---
    # Unreplicate the full state (params, tx, ema_params, step) from devices
    unrep_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state))
    with tf.io.gfile.GFile(run_setup.ckpt_latest, "wb") as f:
        f.write(to_bytes(unrep_state))
    print(f"Checkpoint saved to: {run_setup.ckpt_latest}")

    return rng


def finalize_wandb_and_cleanup(wandb_run: Any, run_setup: Any):
    """Logs the final checkpoint as a W&B Artifact and finishes the run."""
    if wandb_run and wandb:
        artifact = wandb.Artifact('ldm-model', type='model')
        artifact.add_file(run_setup.ckpt_latest)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()