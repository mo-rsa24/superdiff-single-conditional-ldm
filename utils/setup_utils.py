import os
from datetime import datetime
import json
import argparse
import jax
import tensorflow as tf
from flax.serialization import from_bytes
from typing import Tuple, Any, Optional, NamedTuple
import jax.numpy as jnp
import torch.utils.data

# Local Imports
from utils.model_utils import TrainStateWithEMA  # Custom TrainState
from utils import data_utils
from models.ae_kl import AutoencoderKL
from models.ldm_unet import ScoreNet


def ensure_dir(p):
    """Utility for folder setup."""
    os.makedirs(p, exist_ok=True)
    return p


# --- Setup Dirs and Resume Logic ---

class RunSetup(NamedTuple):
    """Holds all paths and initial state derived from arguments."""
    run_dir: str
    ckpt_latest: str
    ldm_meta_path: str
    samples_dir: str
    ckpt_dir: str


def setup_directories_and_metadata(args: argparse.Namespace) -> RunSetup:
    """Handles directory creation, path construction, and metadata saving."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.exp_name}-{ts}"
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, run_name)

    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")
    ldm_meta_path = os.path.join(run_dir, "ldm_meta.json")

    # Save arguments for reproducibility
    with open(ldm_meta_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    return RunSetup(run_dir, ckpt_latest, ldm_meta_path, samples_dir, ckpt_dir)


def resume_state_and_precompute(
        ldm_state: TrainStateWithEMA,
        ldm_model: ScoreNet,
        ae_model: AutoencoderKL,
        unrep_ae_params: Any,
        base_ds: torch.utils.data.Dataset,
        args: argparse.Namespace,
        run_setup: RunSetup,
        global_bs: int
) -> Tuple[TrainStateWithEMA, Optional[jnp.ndarray]]:
    """
    Handles loading the latest checkpoint and precomputing z0 if needed.

    Returns: (Updated LDM State, Precomputed z0 or None)
    """
    ckpt_latest = run_setup.ckpt_latest

    # 1. Resume TrainState
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        print(f"[info] Resuming LDM from {ckpt_latest}")
        with tf.io.gfile.GFile(ckpt_latest, "rb") as f:
            blob = f.read()

        # Create a dummy state matching the structure of the saved state for deserialization
        # This is required because from_bytes needs the structure to know how to deserialize the blob.
        dummy_state = TrainStateWithEMA.create(apply_fn=ldm_model.apply, params=ldm_state.params,
                                               ema_params=ldm_state.ema_params, tx=ldm_state.tx)
        ldm_state = from_bytes(dummy_state, blob)

        # Replicate restored state across devices
        ldm_state = jax.device_put_replicated(ldm_state, jax.local_devices())
        print(f"Resumed at global step: {int(ldm_state.step[0])}")
    else:
        # If not resuming, just replicate the initial state
        ldm_state = jax.device_put_replicated(ldm_state, jax.local_devices())

    # 2. Precompute z0 for Overfit-One
    precomputed_z0 = None
    if args.overfit_one:
        # Recompute z0 using the utility function
        precomputed_z0 = data_utils.precompute_overfit_z0(ae_model, unrep_ae_params, base_ds, args, global_bs)

    return ldm_state, precomputed_z0