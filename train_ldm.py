import os
import jax
import tensorflow as tf

from utils import logger
from utils import model_utils
from utils import data_utils
from utils import setup_utils
from utils.config_utils import parse_args

from train.ldm_step import create_train_step
from train.run_epoch import run_training_epoch
from train.run_sampling import run_sampling_and_checkpoint, finalize_wandb_and_cleanup

# W&B is optional
try:
    import wandb
except ImportError:
    wandb = None


# --- Helper Functions (Only simple ones remain) ---
def ensure_dir(p):
    """Utility for folder setup. Moved here for top-level usage."""
    os.makedirs(p, exist_ok=True)
    return p


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    # --- 1. Setup Directories and Metadata --- (ABSTRACTED)
    run_setup = setup_utils.setup_directories_and_metadata(args)

    # --- 2. Setup W&B and Initial Logging ---
    wandb_run = logger.init_wandb(
        args,
        run_name=args.run_name or run_setup.run_dir.split('/')[-1],  # Use folder name if run_name is None
        project=args.wandb_project,
        tags=args.wandb_tags
    )
    use_wandb = wandb_run is not None
    print(f"EMA Status: {'✅ Enabled' if args.use_ema else '❌ Disabled'}. CFG Dropout: {args.prob_uncond}")

    # --- 3. Setup Data Loaders ---
    loader, base_ds, global_bs = data_utils.setup_dataset_and_loader(args)

    # --- 4. Load VAE and Setup LDM Model/State ---
    ae_model, unrep_ae_params, ae_args = model_utils.load_autoencoder(args.ae_config_path, args.ae_ckpt_path)
    ae_params = jax.device_put_replicated(unrep_ae_params, jax.local_devices())

    rng, model_rng = jax.random.split(rng)
    ldm_model, ldm_state_initial, latent_size, z_channels = model_utils.setup_ldm_model_and_state(model_rng, ae_args,
                                                                                                  args)

    # --- 5. Resume TrainState and Precompute z0 --- (ABSTRACTED)
    ldm_state, precomputed_z0 = setup_utils.resume_state_and_precompute(
        ldm_state_initial, ldm_model, ae_model, unrep_ae_params, base_ds, args, run_setup, global_bs
    )

    # --- 6. Create P-mapped Training Step ---
    pmapped_train_step = create_train_step(ldm_model, ae_model, args)

    # --- 7. Training Loop --- (ABSTRACTED)
    global_step = int(ldm_state.step[0])
    for ep in range(args.epochs):

        # Run one epoch of training
        ldm_state, rng, global_step = run_training_epoch(
            ep=ep,
            ldm_state=ldm_state,
            ae_params=ae_params,
            pmapped_train_step=pmapped_train_step,
            loader=loader,
            args=args,
            rng=rng,
            wandb_run=wandb_run
        )

        # --- 8. Sampling and Checkpointing --- (ABSTRACTED)
        if (ep + 1) % args.sample_every == 0:
            rng = run_sampling_and_checkpoint(
                ep=ep,
                global_step=global_step,
                ldm_state=ldm_state,
                ae_model=ae_model,
                ae_params=ae_params,
                ldm_model=ldm_model,
                args=args,
                rng=rng,
                latent_size=latent_size,
                z_channels=z_channels,
                run_setup=run_setup,
                wandb_run=wandb_run
            )

    # --- 9. Final Cleanup and Artifact Logging ---
    finalize_wandb_and_cleanup(wandb_run, run_setup)

    print(f"Training complete. Artifacts saved to: {run_setup.run_dir}")


if __name__ == "__main__":
    # Ensure TensorFlow doesn't steal JAX's GPU resources
    tf.config.experimental.set_visible_devices([], "GPU")
    main()