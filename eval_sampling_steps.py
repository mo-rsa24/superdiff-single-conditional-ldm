import jax
import jax.numpy as jnp
import numpy as np
import time
import wandb
import os

from utils import model_utils, config_utils
from diffusion.sampling import Ancestral_Sampler
from diffusion.sde_vp import marginal_prob_std_fn

def main():
    p = config_utils.parse_args()
    # Add a custom list of steps to test (comma separated)
    args, _ = p.parse_known_args()

    STEPS_TO_TEST = [50, 100, 200, 300, 500]

    # Initialize W&B for this evaluation run
    run_name = f"eval-steps-{args.run_name}" if args.run_name else f"eval-steps-{int(time.time())}"
    wandb.init(project=args.wandb_project, name=run_name, tags=["eval", "sampling-efficiency"], config=vars(args))

    # 2. Load Models (VAE + LDM)
    print(f"⚡ Loading models...")
    ae_model, unrep_ae_params, ae_args = model_utils.load_autoencoder(args.ae_config_path, args.ae_ckpt_path)

    rng = jax.random.PRNGKey(args.seed)
    rng, model_rng = jax.random.split(rng)

    ldm_model, ldm_state, latent_size, z_channels = model_utils.setup_ldm_model_and_state(model_rng, ae_args, args)

    import tensorflow as tf
    from flax.serialization import from_bytes

    ckpt_path = args.resume_dir if args.resume_dir else os.path.join(args.output_root, "checkpoints/last.flax")
    # Note: You must pass the explicit path to the .flax file using --resume_dir or ensure it exists

    if tf.io.gfile.exists(ckpt_path):
        print(f"⚡ Restoring weights from: {ckpt_path}")
        with tf.io.gfile.GFile(ckpt_path, "rb") as f:
            ldm_state = from_bytes(ldm_state, f.read())
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please provide path via --resume_dir")

    # Replicate/Get params for JAX
    ldm_params = jax.device_get(jax.tree_map(lambda x: x[0], ldm_state.ema_params)) if args.use_ema else jax.device_get(
        ldm_state.params)
    ae_params = jax.device_get(unrep_ae_params)

    # 3. The Sampling Loop
    print(f"🚀 Starting Sampling Efficiency Study: {STEPS_TO_TEST} steps")

    for steps in STEPS_TO_TEST:
        print(f"\n--- Testing N={steps} steps ---")
        start_time = time.time()

        # Run Sampling (Batch size defined in args)
        rng, step_rng = jax.random.split(rng)

        # Create a dummy label batch (e.g., class 1 for TB)
        y = jnp.ones((args.sample_batch_size,), dtype=jnp.int32)

        samples_grid, final_latents = Ancestral_Sampler(
            rng=step_rng,
            ldm_model=ldm_model,
            ldm_params=ldm_params,
            ae_model=ae_model,
            ae_params=ae_params,
            marginal_prob_std_fn=marginal_prob_std_fn,
            latent_size=latent_size,
            batch_size=args.sample_batch_size,
            z_channels=z_channels,
            z_std=args.latent_scale_factor,
            y=y,
            guidance_scale=args.guidance_scale,
            num_classes=args.num_classes,
            num_steps=steps
        )

        duration = time.time() - start_time
        samples_per_sec = args.sample_batch_size / duration

        flat = np.array(final_latents).reshape(args.sample_batch_size, -1)
        dist = np.mean(np.square(flat[:, None] - flat[None, :]))

        log_data = {
            "efficiency/steps": steps,
            "efficiency/time_per_batch": duration,
            "efficiency/samples_per_sec": samples_per_sec,
            "quality/pairwise_mse": dist,
            f"images/sample_{steps}_steps": wandb.Image(samples_grid, caption=f"Steps: {steps}, Time: {duration:.2f}s")
        }
        wandb.log(log_data)
        print(f"   Done. Time: {duration:.2f}s | Diversity: {dist:.4f}")

    print("✅ Study Complete.")
    wandb.finish()


if __name__ == "__main__":
    main()