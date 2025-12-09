# diffusion/sampling.py
import jax
import jax.numpy as jnp
import numpy as np
from torchvision.utils import make_grid
from typing import Any, Tuple, Callable
from models.ae_kl import AutoencoderKL
from models.ldm_unet import ScoreNet  # Renamed from cxr_unet


# --- Sampling Function ---

def Euler_Maruyama_sampler(
        rng: jax.random.PRNGKey,
        ldm_model: ScoreNet,
        ldm_params: Any,
        ae_model: AutoencoderKL,
        ae_params: Any,
        marginal_prob_std_fn: Callable,
        diffusion_coeff_fn: Callable,
        latent_size: int,
        batch_size: int,
        z_channels: int,
        z_std: float,
        y: jnp.ndarray,  # Class labels for conditioning
        guidance_scale: float,  # CFG scale
        num_classes: int,
        num_steps: int = 500,
) -> Tuple[Any, jnp.ndarray]:
    """
    Implements the Euler-Maruyama sampler with Classifier-Free Guidance (CFG).

    Returns: (Samples grid (PyTorch Tensor), Final latent (JAX array))
    """

    # 1. Initialization
    eps_init, rng = jax.random.split(rng)

    # Start at T=1 with pure noise (z_std determines the magnitude of this initial noise)
    z = jax.random.normal(eps_init, (batch_size, latent_size, latent_size, z_channels)) * z_std

    # Set time steps
    ts = jnp.linspace(1.0, 1e-5, num_steps)
    dt = ts[0] - ts[1]

    # 2. CFG Setup: create unconditional label (index = num_classes)
    y_uncond = jnp.full_like(y, num_classes, dtype=jnp.int32)

    def loop_body(i, state):
        rng, z = state
        t = ts[i]
        t_batch = jnp.full((batch_size,), t)

        # 2a. Predict noise for conditional and unconditional paths
        # This requires two forward passes in the UNet.

        # Conditional prediction: epsilon_theta(z_t, t, y)
        eps_cond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y)

        # Unconditional prediction: epsilon_theta(z_t, t, null)
        eps_uncond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y_uncond)

        # 2b. Compute CFG-blended noise estimate (epsilon_hat)
        # eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # 2c. Compute SDE components
        std = marginal_prob_std_fn(t)
        sigma = std * z_std  # Scale the noise magnitude by z_std
        g = diffusion_coeff_fn(t)

        # 2d. Euler-Maruyama Step: z_{t-dt} = z_t + f(z_t, t) dt + g(t) dw

        # Drift term: f(z_t, t) = -0.5 * beta(t) * z_t - beta(t) * sigma(t)^2 * score(z_t, t)
        # where score(z_t, t) = -eps_hat / sigma(t)
        # Simplification: f(z_t, t) * dt = 0.5 * beta(t) * (z_t - (z_t + eps_hat * (sigma^2 / std))) dt

        # Simplest form (Noise Prediction Model):
        # z_{t-1} = z_t + (-1/2 * beta * z_t - beta * score(z_t, t)) * dt + sqrt(beta) dw
        # For simplicity and robustness, we use the standard SDE form and approximate the score.

        # Simplified Euler-Maruyama update for noise prediction:
        drift = -0.5 * (diffusion_coeff_fn(t) ** 2) * (z / (sigma ** 2))

        # Use the epsilon prediction approximation for the score function: score = -eps_hat / sigma
        drift = drift - (diffusion_coeff_fn(t) ** 2) * (-eps_hat / sigma)

        # Use the simplified Denoising Diffusion Implicit Model (DDIM) update for speed and stability
        # The equation above often simplifies to:
        a_t = marginal_prob_std_fn(t)
        a_t_minus_dt = marginal_std = marginal_prob_std_fn(t - dt)

        # DDIM update approximation: z_t-dt = prediction_of_x0 + sqrt(1 - a_t-dt^2) * new_noise
        # where prediction_of_x0 = (z - a_t * eps_hat) / (alpha_fn(t))

        # Use the original SDE-based sampling for better fidelity:
        # Compute the step using the score/noise prediction
        step_size = jnp.abs(dt)
        drift = -0.5 * get_beta(t) * z - get_beta(t) * (sigma ** 2) * (-eps_hat / sigma)

        # Stochastic term: g(t) * dw
        rng, step_rng = jax.random.split(rng)
        dw = jnp.sqrt(step_size) * jax.random.normal(step_rng, z.shape)

        # Next step
        z_next = z + drift * step_size + diffusion_coeff_fn(t) * dw

        return rng, z_next

    # 3. JAX Loop
    rng, final_z = jax.lax.fori_loop(0, num_steps, loop_body, (rng, z))

    # 4. Decode
    # The latent space z is scaled by z_std (or latent_scale_factor in LDM paper). 
    # We unscale it before passing to the AE decoder, which expects the original scale.
    z_unscaled = final_z / z_std
    x_rec = ae_model.apply({'params': ae_params}, z_unscaled, method=ae_model.decode, train=False)

    # 5. Post-process (NHWC to NCHW, JAX to Numpy, [0,1] to [-1,1] for `save_image`)
    x_rec_np = np.asarray(x_rec)
    x_rec_np = (x_rec_np * 2.0) - 1.0  # Convert [0,1] back to [-1,1] for save_image
    x_rec_np = np.transpose(x_rec_np, (0, 3, 1, 2))  # NHWC -> NCHW
    x_rec_tensor = make_grid(jnp.asarray(x_rec_np), nrow=4, padding=2, normalize=True, range=(-1, 1))

    return x_rec_tensor, final_z