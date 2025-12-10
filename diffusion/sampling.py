import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchvision.utils import make_grid
from typing import Any, Tuple, Callable
from diffusion.sde_vp import get_beta
from models.ae_kl import AutoencoderKL
from models.ldm_unet import ScoreNet

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
        y: jnp.ndarray,
        guidance_scale: float,
        num_classes: int,
        num_steps: int = 500,
) -> Tuple[Any, jnp.ndarray]:
    eps_init, rng = jax.random.split(rng)
    z = jax.random.normal(eps_init, (batch_size, latent_size, latent_size, z_channels)) * z_std
    ts = jnp.linspace(1.0, 1e-5, num_steps)
    dt = ts[0] - ts[1]
    y_uncond = jnp.full_like(y, num_classes, dtype=jnp.int32)
    def loop_body(i, state):
        rng, z = state
        t = ts[i]
        t_batch = jnp.full((batch_size,), t)
        eps_cond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y)
        eps_uncond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y_uncond)
        eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        std = marginal_prob_std_fn(t)
        sigma = std * z_std  # Scaled standard deviation
        g = diffusion_coeff_fn(t)  # Diffusion coefficient g(t) = sqrt(beta(t))
        step_size = jnp.abs(dt)
        drift = -0.5 * get_beta(t) * z - get_beta(t) * (-eps_hat / sigma)
        rng, step_rng = jax.random.split(rng)
        dw = jnp.sqrt(step_size) * jax.random.normal(step_rng, z.shape)
        z_next = z + drift * step_size + g * dw
        return rng, z_next
    rng, final_z = jax.lax.fori_loop(0, num_steps, loop_body, (rng, z))
    z_unscaled = final_z / z_std
    x_rec = ae_model.apply({'params': ae_params}, z_unscaled, method=ae_model.decode, train=False)
    x_rec_np = np.asarray(x_rec)
    x_rec_np = (x_rec_np * 2.0) - 1.0  # Convert [0,1] back to [-1,1] for save_image
    x_rec_np = np.transpose(x_rec_np, (0, 3, 1, 2))  # NHWC -> NCHW
    x_rec_tensor = make_grid(
        torch.from_numpy(x_rec_np),
        nrow=4,
        padding=2,
        normalize=True,
        value_range=(-1, 1)
    )
    return x_rec_tensor, final_z