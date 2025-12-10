import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchvision.utils import make_grid
from typing import Any, Tuple, Callable
from diffusion.sde_vp import get_beta, alpha_fn
from models.ae_kl import AutoencoderKL
from models.ldm_unet import ScoreNet


def Euler_Maruyama_sampler(
        rng: jax.random.PRNGKey,
        ldm_model: ScoreNet,
        ldm_params: Any,
        ae_model: AutoencoderKL,
        ae_params: Any,
        marginal_prob_std_fn: Callable,
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
    ts = jnp.linspace(1.0, 1e-4, num_steps)
    y_uncond = jnp.full_like(y, num_classes, dtype=jnp.int32)

    def loop_body(i, state):
        rng, z = state
        t = ts[i]
        s = jnp.where(i == num_steps - 1, 0.0, ts[i + 1])
        t_batch = jnp.full((batch_size,), t)

        eps_cond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y)
        eps_uncond = ldm_model.apply({'params': ldm_params}, z, t_batch, y=y_uncond)
        eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_t = alpha_fn(t)
        alpha_s = alpha_fn(s)
        sigma_t = marginal_prob_std_fn(t)
        sigma_s = marginal_prob_std_fn(s)

        z0_pred = (z - sigma_t * z_std * eps_hat) / alpha_t

        coeff_z0 = (alpha_s * (sigma_t ** 2 - sigma_s ** 2)) / (sigma_t ** 2 + 1e-8)
        coeff_zt = (alpha_t / alpha_s) * (sigma_s ** 2 / (sigma_t ** 2 + 1e-8))
        mu = coeff_z0 * z0_pred + coeff_zt * z

        sigma_post = sigma_s * jnp.sqrt((sigma_t ** 2 - sigma_s ** 2) / (sigma_t ** 2 + 1e-8))

        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, z.shape)
        z_next = mu + sigma_post * z_std * noise

        return rng, z_next

    rng, final_z = jax.lax.fori_loop(0, num_steps, loop_body, (rng, z))

    z_unscaled = final_z / z_std
    x_rec = ae_model.apply({'params': ae_params}, z_unscaled, method=ae_model.decode, train=False)

    x_rec_np = np.asarray(x_rec)
    x_rec_np = (x_rec_np * 2.0) - 1.0
    x_rec_np = np.transpose(x_rec_np, (0, 3, 1, 2))

    x_rec_tensor = make_grid(
        torch.from_numpy(x_rec_np),
        nrow=4,
        padding=2,
        normalize=True,
        value_range=(-1, 1)
    )
    return x_rec_tensor, final_z