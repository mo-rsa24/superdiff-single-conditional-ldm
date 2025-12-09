import jax
import jax.numpy as jnp
import optax
import argparse
from typing import Any, Tuple

# Local imports
from diffusion.sde_vp import marginal_prob_std_fn, alpha_fn
from utils.model_utils import TrainStateWithEMA  # Custom TrainState definition
from models.ldm_unet import ScoreNet
from models.ae_kl import AutoencoderKL


def create_train_step(ldm_model: ScoreNet, ae_model: AutoencoderKL, args: argparse.Namespace):
    """
    Creates the pmapped training step function, capturing the necessary
    models and static arguments.
    """

    # 1. Define the monolithic training step (runs on each device)
    # Note: ae_params are passed as data (not updated)
    def train_step(rng: jax.random.PRNGKey, ldm_state: TrainStateWithEMA, ae_params: Any, x_batch: jnp.ndarray,
                   y_batch: jnp.ndarray) -> Tuple[TrainStateWithEMA, jnp.ndarray, dict]:
        """One pmap-ed training step for epsilon prediction."""

        rng, rng_diff, rng_drop = jax.random.split(rng, 3)

        def loss_fn(ldm_params: Any):
            # 1. Encode image to latent space
            posterior = ae_model.apply({'params': ae_params}, x_batch, method=ae_model.encode, train=False)
            rng_z, rng_noise = jax.random.split(rng_diff, 2)
            z = posterior.sample(rng_z) * args.latent_scale_factor

            # 2. Sample time t and noise ε
            rng_t, rng_noise = jax.random.split(rng_noise, 2)
            t = jax.random.uniform(rng_t, (z.shape[0],), minval=1e-5, maxval=1.0)
            noise = jax.random.normal(rng_noise, z.shape)

            # 3. VP SDE Forward perturbation: x_t = α z + σ ε
            sigma = marginal_prob_std_fn(t)
            alpha = alpha_fn(t)
            sigma_b = sigma[:, None, None, None]
            alpha_b = alpha[:, None, None, None]
            x_t = alpha_b * z + sigma_b * noise

            # 4. Conditional Dropout for CFG Training
            keep_mask = jax.random.bernoulli(rng_drop, p=(1.0 - args.prob_uncond), shape=y_batch.shape)
            y_in = jnp.where(keep_mask, y_batch, args.num_classes)  # args.num_classes is the null token index

            # 5. Predict ε and compute simple ε-MSE loss
            eps_hat = ldm_model.apply({'params': ldm_params}, x_t, t, y=y_in)
            loss = jnp.mean((eps_hat - noise) ** 2)

            # 6. Auxiliary diagnostics (Cosine similarity, norms)
            def _cos(a, b, eps=1e-8):
                num = jnp.sum(a * b, axis=tuple(range(1, a.ndim)))
                den = jnp.linalg.norm(a.reshape(a.shape[0], -1), axis=-1) * \
                      jnp.linalg.norm(b.reshape(b.shape[0], -1), axis=-1) + eps
                return num / den

            aux = dict(
                t_mean=jnp.mean(t),
                sigma_mean=jnp.mean(sigma),
                alpha_mean=jnp.mean(alpha),
                cos_eps=jnp.mean(_cos(eps_hat, noise)),
                z_mean=jnp.mean(z), z_std=jnp.std(z),
                xt_mean=jnp.mean(x_t), xt_std=jnp.std(x_t),
                eps_hat_mean=jnp.mean(eps_hat), eps_hat_std=jnp.std(eps_hat),
            )
            return loss, aux

        # 7. Compute gradients and apply pmean reduction
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ldm_state.params)
        grads = jax.lax.pmean(grads, axis_name='device')
        grad_norm = optax.global_norm(grads)
        aux = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
        aux = {**aux, "grad_norm": jax.lax.pmean(grad_norm, axis_name='device')}
        loss = jax.lax.pmean(loss, axis_name='device')

        # 8. Update state
        new_ldm_state = ldm_state.apply_gradients(grads=grads)

        # 9. EMA Update (if enabled)
        if args.use_ema:
            new_ema_params = optax.incremental_update(new_ldm_state.params, ldm_state.ema_params, args.ema_decay)
            new_ldm_state = new_ldm_state.replace(ema_params=new_ema_params)

        return new_ldm_state, loss, aux

    # 10. Pmap the function for multi-device execution
    return jax.pmap(train_step, axis_name='device')