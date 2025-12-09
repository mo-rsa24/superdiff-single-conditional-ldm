import jax.numpy as jnp

# --- Variance Preserving (VP) SDE Definitions ---

# The VP-SDE is: dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) * dw

def get_beta(t, T=1.0, beta_min=0.1, beta_max=20.0):
    """Linear schedule for beta(t) over time t in [0, T]"""
    return beta_min + t * (beta_max - beta_min) / T

def alpha_sq_fn(t):
    """Integral of beta(t) used in the marginal probability std."""
    T = 1.0
    beta_min = 0.1
    beta_max = 20.0
    # Integral of beta(s) ds from 0 to t
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2

def alpha_fn(t):
    """Scaling factor for x0 (image) in the forward process: alpha(t) = exp(-0.5 * integral(beta(s)ds))"""
    return jnp.exp(-0.5 * alpha_sq_fn(t))

def marginal_prob_std_fn(t):
    """Marginal probability standard deviation: sigma(t) = sqrt(1 - alpha(t)^2)"""
    return jnp.sqrt(1.0 - alpha_fn(t)**2)

def diffusion_coeff_fn(t):
    """Diffusion coefficient: g(t) = sqrt(beta(t))"""
    return jnp.sqrt(get_beta(t))