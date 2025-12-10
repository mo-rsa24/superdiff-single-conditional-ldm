# utils/model_utils.py
import json
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
from typing import Any, Tuple, Dict
import argparse

# Local Imports
from models.ae_kl import AutoencoderKL
from models.ldm_unet import ScoreNet

from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig

# --- Train State with EMA ---
class TrainStateWithEMA(TrainState):
    """Extends TrainState to hold an EMA copy of parameters."""
    ema_params: Any = None


def load_autoencoder(config_path: str, ckpt_path: str) -> Tuple[AutoencoderKL, Any, Dict]:
    """Loads the VAE model, parameters, and configuration arguments needed for LDM setup."""
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # 1. Parse configuration and build VAE model
    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        base_ch = ae_args.get('base_ch', 64)
        ae_ch_mults = tuple(base_ch * m for m in ch_mult_factors)
    else:
        ae_ch_mults = tuple(ae_args['ch_mults'])

    attn_res = tuple(int(r) for r in ae_args.get('attn_res', '16').split(',') if r)

    enc_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args['z_channels'],
                   double_z=True, attn_resolutions=attn_res, in_ch=1)
    dec_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1,
                   attn_resolutions=attn_res)

    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=ae_args['embed_dim'])

    # 2. Initialize dummy VAE state for checkpoint deserialization
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    def get_ae_tx(lr, grad_clip, weight_decay):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    tx = get_ae_tx(lr=ae_args.get('lr', 1e-4), grad_clip=ae_args.get('grad_clip', 1.0),
                   weight_decay=ae_args.get('weight_decay', 1e-4))

    gen_params = {'ae': ae_variables['params']}
    loss_cfg = LPIPSGANConfig(disc_num_layers=ae_args.get('disc_layers', 3))
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg)
    loss_params_dummy = loss_mod.init(rng, x_in=fake_img, x_rec=fake_img, posterior=None, step=jnp.array(0))['params']
    disc_params_dummy = {'loss': loss_params_dummy}

    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    # 3. Load checkpoint
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    # Checkpoint is saved as a tuple: (Generator_State, Discriminator_State)
    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    print("Autoencoder loaded successfully.")

    return ae_model, restored_gen_state.params['ae'], ae_args


def setup_ldm_model_and_state(rng: jax.random.PRNGKey, ae_args: Dict, args: argparse.Namespace) -> Tuple[
    ScoreNet, TrainStateWithEMA, int, int]:
    """Sets up the LDM ScoreNet, calculates latent size, and initializes the TrainState."""

    # 1. Calculate Latent Size (Diagnostics)
    z_channels = ae_args['z_channels']
    num_downsamples = len(ae_args.get('ch_mults', [])) - 1 if isinstance(ae_args['ch_mults'], list) else len(
        ae_args['ch_mults'].split(',')) - 1
    downsample_factor = 2 ** num_downsamples
    latent_size = args.img_size // downsample_factor

    print(f"Latent Spatial Size: {latent_size}x{latent_size}. Channels: {z_channels}")

    # 2. Initialize LDM UNet
    ldm_chans = tuple(args.ldm_base_ch * int(m) for m in args.ldm_ch_mults.split(','))
    attn_res = tuple(int(r) for r in args.ldm_attn_res.split(','))
    ldm_model = ScoreNet(z_channels=z_channels, channels=ldm_chans,
                         num_res_blocks=args.ldm_num_res_blocks, attn_resolutions=attn_res,
                         num_classes=args.num_classes)

    # Init dummy inputs to get initial parameters
    rng, init_rng = jax.random.split(rng)
    fake_latent = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    fake_y = jnp.ones((1,), dtype=jnp.int32)
    ldm_params = ldm_model.init(init_rng, fake_latent, fake_time, fake_y)['params']

    # 3. Setup TrainState (with EMA support)
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr, weight_decay=args.weight_decay))
    ema_params = ldm_params if args.use_ema else None
    ldm_state = TrainStateWithEMA.create(apply_fn=ldm_model.apply, params=ldm_params, ema_params=ema_params, tx=tx)

    return ldm_model, ldm_state, latent_size, z_channels