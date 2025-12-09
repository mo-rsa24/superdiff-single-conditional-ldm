import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence

class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class DenseToMap(nn.Module):
    """Maps the 1D embedding (B, C) to a 4D tensor (B, 1, 1, C) for broadcasting."""
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)[:, None, None, :]


def _pick_gn_groups(C: int) -> int:
    """Helper function to determine the number of groups for GroupNorm."""
    g = min(32, C)
    while g > 1 and (C % g) != 0:
        g //= 2
    return max(1, g)


class ResBlock(nn.Module):
    c: int  # Output channels
    embed_dim: int  # Time/Class embedding dimension
    scale_skip: bool = True  # Scale residual connection by 1/sqrt(2)

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_ch = x.shape[-1]

        # 1st Conv
        h = nn.GroupNorm(num_groups=_pick_gn_groups(in_ch))(x)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)

        # Add Time/Class Embedding
        h = h + DenseToMap(self.c)(t_embed)

        # 2nd Conv
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c))(h)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)

        # Residual Connection
        if in_ch != self.c:
            x = nn.Conv(self.c, (1, 1), padding='SAME', use_bias=False, name='skip_proj')(x)
        if self.scale_skip:
            x = x * (1.0 / jnp.sqrt(2.0))

        return act(h + x)


class SelfAttention2D(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.LayerNorm()(x)
        h = h.reshape((B, H * W, C))
        # Flax attention expects a single sequence, hence the reshape
        h = nn.SelfAttention(num_heads=self.num_heads)(h)
        h = h.reshape((B, H, W, C))
        return x + h


# --- Conditional UNet Model ---

class ScoreNet(nn.Module):
    """
    Conditional UNet model for predicting the noise (epsilon) in the latent space.
    Conditional on time (t) and class label (y).
    """
    z_channels: int = 4
    channels: Sequence[int] = (128, 256, 512)
    embed_dim: int = 256
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    num_heads: int = 4
    num_classes: int = 2  # The number of actual classes (e.g., 2 for Normal/TB)

    @nn.compact
    def __call__(self, x, t, y=None, train: bool = True):
        act = nn.swish

        # 1. Time Embedding
        temb = act(nn.Dense(self.embed_dim)(GaussianFourierProjection(self.embed_dim)(t)))

        # 2. Class Embedding (Conditional Injection via CFG-compatible embedding)
        if y is not None:
            # We use num_classes + 1 embeddings. Index `num_classes` is the "null" token.
            class_embed = nn.Embed(num_embeddings=self.num_classes + 1, features=self.embed_dim)(y)
            temb = temb + class_embed

            # 3. Encoder (Downsampling blocks)
        h = nn.Conv(self.channels[0], (3, 3), padding='SAME')(x)
        skips = [h]
        for i, ch in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                h = ResBlock(ch, self.embed_dim)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads)(h)
                skips.append(h)

            # Downsample
            if i < len(self.channels) - 1:
                h = nn.Conv(self.channels[i + 1], (3, 3), strides=(2, 2), padding='SAME')(h)
                skips.append(h)

        # 4. Bottleneck
        h = ResBlock(self.channels[-1], self.embed_dim)(h, temb)
        h = SelfAttention2D(num_heads=self.num_heads)(h)
        h = ResBlock(self.channels[-1], self.embed_dim)(h, temb)

        # 5. Decoder (Upsampling blocks)
        for i in reversed(range(len(self.channels))):
            # The decoder needs one extra ResBlock application because the skip connection is added first
            for _ in range(self.num_res_blocks + 1):
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = ResBlock(self.channels[i], self.embed_dim)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads)(h)

            # Upsample
            if i > 0:
                h = nn.ConvTranspose(self.channels[i - 1], (4, 4), strides=(2, 2), padding='SAME')(h)

        # 6. Final Output Projection
        h = nn.GroupNorm(num_groups=_pick_gn_groups(h.shape[-1]))(h)
        h = act(h)
        out = nn.Conv(self.z_channels, (3, 3), padding='SAME',
                      kernel_init=nn.initializers.zeros)(h)
        return out