import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import jax
import jax.numpy as jnp
import argparse
from typing import Tuple, Any

# Local imports
from datasets.chestxray import ChestXrayDataset
from models.ae_kl import AutoencoderKL


def setup_dataset_and_loader(args: argparse.Namespace, base_ds_cls=ChestXrayDataset) -> Tuple[
    DataLoader, torch.utils.data.Dataset, int]:
    """
    Sets up the dataset and DataLoader based on command-line arguments.
    Handles overfit_one and overfit_k modes.

    Returns: DataLoader, base_dataset (for z0 precompute), global_batch_size
    """
    filter_val = None if args.class_filter == -1 else args.class_filter

    # Use the base dataset class for initialization
    base_ds = base_ds_cls(root_dir=args.data_root, task=args.task, split=args.split, img_size=args.img_size,
                          class_filter=filter_val)

    print(f"Dataset Size: {len(base_ds)}. Class Filter: {filter_val} (None means all).")
    global_batch_size = args.batch_per_device * jax.local_device_count()

    if args.overfit_one:
        ds = Subset(base_ds, [0])
        ds = ConcatDataset([ds] * args.repeat_len)
    elif args.overfit_k > 0:
        ds = Subset(base_ds, list(range(min(args.overfit_k, len(base_ds)))))
    else:
        ds = base_ds

    if args.overfit_one:
        print("INFO: Overfitting on one sample. Disabling data loader workers and shuffle.")
        loader_kwargs = dict(batch_size=global_batch_size, shuffle=False, num_workers=0, drop_last=True,
                             pin_memory=True)
    else:
        loader_kwargs = dict(batch_size=global_batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    loader = DataLoader(ds, **loader_kwargs)

    return loader, base_ds, global_batch_size


def precompute_overfit_z0(ae_model: AutoencoderKL, unrep_ae_params: Any, base_ds: torch.utils.data.Dataset,
                          args: argparse.Namespace, global_bs: int) -> jnp.ndarray:
    """
    Computes and shards the fixed latent vector z0 for the overfit_one mode.
    """
    # Only load the first sample
    one_loader = DataLoader(Subset(base_ds, [0]), batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    (x0, _), = list(one_loader)
    x0 = jnp.asarray(x0.numpy()).transpose(0, 2, 3, 1)  # NCHW -> NHWC
    x0 = (x0 + 1.0) / 2.0  # [-1,1] -> [0,1]

    # Encode and take the mode (mean)
    posterior0 = ae_model.apply({'params': unrep_ae_params}, x0, method=ae_model.encode, train=False)
    z0 = posterior0.mode() * args.latent_scale_factor  # fixed latent

    # Tile and shard z0 for pmap
    z0_tiled = jnp.tile(z0, (global_bs, 1, 1, 1))
    precomputed_z0 = z0_tiled.reshape((jax.local_device_count(), -1) + z0.shape[1:])

    print("Precomputed z0 for overfit-one:", precomputed_z0.shape)
    return precomputed_z0