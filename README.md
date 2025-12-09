# 🏥 SuperDiff: Single Conditional LDM for Chest X-Rays (CXR)

This repository contains the implementation for a Latent Diffusion Model (LDM) that is trained on a **single model** capable of generating both Normal and Tuberculosis (TB) Chest X-rays (CXR) based on a **conditional input** (e.g., class label).

## 💡 Key Design & Goal

* **Architecture:** A single UNet-based LDM (`models/cxr_unet.py`) is trained across all TB and Normal samples.
* **Conditioning:** The model is conditioned on the class label (TB or Normal) to guide the generation process.
* **Shared VAE:** The LDM operates on a shared latent space created by a **single VAE** (`models/ae_kl.py`) trained on the combined Normal and TB dataset.
* **Goal:** Investigate the model's ability to learn and separate distinct generative modes within a single, unified latent space.

## 🚀 Training and Launching

* **Main Run Script:** `run/ldm.py`
* **Example Launchers:** See `launchers/single_runs/ldm/train_ldm_conditional.sh`
* **W&B Sweeps:** Configuration for hyperparameter and architectural sweeps is located in `sweeps/ldm/hyperparams/` and `sweeps/ldm/z_channels/`.