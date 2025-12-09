Based on the **"Guide to Hyperparameter Experiments"** and the **"Primer on Training"**, here are the optimal run configurations.

The "Safe & Stable" configuration (Experiment 1) is recommended as the optimal starting point to avoid mode collapse.

### **The Optimal "Safe & Stable" Hyperparameters**

Use these flags for all three scenarios to ensure stability:

  * **Learning Rate:** `--lr 3e-5`
  * **Weight Decay:** `--weight_decay 0.01`
  * **Model Capacity:** `--ldm_base_ch 96`

-----

### **1. Overfit One (Single Sample)**

This mode is for verifying that the model can memorize a single input (sanity check).

  * **Mode:** `overfit_one`
  * **Presets:** 1000 Epochs, Sample every 5 epochs.
  * **Flag:** `--overfit_one`

<!-- end list -->

```bash
./launchers/single_runs/ldm/train_ldm_conditional.sh overfit_one \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --overfit_one \
  --epochs 1000 \
  --sample_every 5
```

### **2. Overfit 8 (Tiny Subset)**

This tests generalization on a very small subset (similar to `overfit_16` in the guide).

  * **Mode:** `overfit_8` (Custom)
  * **Presets:** 1500 Epochs (needs more time to converge on multiple items).
  * **Flag:** `--overfit_k 8`

<!-- end list -->

```bash
./launchers/single_runs/ldm/train_ldm_conditional.sh overfit_8 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --overfit_k 8 \
  --epochs 1500 \
  --sample_every 10
```

### **3. Full Train (Entire Dataset)**

This is the standard training run.

  * **Mode:** `full_train`
  * **Presets:** 1000 Epochs (or fewer if dataset is huge), Sample every 10 epochs.
  * **Flag:** None (defaults to full dataset).

<!-- end list -->

```bash
./launchers/single_runs/ldm/train_ldm_conditional.sh full_train \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --epochs 1000 \
  --sample_every 10
```