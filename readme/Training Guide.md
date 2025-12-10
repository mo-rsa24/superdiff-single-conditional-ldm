# Training Guide for Single Conditional LDM (CXR)

This guide provides a structured, pedagogical approach to launching and monitoring training experiments for your Conditional Latent Diffusion Model (LDM). Follow the three phases sequentially to ensure stability and reliable results.

-----

## The Crucial Order of Operations (3 Phases)

You **must** complete these phases in order before starting any main LDM training or sweep.

### Phase 1: Train the Autoencoder (VAE)

The VAE creates the shared latent space that the LDM operates on. First, train an autoencoder for your dataset. The script takes `z_channels` as an argument.

| Goal | Description | Example Command |
| :--- | :--- | :--- |
| **Train VAE** | Train a VAE for a specific task (e.g., TB) with a given number of latent channels (`z_channels`). | `./launchers/single_runs/vae/train_ae_tb.sh 1` |

### Phase 2: Calculate Latent Scale Factor ⚠️ (Crucial Step)

After each VAE model finishes training, you **must** calculate its **latent scale factor**. This value is essential for stable LDM training and must be updated in your LDM launcher scripts.

**How to Run:**

```bash
# Set PYTHONPATH to the root of your repository
export PYTHONPATH="$(pwd)"

python scripts/find_latent_scale.py \
    --ae_config_path "runs/ae_tb_z1_.../run_meta.json" \
    --ae_ckpt_path "runs/ae_tb_z1_.../ckpts/last.flax" \
    --data_root "/datasets/mmolefe/cleaned" \
    --task "TB"
```

**Note:** Repeat this step for every VAE you train.

-----

## Phase 3: LDM Training (Single Runs)

The LDM launchers use sensible presets for different training modes and allow full customization via command-line arguments.

### 3.1 Recommended Presets by Mode

These presets balance performance, logging verbosity, and sampling frequency for stability monitoring.

| Parameter | `full_train` (Generalization) | `overfit_one` (Sanity Check) | `overfit_16` / `overfit_32` (Tiny Subset) | Rationale                                                                                                                                                                           |
| :--- | :--- | :--- | :--- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **BATCH\_PER\_DEVICE** | 16 | 1 | 16 | Use max batch size for main training; small batch size for single-sample mode.                                                                                                      |
| **EPOCHS** | 1000 | 1000 | 1500 | More epochs needed for generalization/tiny subset training.                                                                                                                         |
| **LOG\_EVERY (Steps)** | 100 | 10 | 25-50 | Log less frequently for long runs, more frequently for rapid checks.                                                                                                                |
| **SAMPLE\_EVERY (Epochs)** | 10 | 5 | 10 | **Sampling is expensive.** More frequent for fast convergence (`overfit_one`); less frequent for slow, full training to avoid slowing down the training loop.                       |
| **USE\_WANDB** | Yes | No | Yes | Disable W\&B for quick, ephemeral sanity checks (`overfit_one`).                                                                                                                    |
| **Num Sampling Steps** | **500 (Default)** | **500 (Default)** | **500 (Default)** | This is the default for Ancestral Sampling for good image quality. **Sweet Spot:** You can reduce this to **200-300** for faster checking, but only use **500+** for final results. |

#### Sampling Steps Sweet Spot (`--num_sampling_steps`)

The number of steps dictates sample quality and generation time. The default is 500 steps.

| Goal | Recommended Steps | Impact |
| :--- | :--- | :--- |
| **Quick Checks/Sweeps** | **200 - 300** | Provides a good proxy for mode collapse metrics (`pairwise_mse_mean`) at a much faster generation speed. |
| **Final Results** | **500 or more** | Required for highest quality and most stable image generation. |


### 3.2 The Optimal "Safe & Stable" Hyperparameters

To avoid instability (e.g., exploding gradients, mode collapse), use the recommended base configuration for all new experiments.

| Parameter | Recommended Stable Value | Command Flag |
| :--- | :--- | :--- |
| **Learning Rate (LR)** | `3e-5` | `--lr` |
| **Weight Decay (WD)** | `0.01` | `--weight_decay` |
| **Model Capacity** | `96` | `--ldm_base_ch` |

-----

## 4\. W\&B Hyperparameter Sweeps

Sweeps automate running many experiments to find optimal models and hyperparameters.

**Objective:** Maximize **`sample_diversity/pairwise_mse_mean`** to avoid mode collapse.

### Step 1: Initialize the Sweep

Use the configuration file to initialize the sweep and obtain a `<SWEEP_ID>`.

```bash
wandb sweep config/sweeps/conditional_ldm_sweep.yaml \
  --entity prime_lab \
  --project cxr-conditional-ldm
```

### Step 2: Run the Agent

Submit the agent script to your cluster's job scheduler (SLURM) using the Sweep ID. The agent will then use `sweep_agent_launcher.sh` to submit individual training jobs.

```bash
# To run on the 'bigbatch' partition:
sbatch slurm/ldm_sweep_agent.slurm <SWEEP_ID> bigbatch

# To run on the 'biggpu' partition:
sbatch slurm/ldm_sweep_agent.slurm <SWEEP_ID> biggpu
```

-----

## 5\. Optimal Run Configurations (Tailored Examples)

Based on the **"Guide to Hyperparameter Experiments"** and the stable training principles, the **"Safe & Stable"** configuration is the optimal starting point to avoid **mode collapse**.

### **The Optimal "Safe & Stable" Hyperparameters**

Use these flags for all stable experiments. These override the environment defaults set in `train_ldm_conditional.sh`.

| Parameter | Recommended Stable Value | Command Flag |
| :--- | :--- | :--- |
| **Learning Rate** | `3e-5` | `--lr` |
| **Weight Decay** | `0.01` | `--weight_decay` |
| **Model Capacity** | `96` | `--ldm_base_ch` |

-----

### 5.1 Overfit One (Single Sample)

This mode verifies the model's capacity to memorize a single input, which is a crucial **sanity check**.

  * **Mode:** `overfit_one` (Passed as the first argument to the launcher)
  * **Presets:** 1000 Epochs, Sample every 5 epochs.
  * **Data Flag:** `--overfit_one` (Passed internally by the launcher logic)

<!-- end list -->

```bash
# Execute the launcher from the project root directory
./train_ldm_conditional.sh overfit_one \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --epochs 1000 \
  --sample_every 50 \
  --log_every 50 \
  --num_sampling_steps 300 \
  --partition bigbatch
```

### 5.2 Overfit 8 (Tiny Subset)

This tests generalization on a very small subset of $K=8$ samples, requiring more time to converge on multiple data points.

  * **Mode:** `overfit_8` (The argument is used to set the `--overfit_k 8` flag internally)
  * **Presets:** 1500 Epochs, Sample every 10 epochs.
  * **Data Flag:** `--overfit_k 8`

<!-- end list -->

```bash
# Execute the launcher from the project root directory
./train_ldm_conditional.sh overfit_8 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --epochs 1500 \
  --sample_every 10 \
  --num_sampling_steps 500 \
  --partition bigbatch
```
#### Targeted Sweeps 

To run focused stability sweeps on the `stampede|bigbatch|biggpu` partition, ensure your YAML config has the correct `overfit_k` value (e.g., `[16]` or `[32]`) and launch the agent on `stampede`.

**Group 1: Overfit 16**

  * **Config:** Set `run_cap: 5` in `config/sweeps/conditional_ldm_sweep.yaml` to enforce a hard budget.
  * **Launch Strategy:** Deploy **5 nodes**, where each agent executes **1 run** ($5 \text{ nodes} \times 1 \text{ run} = 5 \text{ total runs}$).
  * **Command:**
    ```bash
    # Initialize Sweep
    wandb sweep config/sweeps/conditional_ldm_sweep.yaml --project cxr-conditional-ldm 

    # Launch Swarm (Usage: ... <ID> <PARTITION> <K=16> <NODES=5> <RUNS_PER_NODE=1>)
    chmod +x launch_sweep_swarm.sh
    ./launch_sweep_swarm.sh <SWEEP_ID> stampede 16 5 1
    ```

**Group 2: Overfit 32**

  * **Config:** Set `run_cap: 5` in `config/sweeps/conditional_ldm_sweep.yaml`.
  * **Launch Strategy:** Deploy **5 nodes**, where each agent executes **1 run**.
  * **Command:**
    ```bash
    # Initialize Sweep (or use existing ID)
    wandb sweep config/sweeps/conditional_ldm_sweep.yaml --project cxr-conditional-ldm 

    # Launch Swarm (Usage: ... <ID> <PARTITION> <K=32> <NODES=5> <RUNS_PER_NODE=1>)
    chmod +x launch_sweep_swarm.sh
    ./launch_sweep_swarm.sh <SWEEP_ID> stampede 32 5 1
    ```
    
### 5.3 Full Train (Entire Dataset)

This is the standard training run on the entire conditional dataset.

  * **Mode:** `full_train`
  * **Presets:** 1000 Epochs, Sample every 10 epochs.
  * **Data Flag:** None (defaults to full dataset logic)

<!-- end list -->

```bash
# Execute the launcher from the project root directory
./train_ldm_conditional.sh full_train \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --ldm_base_ch 96 \
  --epochs 1000 \
  --sample_every 10 \
  --num_sampling_steps 500 \
  --partition bigbatch
```

**Group 3: Full Training Mode**

  * **Config:** Set `run_cap: 5` in `config/sweeps/conditional_ldm_sweep.yaml`.
  * **Launch Strategy:** Deploy **5 nodes**, where each agent executes **1 run**.
  * **Command:**
    ```bash
    # Initialize Sweep (or use existing ID)
    wandb sweep config/sweeps/conditional_ldm_sweep.yaml --project cxr-conditional-ldm 

    # Launch Swarm (Usage: ... <ID> <PARTITION> <K=0> <NODES=5> <RUNS_PER_NODE=1>)
    # Note: K=0 triggers full dataset training
    chmod +x launch_sweep_swarm.sh
    ./launch_sweep_swarm.sh <SWEEP_ID> stampede 0 5 1
    ```

#### Verifying Which Partition Array Jobs Are Running On
`1. The "Better Squeue" Alias`

Run this command in your terminal to create the `sq` alias for this session:

```bash
alias sq='squeue -u $USER -o "%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R"'
```

Now, simply type `sq` to check your jobs.

  * **`%.18i`**: Allocates space for long array IDs (e.g., `170789_[1-5]`).
  * **`%.12P`**: Shows the **Partition** clearly so you can verify if it says `stampede` or `bigbatch`.
  * **`%R`**: Shows the "Reason" (why it's pending) or the "NodeList" (where it's running).

To make it permanent, add that line to the bottom of your `~/.bashrc` file:

```bash
echo "alias sq='squeue -u $USER -o \"%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R\"'" >> ~/.bashrc
source ~/.bashrc
```

-----

### 2\. Deep Dive with `scontrol`

If you want to be 100% sure about the configuration of a specific job (even one that is pending), use `scontrol`.

For an array job like `170789`, you can check the first task:

```bash
scontrol show job 170789_1
```

Look for the `Partition=` line in the output. If you applied the fix from my previous message, it should now explicitly say `Partition=stampede`.

## Appendix: Interpreting LDM Training Metrics

Monitoring key metrics on your W\&B dashboard is critical for diagnosing instability.

| Metric | Good Sign | Bad Sign (Diagnosis) |
| :--- | :--- | :--- |
| **loss** | Steadily decreases, then plateaus. | **NaN/inf:** Training collapsed, likely due to a too-high learning rate (exploding gradients). |
| **grad\_norm** | Stable, non-zero value, decreases over time. | **Very high (\> 100):** Exploding gradients. |
| **pairwise\_mse\_mean** | **High and stable** (e.g., \> 0.1 for normalized latents). | **Very low (\< 0.01)** and trending toward zero (Mode collapse, sampler produces identical output). |



This guide explains how to interpret the key statistics logged during the training and sampling of your Latent Diffusion Model. Understanding these numbers is crucial for diagnosing issues like training instability, slow convergence, and mode collapse.

---

1. `train/metrics`

These metrics are logged at every `log_every` step and give you a real-time view of the model's learning process.

| Metric            | What it Measures                                                                                                         | Good Sign                                                                                                                | Bad Sign (and What it Means)                                                                                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **loss**          | The value of the denoising loss function. It measures how well the model predicts the noise that was added to an image.  | The loss should **steadily decrease** over time, especially in the early stages of training. It will eventually plateau. | - **NaN** or **inf**: Training has collapsed, likely due to a learning rate that is too high (exploding gradients). <br> - **Stays high / fluctuates wildly**: Model is not learning effectively. Could be caused by too low learning rate, poor data quality, or architectural issues.       |
| **grad_norm**     | The overall magnitude (L2 norm) of the gradients before they are clipped. It indicates how large the weight updates are. | A stable, non-zero value. Normal for it to be higher at the start of training and decrease over time.                    | - **Very high (> 100)**: Exploding gradients. Updates are too large → instability. `grad_clip` prevents a crash, but the root cause (likely high learning rate) must be fixed. <br> - **Very low (< 1e-6)**: Vanishing gradients. Model isn’t learning; gradients too weak to update weights. |
| **learning_rate** | The current learning rate being used by the optimizer.                                                                   | Should match the value you set. If using a scheduler (e.g., warmup/decay), you’ll see it change over time.               | N/A                                                                                                                                                                                                                                                                                           |

---

2. `sample_diversity/metrics`

This block is the most important diagnostic for **mode collapse**. It is calculated during sampling by measuring the pairwise Mean Squared Error (MSE) between all generated latents in a batch.

| Metric                | What it Measures                                                                 | Good Sign (Healthy Model)                                                                                                      | Bad Sign (Mode Collapse)                                                                                                                   |
| --------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **pairwise_mse_mean** | Average pixel-wise difference between all pairs of generated samples in a batch. | A high and stable value. For normalized latents, > **0.1** is good. Higher = more varied outputs from different noise vectors. | Very low (< **0.01**) and trending toward zero. Clear sign of mode collapse → sampler produces the same output regardless of random noise. |
| **pairwise_mse_std**  | Standard deviation of the pairwise MSE values.                                   | Healthy, non-zero value.                                                                                                       | Trends toward zero during mode collapse.                                                                                                   |
| **pairwise_mse_min**  | The smallest MSE between any two samples.                                        | Should be significantly > 0.                                                                                                   | Extremely close to zero.                                                                                                                   |
| **pairwise_mse_max**  | The largest MSE between any two samples.                                         | Should be a high value.                                                                                                        | Becomes very small → no meaningful diversity between samples.                                                                              |

---

3. `final_latent_stats`

These stats describe the distribution of the final latent vectors (`z`) produced by the sampler, right before they are passed to the VAE decoder.

| Metric        | What it Measures                                                  | Good Sign                                                                                                   | Bad Sign (and What it Means)                                                                              |
| ------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **mean**      | Average value across all dimensions of latent vectors in a batch. | Close to **0.0**.                                                                                           | Large positive/negative shift → latents not centered. Can lead to dull or discolored images.              |
| **std**       | Standard deviation of the latent vectors.                         | Close to target std = **1.0 / latent_scale_factor**. <br> Example: scale factor = 1.86 → target std ≈ 0.53. | - **Very low**: Latents too similar (mode collapse). <br> - **Very high**: Sampler unstable or too noisy. |
| **min / max** | Minimum and maximum values in the latent vectors.                 | Within reasonable range (e.g., [-3.0, 3.0]).                                                                | Extreme values (±20) suggest instability in sampling → artifacts or blank images.                         |

---
