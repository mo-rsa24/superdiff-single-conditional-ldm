#!/usr/bin/env bash
set -euo pipefail

# --- 1. CORE DEFAULTS (Conditional Training) ---
export TASK="ALL_CXR" # Dataset name reflecting combined data
export ENV_NAME="jax115"
export IMG_SIZE="128"
export TRAINING_MODE="${1:-full_train}" # Reads mode (e.g., full_train) from the first argument

# ❗ CRITICAL: Set DISEASE to -1 to include ALL classes (TB and Normal) for conditional training
export DISEASE="-1"

# --- 2. SAFE & STABLE HYPERPARAMETER DEFAULTS ---
# Based on the Guide to Hyperparameter Experiments for LDM Training
export LR="3e-5"            # Safe Learning Rate
export WEIGHT_DECAY="0.01"  # Appropriate Regularization
export LDM_BASE_CH="96"     # Smaller, Stable Model Capacity

# General LDM Architecture & Training Settings
export GRAD_CLIP="1.0"
export BATCH_PER_DEVICE="16"
export EPOCHS="500"
export LOG_EVERY="100"
export SAMPLE_EVERY="10"
export SAMPLE_BATCH_SIZE="16"
export LDM_CH_MULTS="1,2,4"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16"
export WANDB="1"

# --- 3. CFG/EMA CONFIGURATION ---
export PROB_UNCOND="0.1"    # Probability of label dropping for CFG training
export GUIDANCE_SCALE="4.0" # Default CFG scale for sampling
export USE_EMA="1"          # Enable EMA for improved stability
export EMA_DECAY="0.999"

# --- 4. SHARED VAE AND SCALE FACTOR (MUST BE UPDATED) ---
export AE_CKPT_PATH="runs/unified-ae-128-z4_z4_20251008-161725/20251008-170121/ckpts/last.flax"
export AE_CONFIG_PATH="runs/unified-ae-128-z4_z4_20251008-161725/20251008-170121/run_meta.json"
export LATENT_SCALE_FACTOR="0.997570"

# --- 5. SLURM DEFAULTS (Override via command line) ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-${TASK,,}-conditional"

# --- 6. Robust Argument Parsing Loop ---
OTHER_ARGS=()
# Shift away the first argument (training_mode)
if [[ "$#" -gt 0 ]]; then shift; fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)           export SLURM_PARTITION="$2"; shift 2 ;;
    --job-name)            export SLURM_JOB_NAME="$2"; shift 2 ;;
    --lr)                  export LR="$2"; shift 2 ;;
    --weight_decay)        export LR="$2"; shift 2 ;; # Note: If you want to override WD
    --ldm_base_ch)         export LDM_BASE_CH="$2"; shift 2 ;;
    --prob_uncond)         export PROB_UNCOND="$2"; shift 2 ;;
    --guidance_scale)      export GUIDANCE_SCALE="$2"; shift 2 ;;
    --latent_scale_factor) export LATENT_SCALE_FACTOR="$2"; shift 2 ;;
    --epochs)              export EPOCHS="$2"; shift 2 ;;
    --log_every)           export LOG_EVERY="$2"; shift 2 ;;
    *)                     OTHER_ARGS+=("$1"); shift ;; # Save unrecognized arg
  esac
done

# --- 7. Run Naming & W&B Configuration (Uses final values) ---
export RUN_NAME="${SLURM_JOB_NAME}_lr${LR}_ch${LDM_BASE_CH}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-conditional-ldm"
export WANDB_TAGS="ldm,${TASK,,},conditional"

# --- 8. Prettier Submit Message ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
rule(){ printf "${BLU}%.0s" $(seq 1 60); printf "${RST}\n"; }

rule
printf "${BLD}${BLU}🚀 Submitting Conditional LDM Training Job${RST}\n"
rule
kv "SLURM Job Name" "${SLURM_JOB_NAME}"
kv "SLURM Partition" "${SLURM_PARTITION}"
printf "\n"
kv "📊 Dataset Task" "${TASK}"
kv "Training Mode" "Conditional (Filter: ${DISEASE})"
kv "CFG Prob Uncond" "${PROB_UNCOND}"
kv "CFG Guidance Scale" "${GUIDANCE_SCALE}"
printf "\n"
kv "🧠 Model Base CH" "${LDM_BASE_CH}"
kv "EMA Enabled" "${USE_EMA}"
printf "\n"
kv "⚙️ Learning Rate" "${LR}"
kv "Weight Decay" "${WEIGHT_DECAY}"
kv "Epochs" "${EPOCHS}"
rule

# --- 9. Submit to SLURM ---
# The train_ldm.slurm script expects these env vars to be set and will pass them as arguments.
# Assuming your 'train_ldm.slurm' file is accessible as 'slurm_scripts/cxr_ldm.slurm'
sbatch --partition="$SLURM_PARTITION" --job-name="$SLURM_JOB_NAME" \
  slurm_scripts/cxr_ldm.slurm \
  --latent_scale_factor "$LATENT_SCALE_FACTOR" \
  --prob_uncond "$PROB_UNCOND" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --use_ema "$USE_EMA" \
  --ema_decay "$EMA_DECAY" \
  --class_filter "$DISEASE"

echo "✅ Job successfully submitted!"