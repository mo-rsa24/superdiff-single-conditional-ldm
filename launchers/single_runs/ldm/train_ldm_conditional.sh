#!/usr/bin/env bash
set -euo pipefail

# --- Defaults ---
export TASK="All_CXR"
export ENV_NAME="jax115"
export IMG_SIZE="128"
export TRAINING_MODE="${1:-full_train}"

# ❗ CRITICAL: Set DISEASE to -1 to include ALL classes (TB and Normal) for conditional training
export DISEASE="-1"

# --- Hyperparameter Defaults ---
export LR="1e-4"
export WEIGHT_DECAY="0.05"
export LDM_BASE_CH="64"
export GRAD_CLIP="1.0"
export BATCH_PER_DEVICE="16"
export EPOCHS="300"
export LOG_EVERY="100"
export SAMPLE_EVERY="10"
export SAMPLE_BATCH_SIZE="16"
export LDM_CH_MULTS="1,2,4"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16"
export WANDB="1"

# ❗ CRITICAL: Probability of unconditional generation (label dropping) for CFG
export PROB_UNCOND="0.1"

# --- Shared VAE and Scale Factor (Update as needed) ---
export AE_CKPT_PATH="runs/unified-ae-128-z4_z4_20251008-161725/20251008-170121/ckpts/last.flax"
export AE_CONFIG_PATH="runs/unified-ae-128-z4_z4_20251008-161725/20251008-170121/run_meta.json"
export LATENT_SCALE_FACTOR="0.994534"

# --- SLURM Defaults ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-${TASK,,}-conditional"

# --- EMA Configuration ---
export USE_EMA="1"
export EMA_DECAY="0.999"

# --- Argument Parsing ---
OTHER_ARGS=()
shift

while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)          export SLURM_PARTITION="$2"; shift 2 ;;
    --job-name)           export SLURM_JOB_NAME="$2"; shift 2 ;;
    --lr)                 export LR="$2"; shift 2 ;;
    --weight_decay)       export WEIGHT_DECAY="$2"; shift 2 ;;
    --ldm_base_ch)        export LDM_BASE_CH="$2"; shift 2 ;;
    --grad_clip)          export GRAD_CLIP="$2"; shift 2 ;;
    --epochs)             export EPOCHS="$2"; shift 2 ;;
    --batch_per_device)   export BATCH_PER_DEVICE="$2"; shift 2 ;;
    --ldm_ch_mults)       export LDM_CH_MULTS="$2"; shift 2 ;;
    --ldm_num_res_blocks) export LDM_NUM_RES_BLOCKS="$2"; shift 2 ;;
    --ldm_attn_res)       export LDM_ATTN_RES="$2"; shift 2 ;;
    --log_every)          export LOG_EVERY="$2"; shift 2 ;;
    --sample_every)       export SAMPLE_EVERY="$2"; shift 2 ;;
    --sample_batch_size)  export SAMPLE_BATCH_SIZE="$2"; shift 2 ;;
    --latent_scale_factor) export LATENT_SCALE_FACTOR="$2"; shift 2 ;;
    *)                    OTHER_ARGS+=("$1"); shift ;;
  esac
done

export RUN_NAME="${SLURM_JOB_NAME}_lr${LR}_wd${WEIGHT_DECAY}_ch${LDM_BASE_CH}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-ldm-composition"
export WANDB_TAGS="ldm,${TASK,,},conditional"

# --- Prettier Submit Message ---
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
kv "Training Mode" "Conditional (Class Filter: ${DISEASE})"
kv "CFG Prob Uncond" "${PROB_UNCOND}"
printf "\n"
kv "⚙️ Learning Rate" "${LR}"
kv "Epochs" "${EPOCHS}"
kv "Batch Size" "${BATCH_PER_DEVICE}"
rule

# Pass the prob_uncond argument here
sbatch --partition="$SLURM_PARTITION" --job-name="$SLURM_JOB_NAME" \
  slurm_scripts/cxr_ldm.slurm \
  --latent_scale_factor "$LATENT_SCALE_FACTOR" \
  --prob_uncond "$PROB_UNCOND"

echo "✅ Job successfully submitted!"