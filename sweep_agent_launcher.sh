#!/usr/bin/env bash
set -euo pipefail

# This script is called by the 'wandb agent' to process sweep parameters,
# set environment variables, and submit the SLURM job.

# --- Fixed Defaults ---
export ENV_NAME="jax115"
# Set TASK and DISEASE to ensure conditional training is used
export TASK="ALL_CXR"
export DISEASE="-1"
export PYFILE="train_ldm.py" # Assuming train_ldm.py is the executable

# --- Parse W&B agent args and export as env vars ---
for arg in "$@"; do
    # Skip the program argument which is passed by the agent
    if [[ "$arg" == --program* ]]; then
        continue
    fi
    VAR_NAME=$(echo "$arg" | cut -d'=' -f1 | sed 's/--//' | tr '[:lower:]' '[:upper:]')
    VAR_VALUE="${arg#*=}"
    export "$VAR_NAME"="$VAR_VALUE"
done

# --- Ensure Fixed Conditional Parameters are set if not swept ---
export PROB_UNCOND="${PROB_UNCOND:-0.1}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
export USE_EMA="${USE_EMA:-1}"
export EMA_DECAY="${EMA_DECAY:-0.999}"

# --- Determine Training Mode from Sweep Parameter ---
TRAINING_MODE="full_train"
if [[ "${OVERFIT_K}" -eq 8 ]]; then
    TRAINING_MODE="overfit_8"
elif [[ "${OVERFIT_K}" -eq 16 ]]; then
    TRAINING_MODE="overfit_16"
elif [[ "${OVERFIT_K}" -gt 0 ]]; then
    TRAINING_MODE="overfit_${OVERFIT_K}"
fi

# --- Set Run Name and Tags ---
# Ensure tags reflect the conditional nature of the sweep
TAGS="sweep,ldm,conditional,${TRAINING_MODE}"
[[ -n "${LDM_BASE_CH:-}" ]] && TAGS+=",ch${LDM_BASE_CH}"
[[ -n "${LR:-}" ]] && TAGS+=",lr${LR}"

export WANDB_TAGS="$TAGS"
export WANDB_PROJECT="cxr-conditional-ldm" # Matches your desired project
export RUN_NAME="sweep-${TRAINING_MODE}-${LDM_BASE_CH:-0}-lr${LR:-0}_$(date +%s)"

# --- Submit Job to SLURM ---
# This command relies on the SLURM script train_ldm.slurm being accessible.
echo "LAUNCHING SWEEP RUN: ${RUN_NAME} with TAGS: ${WANDB_TAGS}"

sbatch slurm/train_ldm.slurm # Assumes train_ldm.slurm is in the current working directory