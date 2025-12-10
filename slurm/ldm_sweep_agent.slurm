#!/bin/bash
# slurm/ldm_sweep_agent.slurm
# Usage: sbatch slurm/ldm_sweep_agent.slurm <sweep_id> <partition_name> <overfit_k>

# Arguments
SWEEP_ID=$1
PARTITION_NAME=$2
OVERFIT_K_ARG=${3:-0} # Default to 0 (Full Train) if not provided

# Export so the underlying script inherits it
export OVERFIT_K="$OVERFIT_K_ARG"

# SLURM Directives
#SBATCH --job-name=cond-ldm-sweep
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=${PARTITION_NAME}
#SBATCH --exclude=mscluster65,mscluster76,mscluster59
#SBATCH --time=72:00:00

# --- Pretty printing helpers (Borrowed from train_ldm.slurm) ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m');
RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }

# --- Argument Check ---
if [ -z "$SWEEP_ID" ] || [ -z "$PARTITION_NAME" ]; then
    printf "${RED}!! Error: Missing arguments.${RST}\n"
    echo "Usage: sbatch slurm/ldm_sweep_agent.slurm <sweep_id> <partition_name> <overfit_k>"
    exit 1
fi

# --- Environment Setup (CRITICAL) ---
banner "Environment Setup"
export ENV_NAME="jax115"
export WORKDIR="$PWD"

# 1. Setup Python Path and JAX
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3

# 2. Activate Environment
source ~/.bashrc
mamba activate "${ENV_NAME:?Must set ENV_NAME}"

kv "Work Dir" "$WORKDIR"
kv "Environment" "$ENV_NAME"
kv "Python Path" "$PYTHONPATH"

# --- Run Info ---
banner "Sweep Agent Info"
kv "Sweep ID"       "$SWEEP_ID"
kv "Partition"      "$PARTITION_NAME"
kv "Overfit K"      "$OVERFIT_K"
kv "W&B Project"    "cxr-conditional-ldm"

# --- Start Agent ---
banner "Starting W&B Agent"
# The agent will now inherit all the exports above and run train_ldm.py successfully
wandb agent --count 10 "prime_lab/cxr-conditional-ldm/${SWEEP_ID}"

banner "Done"