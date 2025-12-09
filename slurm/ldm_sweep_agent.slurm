#!/bin/bash
# slurm/ldm_sweep_agent.sbatch
# Usage: sbatch ldm_sweep_agent.sbatch <sweep_id> <partition_name>

# Arguments
SWEEP_ID=$1
PARTITION_NAME=$2

# SLURM Directives 
#SBATCH --job-name=cond-ldm
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=${PARTITION_NAME} # bigbatch, stampede, biggpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# W&B Agent Settings
WANDB_PROJECT="cxr-ldm"
WANDB_ENTITY="your_wandb_entity" # Replace with your W&B entity

# --- Environment Setup ---
# Check if arguments were passed
if [ -z "$SWEEP_ID" ] || [ -z "$PARTITION_NAME" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch ldm_sweep_agent.sbatch <sweep_id> <partition_name>"
    exit 1
fi

# Load your environment (e.g., Anaconda or module system)
# module load ...
# conda activate superdiff-env
# export PYTHONUNBUFFERED=1 # Important for real-time cluster output

echo "Starting W&B Agent for Sweep ID: ${SWEEP_ID} on Partition: ${PARTITION_NAME}"

# The core command: tell the agent how many runs to execute (e.g., 10 runs per SLURM job)
wandb agent --count 10 "${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"

echo "W&B Agent finished."