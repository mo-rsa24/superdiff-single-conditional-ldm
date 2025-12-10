#!/bin/bash
# Usage: ./launch_sweep_swarm.sh <SWEEP_ID> <PARTITION> <OVERFIT_K> <NUM_NODES>

SWEEP_ID=$1
PARTITION=$2
OVERFIT_K=${3:-0}
NUM_NODES=${4:-1}
AGENT_COUNT=${5:-1}

if [ -z "$SWEEP_ID" ] || [ -z "$PARTITION" ]; then
    echo "Usage: ./launch_sweep_swarm.sh <SWEEP_ID> <PARTITION> <OVERFIT_K> <NUM_NODES>"
    exit 1
fi

echo "🚀 Launching a swarm of ${NUM_NODES} agents for Sweep: ${SWEEP_ID}"
echo "   Partition: ${PARTITION} | Overfit_K: ${OVERFIT_K} | Jobs/Agent: ${AGENT_COUNT}"

# Submit N separate jobs
sbatch --partition="$PARTITION" \
       --array=1-${NUM_NODES} \
       --job-name="sweep-swarm" \
       slurm/ldm_sweep_agent.slurm "$SWEEP_ID" "$PARTITION" "$OVERFIT_K" "$AGENT_COUNT"

echo "✅ Submitted ${NUM_NODES} agent jobs."