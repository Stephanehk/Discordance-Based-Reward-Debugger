#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=wipe_rollout
#SBATCH --output=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.out
#SBATCH --error=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.err

# Roll out one round's data-collection policies in parallel across CPU cores.
# robosuite/MuJoCo renders offscreen (CAPTURE_FRAMES=True), so each worker needs EGL.
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,WS=wipe,ROUND=2,NPER=20,WORKERS=16 \
#          slurm_scripts/wipe_rollout_round.sh
WS=${WS:-wipe}
ROUND=${ROUND:-1}
NPER=${NPER:-12}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh
cd /next/u/stephhk/automated_reward_repairer
conda activate "$(cat workspaces/${WS}/.conda_env 2>/dev/null || echo llm)"

# The `next` compute nodes have no EGL device display, so render through the
# software rasterizer instead (~2x slower per step, but it works everywhere).
export MUJOCO_GL=osmesa

echo "[slurm] host=$(hostname) ws=$WS round=$ROUND rollouts/policy=$NPER workers=$WORKERS"
python -m runtime.run_round_parallel \
    --workspace "workspaces/${WS}" \
    --round "${ROUND}" \
    --rollouts-per-policy "${NPER}" \
    --workers "${WORKERS}"
