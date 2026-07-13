#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next1,next2
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=mw_rollout
#SBATCH --output=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.out
#SBATCH --error=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.err

# Roll out one round's data-collection policies in parallel across CPU cores.
# MuJoCo renders offscreen (CAPTURE_FRAMES=True), so each worker needs EGL.
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,WS=metaworld_drawer_open,ROUND=2,NPER=25,WORKERS=16 \
#          slurm_scripts/metaworld_rollout_round.sh
WS=${WS:-metaworld_drawer_open}
ROUND=${ROUND:-1}
NPER=${NPER:-25}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh
cd /next/u/stephhk/automated_reward_repairer
conda activate "$(cat workspaces/${WS}/.conda_env 2>/dev/null || echo llm)"

export MUJOCO_GL=egl

echo "[slurm] host=$(hostname) ws=$WS round=$ROUND rollouts/policy=$NPER workers=$WORKERS"
python -m runtime.run_round_parallel \
    --workspace "workspaces/${WS}" \
    --round "${ROUND}" \
    --rollouts-per-policy "${NPER}" \
    --workers "${WORKERS}"
