#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next7
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --job-name=pandemic_rollout
#SBATCH --output=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.out
#SBATCH --error=/next/u/stephhk/automated_reward_repairer/run_logs/slurmjob_%j.err
#SBATCH --mail-user=stephhk@stanford.edu
#SBATCH --mail-type=END,FAIL

# Roll out one round's data-collection policies in parallel across CPU cores.
# These policies are lightweight hand-coded heuristics (no GPU / no neural net),
# so the work is pure CPU and fans out cleanly over the cores requested above.
#
# Override any of these at submit time, e.g.:
#   sbatch --export=ALL,WS=pandemic,ROUND=2,NPER=3,WORKERS=24 \
#          slurm_scripts/pandemic_rollout_round.sh
WS=${WS:-pandemic}
ROUND=${ROUND:-1}
NPER=${NPER:-3}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-24}}

source /next/u/stephhk/miniconda3/etc/profile.d/conda.sh
conda activate llm
cd /next/u/stephhk/automated_reward_repairer

echo "[slurm] host=$(hostname) ws=$WS round=$ROUND rollouts/policy=$NPER workers=$WORKERS"
python -m runtime.run_round_parallel \
    --workspace "workspaces/${WS}" \
    --round "${ROUND}" \
    --rollouts-per-policy "${NPER}" \
    --workers "${WORKERS}"
