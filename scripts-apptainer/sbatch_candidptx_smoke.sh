#!/bin/bash
# One-cycle smoke of task set (f): localization then segmentation, two A100s.
# Writes to /scratch/$USER/FoundationX/candidptx_tasksets/smoke_f, not production f.
#
#   sbatch scripts-apptainer/sbatch_candidptx_smoke.sh

#SBATCH --job-name=candidptx_smoke_f
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00
#SBATCH -A grp_jliang12
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:a100:2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cd "$(dirname "$0")/.."

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export LOGFILE="${LOGFILE:-${SCRATCH:-/scratch/$USER}/FoundationX/candidptx_tasksets/smoke_f}"

exec ./scripts-apptainer/run_apptainer_candidptx_taskset.sh smoke_f
