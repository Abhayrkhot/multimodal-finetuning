#!/usr/bin/env bash
#SBATCH --job-name=mmft-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/mmft/logs/%x-%j.out
# One CPU stage (download, merge), submitted from a code directory like gpu.sbatch:
#   sbatch slurm/cpu.sbatch scripts/stage_merge.sh v7w_r16_vis
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$MMFT_HOME"
bash "$MMFT_SRC/$1" "${@:2}"
