#!/usr/bin/env bash
#SBATCH --job-name=mmft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/mmft/logs/%x-%j.out
# One stage script on one GPU, after the test suite passes on that node:
#   sbatch --gres=gpu:h200:1 slurm/gpu.sbatch scripts/stage_train.sh
#   sbatch --gres=gpu:a100:1 --constraint=a100@80g slurm/gpu.sbatch scripts/stage_bench.sh
set -euo pipefail
source /scratch/$USER/mmft/src/slurm/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
(cd "$MMFT_HOME/src" && python -m pytest -q -p no:cacheprovider)
cd "$MMFT_HOME"
bash "src/$1" "${@:2}"
