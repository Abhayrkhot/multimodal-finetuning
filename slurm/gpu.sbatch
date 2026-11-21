#!/usr/bin/env bash
#SBATCH --job-name=mmft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/mmft/logs/%x-%j.out
# One stage script on one GPU, after the test suite passes on that node. Submit from the code
# directory (mxe sync prints it), which pins the job to that commit:
#   cd /scratch/$USER/mmft/code/<sha>
#   sbatch --gres=gpu:h200:1 slurm/gpu.sbatch scripts/stage_train.sh
#   sbatch --gres=gpu:a100:1 --constraint=a100@80g slurm/gpu.sbatch scripts/stage_bench.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
(cd "$MMFT_SRC" && python -m pytest -q -p no:cacheprovider)
cd "$MMFT_HOME"
bash "$MMFT_SRC/$1" "${@:2}"
