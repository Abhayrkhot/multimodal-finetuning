#!/usr/bin/env bash
#SBATCH --job-name=mmft-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/mmft/logs/%x-%j.out
# CPU node: venv (torch cu126 wheels), Visual7W + ScienceQA-IMG, LLaVA-1.5-7B weights.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
cd "$MMFT_HOME"
[ -x venv/bin/python ] || python3 -m venv venv
pip="venv/bin/pip --cache-dir $MMFT_HOME/pip-cache"
$pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
$pip install -q -r "$MMFT_SRC/requirements.txt" bitsandbytes==0.50.2 pytest==9.2.0 httpx==0.28.1
python -c "import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__)"
mmft data v7w --root data/visual7w
mmft data sqa --root data/sqa
python -c "from huggingface_hub import snapshot_download as s; print(s('llava-hf/llava-1.5-7b-hf', allow_patterns=['*.json', '*.safetensors', '*.model']))"
