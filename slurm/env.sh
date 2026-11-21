# Sourced by every job from the code directory it was submitted from, so a job always runs the
# commit it was submitted with. Layout under /scratch/$USER/mmft: code/<sha>/ (git archive +
# GIT_SHA), venv/, data/, hf/ (model cache), runs/, results/, logs/.
module load python/3.13.5
export MMFT_HOME=/scratch/$USER/mmft
MMFT_SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MMFT_SRC
export HF_HOME=$MMFT_HOME/hf
export PYTHONNOUSERSITE=1            # ~/.local holds an old torch that would shadow the venv's
export PYTHONPATH=$MMFT_SRC
export PATH=$MMFT_HOME/venv/bin:$PATH
MMFT_GIT_SHA=$(cat "$MMFT_SRC/GIT_SHA" 2>/dev/null || echo unknown)
export MMFT_GIT_SHA
export TOKENIZERS_PARALLELISM=false
mmft() { python -m mmft.cli "$@"; }
export -f mmft
