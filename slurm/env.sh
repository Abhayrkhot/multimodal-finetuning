# Sourced by every job. Layout under /scratch/$USER/mmft: src/ (git archive + GIT_SHA), venv/,
# data/, hf/ (model cache), runs/ (adapters, merged weights), results/, logs/.
module load python/3.13.5
export MMFT_HOME=/scratch/$USER/mmft
export HF_HOME=$MMFT_HOME/hf
export PYTHONNOUSERSITE=1            # ~/.local holds an old torch that would shadow the venv's
export PYTHONPATH=$MMFT_HOME/src
export PATH=$MMFT_HOME/venv/bin:$PATH
export MMFT_GIT_SHA=$(cat "$MMFT_HOME/src/GIT_SHA" 2>/dev/null || echo unknown)
export TOKENIZERS_PARALLELISM=false
mmft() { python -m mmft.cli "$@"; }
export -f mmft
