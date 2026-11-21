# Fetch a model's weights into the shared HF cache (CPU node).   stage_download.sh model-id
set -euo pipefail
python -c "from huggingface_hub import snapshot_download as s; import sys; print(s(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.model']))" "$1"
