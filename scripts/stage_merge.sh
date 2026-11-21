# Fold the adapters of one run into its base weights (CPU is enough) for quantized deployment.
#   stage_merge.sh run-name
set -euo pipefail
NAME=$1
[ -d runs/${NAME}_merged ] || mmft merge --adapter runs/$NAME --out runs/${NAME}_merged --device cpu
