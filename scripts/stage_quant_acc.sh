# Test accuracy of the merged model at each precision (bf16 also checks that merging the adapters
# changed nothing).   stage_quant_acc.sh [run-name]
set -euo pipefail
NAME=${1:-v7w_r16}
R=results/explorer; M=runs/${NAME}_merged; mkdir -p $R
for q in bf16 nf4 int8; do
  mmft eval --model $M --quant $q --root data/visual7w --split test --out $R/ft_${NAME}_${q}_v7w_test.jsonl.gz
done
