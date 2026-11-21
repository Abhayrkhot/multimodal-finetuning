# Quantized inference on one GPU: memory/latency/throughput per batch size for bf16, int8 and NF4,
# then test accuracy of each precision (bf16 here also checks that merging changed nothing).
#   stage_bench.sh [run-name] [gpu-tag]
set -euo pipefail
NAME=${1:-v7w_r16}; TAG=${2:-a100}
R=results/explorer; M=runs/${NAME}_merged; mkdir -p $R
for q in bf16 int8 nf4; do
  mmft bench --model $M --quant $q --root data/visual7w --out $R/bench_${TAG}_${q}.json
done
for q in bf16 nf4 int8; do
  mmft eval --model $M --quant $q --root data/visual7w --split test --out $R/ft_${NAME}_${q}_v7w_test.jsonl.gz
done
