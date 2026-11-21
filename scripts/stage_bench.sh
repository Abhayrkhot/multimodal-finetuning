# Quantized inference on one GPU (fits the 1 h `sharing` partition): memory, latency and
# throughput per batch size for bf16, int8 and NF4, then the HTTP service on NF4 checked against
# offline NF4 predictions for the same 500 test questions. Full-test accuracy per precision is
# stage_quant_acc.sh.
#   stage_bench.sh [run-name] [gpu-tag]
set -euo pipefail
NAME=${1:-v7w_r16}; TAG=${2:-a100}
R=results/explorer; M=runs/${NAME}_merged; mkdir -p $R
for q in bf16 int8 nf4; do
  mmft bench --model $M --quant $q --root data/visual7w --out $R/bench_${TAG}_${q}.json
done
mmft eval --model $M --quant nf4 --root data/visual7w --split test --limit 500 --seed 1 \
  --out $R/serve_${TAG}_nf4_offline.jsonl.gz

# Serving: temperature fitted on validation predictions, abstain below 50% confidence.
T=$(python -c "
from mmft.calibration import fit_temperature
from mmft.evaluate import logits_matrix, read_predictions
print(round(fit_temperature(*logits_matrix(read_predictions('$R/ft_${NAME}_v7w_val5k.jsonl.gz')[1])), 4))")
PORT=$((20000 + RANDOM % 20000))
mmft serve --model $M --quant nf4 --temperature "$T" --abstain-below 0.5 --port $PORT > logs/serve_${TAG}.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 120); do
  curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  sleep 5
done
python "$MMFT_SRC/scripts/serve_check.py" --url "http://127.0.0.1:$PORT" --root data/visual7w \
  --preds $R/serve_${TAG}_nf4_offline.jsonl.gz --n 500 --out $R/serve_${TAG}_nf4.json
