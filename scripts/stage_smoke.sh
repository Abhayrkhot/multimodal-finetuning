# Fast end-to-end pass on a GPU before the long runs: every stage on a few hundred questions.
set -euo pipefail
S=smoke; rm -rf $S; mkdir -p $S
mmft eval --root data/visual7w --split val --limit 256 --out $S/zs_val.jsonl.gz
mmft eval --dataset sqa --root data/sqa --limit 128 --out $S/zs_sqa.jsonl.gz
python - <<'PY'
import yaml
c = yaml.safe_load(open("src/configs/llava15_7b_v7w.yaml"))
c.update(n_train=640, n_val=64, eval_every=10, log_every=5)
yaml.safe_dump(c, open("smoke/cfg.yaml", "w"))
PY
mmft train --config $S/cfg.yaml --out $S/run
mmft eval --adapter $S/run --root data/visual7w --split val --limit 256 --out $S/ft_val.jsonl.gz
mmft merge --adapter $S/run --out $S/merged
mmft eval --model $S/merged --root data/visual7w --split val --limit 256 --out $S/merged_val.jsonl.gz
for q in bf16 int8 nf4; do
  mmft bench --model $S/merged --quant $q --root data/visual7w --batch-sizes 1 16 --iters 3 --out $S/bench_$q.json
done
mmft eval --model $S/merged --quant nf4 --root data/visual7w --split val --limit 256 --out $S/nf4_val.jsonl.gz
PORT=$((20000 + RANDOM % 20000))
mmft serve --model $S/merged --quant nf4 --port $PORT > $S/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null && break; sleep 5; done
python src/scripts/serve_check.py --url "http://127.0.0.1:$PORT" --root data/visual7w --preds $S/nf4_val.jsonl.gz --n 20 --out $S/serve.json
for f in $S/*.jsonl.gz; do mmft summarize --test $f --out ${f%.jsonl.gz}.json; done
