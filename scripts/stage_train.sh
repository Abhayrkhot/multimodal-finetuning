# LoRA fine-tune, then evaluate on the val subset, test, and ScienceQA-IMG (forgetting). The base
# model comes from the config and is recorded with the adapter. Merging is stage_merge.sh, run
# only for the selected run.   stage_train.sh [config] [run-name]
set -euo pipefail
CFG=${1:-$MMFT_SRC/configs/llava15_7b_v7w.yaml}; NAME=${2:-v7w_r16}
R=results/explorer; RUN=runs/$NAME; mkdir -p $R
mmft train --config "$CFG" --out $RUN
cp $RUN/run.json $R/train_${NAME}.json; cp $RUN/train_log.jsonl $R/train_${NAME}_log.jsonl
mmft eval --adapter $RUN --root data/visual7w --split val --limit 5000 --out $R/ft_${NAME}_v7w_val5k.jsonl.gz
mmft eval --adapter $RUN --root data/visual7w --split test --out $R/ft_${NAME}_v7w_test.jsonl.gz
mmft eval --adapter $RUN --dataset sqa --root data/sqa --out $R/ft_${NAME}_sqa_test.jsonl.gz
