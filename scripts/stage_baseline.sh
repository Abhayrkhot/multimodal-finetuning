# Zero-shot baselines: ScienceQA-IMG (harness check vs the published 66.8% / 71.6% for 7B / 13B),
# a 5,000-question Visual7W val subset (for temperature fitting), Visual7W test, and Visual7W test
# with blank images.   stage_baseline.sh [model-id] [file-prefix]   e.g. "" for 7B, 13b_ for 13B
set -euo pipefail
MODEL=${1:-llava-hf/llava-1.5-7b-hf}; P=${2:-}
R=results/explorer; mkdir -p $R
mmft eval --model "$MODEL" --dataset sqa --root data/sqa --out $R/zs_${P}sqa_test.jsonl.gz
mmft eval --model "$MODEL" --root data/visual7w --split val --limit 5000 --out $R/zs_${P}v7w_val5k.jsonl.gz
mmft eval --model "$MODEL" --root data/visual7w --split test --out $R/zs_${P}v7w_test.jsonl.gz
mmft eval --model "$MODEL" --root data/visual7w --split test --blind --out $R/blind_${P}v7w_test.jsonl.gz
