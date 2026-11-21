# Zero-shot LLaVA-1.5-7B: ScienceQA-IMG (harness check vs the published 66.8%), Visual7W val
# subset (for temperature fitting), Visual7W test, and Visual7W test with blank images.
set -euo pipefail
R=results/explorer; mkdir -p $R
mmft eval --dataset sqa --root data/sqa --out $R/zs_sqa_test.jsonl.gz
mmft eval --root data/visual7w --split val --limit 5000 --out $R/zs_v7w_val5k.jsonl.gz
mmft eval --root data/visual7w --split test --out $R/zs_v7w_test.jsonl.gz
mmft eval --root data/visual7w --split test --blind --out $R/blind_v7w_test.jsonl.gz
