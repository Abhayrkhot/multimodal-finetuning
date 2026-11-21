"""Send Visual7W test questions to a running `mmft serve` one at a time and compare its answers
with the offline predictions of the same model and precision.

    python scripts/serve_check.py --url http://127.0.0.1:8000 --root data/visual7w \
        --preds results/explorer/ft_v7w_r16_nf4_v7w_test.jsonl.gz --n 500 --out serve.json
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmft.data import load_v7w  # noqa: E402
from mmft.evaluate import read_predictions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta, rows = read_predictions(args.preds)
    offline = {r["qid"]: max(range(len(r["logits"])), key=r["logits"].__getitem__) for r in rows}
    pool = load_v7w(args.root, meta["split"], limit=meta["limit"], seed=meta["seed"])
    items = random.Random(1).sample(pool, min(args.n, len(pool)))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # compute nodes set http_proxy
    client_ms, server_ms, agree, correct, abstain = [], [], 0, 0, 0
    for it in items:
        body = json.dumps(
            {
                "image_b64": base64.b64encode(Path(it.image).read_bytes()).decode(),
                "question": it.question,
                "choices": list(it.choices),
            }
        ).encode()
        req = urllib.request.Request(f"{args.url}/v1/answer", body, {"content-type": "application/json"})
        t0 = time.perf_counter()
        with opener.open(req, timeout=60) as r:
            resp = json.load(r)
        client_ms.append((time.perf_counter() - t0) * 1000)
        server_ms.append(resp["latency_ms"])
        agree += resp["index"] == offline[it.qid]
        correct += resp["index"] == it.answer
        abstain += resp["abstain"]

    def q(xs, p):
        return sorted(xs)[int(p * (len(xs) - 1))]

    out = {
        "n": len(items),
        "agreement_with_offline": agree / len(items),
        "accuracy": correct / len(items),
        "abstain_rate": abstain / len(items),
        "client_latency_ms_p50": statistics.median(client_ms),
        "client_latency_ms_p95": q(client_ms, 0.95),
        "server_latency_ms_p50": statistics.median(server_ms),
        "server_latency_ms_p95": q(server_ms, 0.95),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
