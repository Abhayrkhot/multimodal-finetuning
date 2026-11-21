"""Score multiple-choice items with one forward pass each and summarise accuracy, slices and
calibration.

The prediction is the argmax over the option letters' next-token logits after "ASSISTANT:";
those logits (restricted to the options the question has) are also the model's probabilities for
calibration. Predictions are written per question so every metric can be recomputed offline.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import calibration as cal
from .batching import Encoder, make_loader
from .data import LETTERS, MCItem


def vision_dtype(model) -> torch.dtype:
    return next(p.dtype for n, p in model.named_parameters() if "vision_tower" in n)


def forward_last(model, batch: dict, keep: int = 1):
    """Model forward on a collated batch, returning logits for the last `keep` positions only."""
    device = next(model.parameters()).device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            position_ids=batch["position_ids"].to(device, non_blocking=True),
            pixel_values=batch["pixel_values"].to(device, vision_dtype(model), non_blocking=True),
            logits_to_keep=keep,
        ).logits


def letter_logits(model, batch: dict, letter_ids: torch.Tensor) -> torch.Tensor:
    return forward_last(model, batch, 1)[:, -1].float().index_select(-1, letter_ids)


@torch.no_grad()
def predict(
    model,
    encoder: Encoder,
    items: list[MCItem],
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    blind: bool = False,
    log_every: int = 200,
) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    loader = make_loader(
        items, encoder, batch_size=batch_size, train=False, num_workers=num_workers, blind=blind
    )
    letter_ids = torch.tensor(encoder.letter_ids, device=device)
    rows, start = [], time.time()
    for step, batch in enumerate(loader):
        logits = letter_logits(model, batch, letter_ids).cpu()
        for j in range(logits.shape[0]):
            item = items[len(rows)]
            n = int(batch["n_choices"][j])
            rows.append(
                {
                    "qid": item.qid,
                    "group": item.group,
                    "answer": item.answer,
                    "logits": [round(v, 4) for v in logits[j, :n].tolist()],
                }
            )
        if log_every and (step + 1) % log_every == 0:
            print(
                f"  scored {len(rows)}/{len(items)} ({len(rows) / (time.time() - start):.1f} q/s)", flush=True
            )
    return rows


def write_predictions(rows: list[dict], path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_predictions(path: str | Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    return json.loads(lines[0])["meta"], [json.loads(x) for x in lines[1:]]


def logits_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r["logits"]) for r in rows)
    x = np.full((len(rows), width), -np.inf)
    for i, r in enumerate(rows):
        x[i, : len(r["logits"])] = r["logits"]
    return x, np.array([r["answer"] for r in rows])


def summarize(rows: list[dict], temperature: float = 1.0, n_bins: int = 15) -> dict:
    x, y = logits_matrix(rows)
    probs = cal.softmax(x, temperature)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    conf = probs.max(1)
    groups = sorted({r["group"] for r in rows})
    g = np.array([r["group"] for r in rows])
    per_group = {k: {"n": int((g == k).sum()), "acc": float(correct[g == k].mean())} for k in groups}
    worst = min(per_group, key=lambda k: per_group[k]["acc"])
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "per_group": per_group,
        "worst_group": worst,
        "worst_group_acc": per_group[worst]["acc"],
        "temperature": temperature,
        "ece": cal.ece(conf, correct, n_bins),
        "nll": cal.nll(probs, y),
        "brier": cal.brier(probs, y),
        "mean_confidence": float(conf.mean()),
        "answer_position_rate": {LETTERS[i]: float((y == i).mean()) for i in range(x.shape[1])},
        "pred_position_rate": {LETTERS[i]: float((pred == i).mean()) for i in range(x.shape[1])},
        "reliability": cal.reliability_bins(conf, correct, n_bins),
    }
