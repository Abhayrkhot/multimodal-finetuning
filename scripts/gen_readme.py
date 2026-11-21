"""Fill the generated sections of README.md from results/explorer.

Every number in those sections is recomputed here from per-question predictions
(*.jsonl.gz), training run records and benchmark JSON, so the README cannot drift from the data.
    python scripts/gen_readme.py           rewrite README.md
    python scripts/gen_readme.py --check   exit 1 if README.md is stale (CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmft.calibration import fit_temperature  # noqa: E402
from mmft.data import V7W_TYPES  # noqa: E402
from mmft.evaluate import logits_matrix, read_predictions, summarize  # noqa: E402

R = ROOT / "results" / "explorer"
RUNS = ("v7w_r16", "v7w_r16_e2", "v7w_r16_vis", "v7w_r16_vis_e2")  # reported: the best on validation
PENDING = "_Pending: not measured yet._"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ci95(acc: float, n: int) -> str:
    """Normal-approximation 95% interval half-width for an accuracy, in points."""
    return f"{196 * (acc * (1 - acc) / n) ** 0.5:.1f}"


def err_cut(before_acc: float, after_acc: float) -> str:
    """Relative reduction in error rate, e.g. 20% -> 10% error is -50%."""
    if before_acc >= 1.0:
        return "n/a"
    x = round(-100 * (1 - (1 - after_acc) / (1 - before_acc)))
    return f"{x:+d}%" if x else "0%"


def load(name: str):
    p = R / f"{name}.jsonl.gz"
    return read_predictions(p) if p.exists() else (None, None)


def summary(name: str, val: str | None = None) -> dict | None:
    meta, rows = load(name)
    if rows is None:
        return None
    s = summarize(rows)
    s["meta"] = meta
    if val:
        _, vrows = load(val)
        if vrows is not None:
            t = fit_temperature(*logits_matrix(vrows))
            s["ts"] = summarize(rows, t)
    return s


def pick_run() -> tuple[str, list[dict]]:
    """Choose the training run by accuracy on the 5,000-question validation subset."""
    rows = []
    for run in RUNS:
        val, test = summary(f"ft_{run}_v7w_val5k"), summary(f"ft_{run}_v7w_test")
        tp = R / f"train_{run}.json"
        if val and tp.exists():
            t = json.loads(tp.read_text())
            rows.append(
                {
                    "run": run,
                    "recipe": "LM LoRA + projector"
                    + (" + CLIP LoRA" if t["config"].get("vision_lora_targets") else ""),
                    "epochs": t["config"]["epochs"],
                    "val": val["accuracy"],
                    "test": test["accuracy"] if test else None,
                    "minutes": t["train_seconds"] / 60,
                }
            )
    best = max(rows, key=lambda r: r["val"])["run"] if rows else RUNS[0]
    return best, rows


def headline(zs, ft, sqa_zs, sqa_ft, train, bench) -> str:
    if not (zs and ft):
        return PENDING
    zw, fw = zs["worst_group"], ft["worst_group"]
    z_err, f_err = 1 - zs["worst_group_acc"], 1 - ft["per_group"][zw]["acc"]
    lines = [
        "| | Zero-shot LLaVA-1.5-7B | + LoRA fine-tuning | Change |",
        "|---|---|---|---|",
        f"| Visual7W test accuracy ({ft['n']:,} questions, 95% CI) | {pct(zs['accuracy'])} ± {ci95(zs['accuracy'], zs['n'])} "
        f"| {pct(ft['accuracy'])} ± {ci95(ft['accuracy'], ft['n'])} "
        f"| +{100 * (ft['accuracy'] - zs['accuracy']):.1f} pts |",
        f"| Error on the worst question type (*{zw}*) | {pct(z_err)} | {pct(f_err)} "
        f"| {err_cut(1 - z_err, 1 - f_err)} relative |",
    ]
    if fw != zw:
        lines.append(
            f"| Worst type after fine-tuning (*{fw}*) error | | {pct(1 - ft['worst_group_acc'])} | |"
        )
    if "ts" in zs and "ts" in ft:
        lines.append(
            f"| Expected calibration error, raw / temperature-scaled | {pct(zs['ece'], 1)} / {pct(zs['ts']['ece'], 1)} "
            f"| {pct(ft['ece'], 1)} / {pct(ft['ts']['ece'], 1)} | |"
        )
    if sqa_zs and sqa_ft:
        lines.append(
            f"| ScienceQA-IMG accuracy (never trained on; forgetting check) | {pct(sqa_zs['accuracy'])} "
            f"| {pct(sqa_ft['accuracy'])} | {100 * (sqa_ft['accuracy'] - sqa_zs['accuracy']):+.1f} pts |"
        )
    if train:
        p = train["params"]
        lines.append(
            f"| Trainable parameters | {p['base_params'] / 1e9:.2f}B (full fine-tune) "
            f"| {p['trainable_params'] / 1e6:.1f}M ({pct(p['trainable_fraction'], 2)}) "
            f"| -{100 * (1 - p['trainable_fraction']):.1f}% |"
        )
    if bench.get("bf16") and bench.get("nf4"):
        b16 = {r["batch_size"]: r["peak_mem_gb"] for r in bench["bf16"]["rows"]}
        n4 = {r["batch_size"]: r["peak_mem_gb"] for r in bench["nf4"]["rows"]}
        red = [1 - n4[b] / b16[b] for b in b16]
        gpu = bench["bf16"]["env"]["gpu"]
        lines.append(
            f"| Peak GPU memory, NF4 vs BF16, batch 1-16 ({gpu}) | {b16[min(b16)]:.1f}-{b16[max(b16)]:.1f} GB "
            f"| {n4[min(n4)]:.1f}-{n4[max(n4)]:.1f} GB | -{100 * min(red):.0f}% to -{100 * max(red):.0f}% |"
        )
    return "\n".join(lines)


def slices(zs, ft, blind) -> str:
    if not (zs and ft):
        return PENDING
    cols = [
        "Question type",
        "Questions",
        *(["Blank image (zero-shot)"] if blind else []),
        "Zero-shot",
        "Fine-tuned",
        "Error reduction",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in [*V7W_TYPES, "all"]:
        if t == "all":
            name, n, z, f = "**all**", ft["n"], zs["accuracy"], ft["accuracy"]
            b = blind["accuracy"] if blind else None
        else:
            name, n = t, ft["per_group"][t]["n"]
            z, f = zs["per_group"][t]["acc"], ft["per_group"][t]["acc"]
            b = blind["per_group"][t]["acc"] if blind else None
        row = f"| {name} | {n:,} |" + (f" {pct(b)} |" if blind else "")
        row += f" {pct(z)} | {pct(f)} | {err_cut(z, f)} |"
        lines.append(row)
    return "\n".join(lines)


def calibration(zs, ft) -> str:
    if not (zs and ft and "ts" in zs and "ts" in ft):
        return PENDING
    lines = [
        "| Model | Temperature | Accuracy | Mean confidence | ECE | NLL | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in (("Zero-shot", zs), ("Fine-tuned", ft)):
        for tag, x in (("raw", s), ("scaled", s["ts"])):
            lines.append(
                f"| {name}, {tag} | {x['temperature']:.2f} | {pct(x['accuracy'])} | {pct(x['mean_confidence'])} "
                f"| {pct(x['ece'], 2)} | {x['nll']:.3f} | {x['brier']:.3f} |"
            )
    return "\n".join(lines)


def quant(bench, accs) -> str:
    if not bench.get("bf16"):
        return PENDING
    gpu = bench["bf16"]["env"]["gpu"]
    lines = [
        f"Measured on one {gpu}. Workload: one forward pass over a Visual7W prompt "
        f"(~{bench['bf16']['rows'][0]['seq_len']} tokens incl. 576 image tokens) that scores the answer letters. "
        "Peak memory is `torch.cuda.max_memory_allocated`; latency is the median of 20 CUDA-event-timed runs.",
        "",
        "| Precision | Weights in memory | Test accuracy | Batch | Peak memory | vs BF16 | Latency p50 | Questions/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = {r["batch_size"]: r["peak_mem_gb"] for r in bench["bf16"]["rows"]}
    for q in ("bf16", "int8", "nf4"):
        b = bench.get(q)
        if not b:
            continue
        acc = pct(accs[q]["accuracy"]) if accs.get(q) else "pending"
        for i, r in enumerate(b["rows"]):
            first = i == 0
            weights = f"{b['weights_mem_gb']:.2f} GB" if first else ""
            vs = "" if q == "bf16" else f"-{100 * (1 - r['peak_mem_gb'] / base[r['batch_size']]):.0f}%"
            lines.append(
                f"| {q.upper() if first else ''} | {weights} | {acc if first else ''} | {r['batch_size']} "
                f"| {r['peak_mem_gb']:.2f} GB | {vs} | {r['latency_ms_p50']:.1f} ms | {r['throughput_qps']:.1f} |"
            )
    return "\n".join(lines)


def training(train, run, candidates) -> str:
    if not train:
        return PENDING
    sel = ""
    if len(candidates) > 1:
        sel = "\n".join(
            [
                "",
                f"{len(candidates)} recipes were trained; the reported run is the one with the highest accuracy "
                "on 5,000 validation questions (test accuracy shown only for completeness):",
                "",
                "| Run | Adapters | Epochs | Wall time | Val accuracy (5,000) | Test accuracy | Reported |",
                "|---|---|---|---|---|---|---|",
                *[
                    f"| `{c['run']}` | {c['recipe']} | {c['epochs']} | {c['minutes']:.0f} min | {pct(c['val'], 2)} "
                    f"| {pct(c['test'], 2) if c['test'] is not None else 'pending'} | {'yes' if c['run'] == run else ''} |"
                    for c in candidates
                ],
            ]
        )
    c, p, e = train["config"], train["params"], train["env"]
    return (
        "\n".join(
            [
                "| | |",
                "|---|---|",
                f"| Training questions | {train['n_train']:,} (Visual7W train) x {c['epochs']} epoch(s) |",
                f"| Adapters | LoRA r={c['lora_r']}, alpha={c['lora_alpha']:g} on {p['lora_modules']} language-model linear layers "
                f"({(p['lora_params'] - p.get('vision_lora_params', 0)) / 1e6:.1f}M)"
                + (
                    f" + {p['vision_lora_modules']} CLIP linear layers ({p['vision_lora_params'] / 1e6:.1f}M)"
                    if p.get("vision_lora_modules")
                    else ""
                )
                + f" + projector ({p['projector_params'] / 1e6:.1f}M) |",
                f"| Optimiser | AdamW, lr {c['lr']:g} (LoRA) / {c['projector_lr']:g} (projector), cosine, "
                f"{pct(c['warmup_ratio'], 0)} warmup, batch {c['batch_size'] * c['grad_accum']} |",
                f"| Steps / wall time | {train['optimizer_steps']:,} / {train['train_seconds'] / 60:.0f} min on one {e['gpu']} |",
                f"| Throughput / peak memory | {train['samples_per_s']:.1f} questions/s, {train['tokens_per_s']:,.0f} tokens/s "
                f"/ {train['peak_mem_gb']:.1f} GB |",
                f"| Val accuracy at end ({train['n_val_monitor']:,}-question monitor subset) | {pct(train['final_val_acc'])} |",
                f"| Code / stack | `{e['git_sha']}`, torch {e['torch']} (CUDA {e['cuda']}), transformers {e['transformers']} |",
            ]
        )
        + sel
    )


def controls(zs, ft, blind, ft_merged, sqa_zs) -> str:
    if not (zs and ft):
        return PENDING
    lines = []
    if sqa_zs:
        lines.append(
            f"- **Harness check.** Zero-shot ScienceQA-IMG accuracy is {pct(sqa_zs['accuracy'])} with this "
            "prompt, image padding and letter scoring; the LLaVA-1.5 paper reports 66.8% for the same model."
        )
    if blind:
        lines.append(
            f"- **The image matters.** With every image replaced by a blank one, zero-shot accuracy falls from "
            f"{pct(zs['accuracy'])} to {pct(blind['accuracy'])} (chance is 25%): Visual7W's distractors leave "
            "some text-only signal, which is why the blank-image row is reported per type."
        )
    pos = ft["pred_position_rate"]
    lines.append(
        "- **No answer-position shortcut.** Correct answers are uniform over A-D by construction; the fine-tuned "
        "model predicts " + ", ".join(f"{k} {pct(v)}" for k, v in pos.items()) + "."
    )
    if ft_merged:
        same = ft_merged["accuracy"] - ft["accuracy"]
        lines.append(
            f"- **Merging is lossless.** The merged BF16 checkpoint scores {pct(ft_merged['accuracy'], 2)} vs "
            f"{pct(ft['accuracy'], 2)} with separate adapters ({100 * same:+.2f} pts)."
        )
    return "\n".join(lines)


def serving(serve, temperature) -> str:
    if not serve:
        return PENDING
    return "\n".join(
        [
            f"`mmft serve` on the NF4 model (temperature {temperature:.2f} fitted on validation, abstain below 50%), "
            f"{serve['n']} test questions sent one at a time over HTTP on the same GPU:",
            "",
            "| Agreement with offline NF4 predictions | Accuracy | Abstained | Server latency p50 / p95 | Client latency p50 / p95 |",
            "|---|---|---|---|---|",
            f"| {pct(serve['agreement_with_offline'])} | {pct(serve['accuracy'])} | {pct(serve['abstain_rate'])} "
            f"| {serve['server_latency_ms_p50']:.0f} / {serve['server_latency_ms_p95']:.0f} ms "
            f"| {serve['client_latency_ms_p50']:.0f} / {serve['client_latency_ms_p95']:.0f} ms |",
        ]
    )


def render() -> str:
    readme = (ROOT / "README.md").read_text()
    run, candidates = pick_run()
    zs = summary("zs_v7w_test", "zs_v7w_val5k")
    ft = summary(f"ft_{run}_v7w_test", f"ft_{run}_v7w_val5k")
    blind = summary("blind_v7w_test")
    sqa_zs, sqa_ft = summary("zs_sqa_test"), summary(f"ft_{run}_sqa_test")
    accs = {q: summary(f"ft_{run}_{q}_v7w_test") for q in ("bf16", "int8", "nf4")}
    train_p = R / f"train_{run}.json"
    train = json.loads(train_p.read_text()) if train_p.exists() else None
    bench = {}
    for q in ("bf16", "int8", "nf4"):
        p = R / f"bench_a100_{q}.json"
        if p.exists():
            bench[q] = json.loads(p.read_text())
    serve_p = R / "serve_a100_nf4.json"
    serve = json.loads(serve_p.read_text()) if serve_p.exists() else None
    sections = {
        "headline": headline(zs, ft, sqa_zs, sqa_ft, train, bench),
        "slices": slices(zs, ft, blind),
        "calibration": calibration(zs, ft),
        "quant": quant(bench, accs),
        "training": training(train, run, candidates),
        "controls": controls(zs, ft, blind, accs.get("bf16"), sqa_zs),
        "serving": serving(serve, ft["ts"]["temperature"] if ft and "ts" in ft else 1.0),
    }
    for name, body in sections.items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->\n)(?:.*?\n)?(<!-- END {name} -->)", re.S)
        if not pat.search(readme):
            raise SystemExit(f"README.md has no {name} markers")
        readme = pat.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), readme)
    return readme


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    new = render()
    cur = (ROOT / "README.md").read_text()
    if args.check:
        if new != cur:
            print("README.md is stale: run python scripts/gen_readme.py", file=sys.stderr)
            return 1
        return 0
    (ROOT / "README.md").write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
