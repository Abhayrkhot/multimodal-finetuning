"""mmft command line: data | train | eval | merge | bench | summarize | serve."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _model(args):
    from .model import load_adapter, load_model, merge_lora

    quant = getattr(args, "quant", None)
    adapter = getattr(args, "adapter", None)
    if adapter and quant not in (None, "bf16"):
        raise SystemExit("quantized runs load a merged model dir (`mmft merge`), not --adapter")
    processor, model = load_model(args.model, quant=quant)
    if adapter:
        load_adapter(model, adapter)
        merge_lora(model)
    return processor, model


def cmd_data(args):
    from .data import download_v7w, prepare_sqa

    {"v7w": download_v7w, "sqa": prepare_sqa}[args.dataset](args.root)


def cmd_train(args):
    from .config import TrainConfig
    from .train import train

    cfg = TrainConfig.load(args.config, n_train=args.n_train, epochs=args.epochs, data_root=args.data_root)
    summary = train(cfg, args.out)
    print(json.dumps({k: summary[k] for k in ("n_train", "train_seconds", "final_val_acc")}))


def cmd_eval(args):
    import torch

    from .batching import Encoder
    from .data import load_items
    from .evaluate import predict, summarize, write_predictions
    from .runmeta import run_meta

    items = load_items(args.dataset, args.root, args.split, seed=args.seed, limit=args.limit)
    processor, model = _model(args)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    rows = predict(
        model,
        Encoder(processor),
        items,
        batch_size=args.batch_size,
        num_workers=args.workers,
        blind=args.blind,
    )
    meta = {
        "model": args.model,
        "adapter": args.adapter,
        "quant": args.quant or "bf16",
        "dataset": args.dataset,
        "split": args.split,
        "blind": args.blind,
        "limit": args.limit,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seconds": time.time() - start,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "env": run_meta(),
    }
    write_predictions(rows, args.out, meta)
    s = summarize(rows)
    print(json.dumps({"n": s["n"], "accuracy": s["accuracy"], "ece": s["ece"], "per_group": s["per_group"]}))


def cmd_merge(args):
    from .model import load_adapter, load_model, merge_and_save

    processor, model = load_model(args.model)
    load_adapter(model, args.adapter)
    n = merge_and_save(model, processor, args.out)
    print(f"merged {n} LoRA modules into {args.out}")


def cmd_bench(args):
    import torch

    from .batching import Encoder
    from .bench import bench_model
    from .data import load_items
    from .runmeta import run_meta

    items = load_items("v7w", args.root, "test", seed=args.seed, limit=max(args.batch_sizes))
    torch.cuda.reset_peak_memory_stats()
    processor, model = _model(args)
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 2**30
    rows = bench_model(model, Encoder(processor), items, batch_sizes=args.batch_sizes, iters=args.iters)
    result = {
        "model": args.model,
        "quant": args.quant or "bf16",
        "weights_mem_gb": weights_gb,
        "rows": rows,
        "env": run_meta(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


def cmd_summarize(args):
    from .calibration import fit_temperature
    from .evaluate import logits_matrix, read_predictions, summarize

    meta, rows = read_predictions(args.test)
    t = 1.0
    if args.val:
        _, val_rows = read_predictions(args.val)
        t = fit_temperature(*logits_matrix(val_rows))
    out = {
        "meta": meta,
        "raw": summarize(rows),
        "temperature_scaled": summarize(rows, t) if args.val else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{args.test}: acc {out['raw']['accuracy']:.4f} ece {out['raw']['ece']:.4f} T={t:.3f}")


def cmd_serve(args):
    import uvicorn

    from .serve import create_app

    uvicorn.run(
        create_app(
            args.model, quant=args.quant, temperature=args.temperature, abstain_below=args.abstain_below
        ),
        host=args.host,
        port=args.port,
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="mmft")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="download and verify a dataset")
    s.add_argument("dataset", choices=["v7w", "sqa"])
    s.add_argument("--root", required=True)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("train", help="LoRA fine-tune")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--data-root")
    s.add_argument("--n-train", type=int)
    s.add_argument("--epochs", type=int)
    s.set_defaults(fn=cmd_train)

    def model_args(s, adapter=True):
        s.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="HF id or merged model dir")
        if adapter:
            s.add_argument("--adapter", help="adapter dir from `mmft train` (merged after loading)")
        s.add_argument("--quant", choices=["bf16", "int8", "nf4"])

    s = sub.add_parser("eval", help="score a multiple-choice split, write per-question predictions")
    model_args(s)
    s.add_argument("--dataset", choices=["v7w", "sqa"], default="v7w")
    s.add_argument("--root", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--blind", action="store_true", help="replace every image with a blank one")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("merge", help="fold adapters into the base weights and save a bf16 model")
    s.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("bench", help="memory/latency/throughput for one precision")
    model_args(s, adapter=False)
    s.add_argument("--root", required=True)
    s.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    s.add_argument("--iters", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_bench, adapter=None)

    s = sub.add_parser("summarize", help="metrics (+ temperature fitted on val predictions)")
    s.add_argument("--test", required=True)
    s.add_argument("--val")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_summarize)

    s = sub.add_parser("serve", help="HTTP API over a merged model")
    s.add_argument("--model", required=True)
    s.add_argument("--quant", choices=["bf16", "int8", "nf4"])
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--abstain-below", type=float, default=0.0)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
