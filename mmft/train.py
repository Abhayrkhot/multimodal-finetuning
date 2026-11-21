"""LoRA fine-tuning loop: plain PyTorch, bf16 autocast, fp32 adapter weights, cosine schedule."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .batching import Encoder, make_loader
from .config import TrainConfig
from .data import load_v7w
from .evaluate import forward_last, predict, summarize
from .lora import lora_parameters
from .model import load_model, prepare_for_training, save_adapter
from .runmeta import run_meta


def cosine_with_warmup(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return f


def train(
    cfg: TrainConfig,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    processor=None,
    model=None,
    train_items=None,
    val_items=None,
) -> dict:
    """Train adapters and write adapter.safetensors, adapter.json, train_log.jsonl and run.json to
    out_dir. Tests pass a tiny model and items; real runs load everything from cfg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if model is None:
        processor, model = load_model(cfg.model_id, device=device)
    counts = prepare_for_training(
        model,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        targets=cfg.lora_targets,
        vision_targets=cfg.vision_lora_targets,
        train_projector=cfg.train_projector,
    )
    print(json.dumps(counts), flush=True)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    enc = Encoder(processor)
    if train_items is None:
        train_items = load_v7w(cfg.data_root, "train", seed=cfg.seed, limit=cfg.n_train)
    if val_items is None:
        val_items = load_v7w(cfg.data_root, "val", seed=cfg.seed, limit=cfg.n_val)
    loader = make_loader(
        train_items, enc, batch_size=cfg.batch_size, train=True, num_workers=cfg.num_workers, seed=cfg.seed
    )
    total = (len(loader) // cfg.grad_accum) * cfg.epochs
    if total == 0:
        raise ValueError("fewer training batches than grad_accum")
    warmup = max(1, int(total * cfg.warmup_ratio))

    lora = lora_parameters(model)
    lora_ids = {id(p) for p in lora}
    projector = [p for p in model.parameters() if p.requires_grad and id(p) not in lora_ids]
    groups = [{"params": lora, "lr": cfg.lr}]
    if projector:
        groups.append({"params": projector, "lr": cfg.projector_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=device.startswith("cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(total, warmup))

    dev = torch.device(device)
    letter_ids = torch.tensor(enc.letter_ids, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log = (out / "train_log.jsonl").open("w")
    history, step, micro = [], 0, 0
    run_loss = run_acc = run_n = 0.0
    seen = tokens = 0
    start = time.time()

    def evaluate_val() -> float:
        acc = summarize(
            predict(
                model, enc, val_items, batch_size=cfg.batch_size, num_workers=cfg.num_workers, log_every=0
            )
        )["accuracy"]
        model.train()
        return acc

    model.train()
    for _epoch in range(cfg.epochs):
        for batch in loader:
            answer = batch["answer"].to(dev)
            targets = torch.stack([letter_ids[answer], torch.full_like(answer, enc.eos_id)], dim=1)
            logits = forward_last(model, batch, keep=3)[:, :2].float()  # predicts "<letter>", "</s>"
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            (loss / cfg.grad_accum).backward()
            with torch.no_grad():
                pred = logits[:, 0].index_select(-1, letter_ids).argmax(-1)
                run_acc += (pred == answer).float().sum().item()
            run_loss += loss.item() * len(answer)
            run_n += len(answer)
            seen += len(answer)
            tokens += int(batch["attention_mask"].sum())
            micro += 1
            if micro % cfg.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = None
            if step % cfg.log_every == 0 or step == total:
                el = time.time() - start
                rec = {
                    "step": step,
                    "of": total,
                    "loss": run_loss / run_n,
                    "train_acc": run_acc / run_n,
                    "lr": sched.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "samples": seen,
                    "samples_per_s": seen / el,
                    "tokens_per_s": tokens / el,
                    "elapsed_s": el,
                }
                if dev.type == "cuda":
                    rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                run_loss = run_acc = run_n = 0.0
            if (cfg.eval_every and step % cfg.eval_every == 0) or step == total:
                rec = rec or {"step": step, "of": total}
                rec["val_acc"] = evaluate_val()
            if rec:
                history.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step >= total:
                break
    log.close()
    elapsed = time.time() - start

    meta = {
        "base_model": cfg.model_id,
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "targets": list(cfg.lora_targets),
            "modules": counts["lora_modules"],
        },
        "vision_lora": (
            {"targets": list(cfg.vision_lora_targets), "modules": counts["vision_lora_modules"]}
            if cfg.vision_lora_targets
            else None
        ),
        "train_projector": cfg.train_projector,
        "params": counts,
    }
    save_adapter(model, out, meta)
    summary = {
        "config": cfg.to_dict(),
        "params": counts,
        "n_train": len(train_items),
        "n_val_monitor": len(val_items),
        "optimizer_steps": step,
        "train_seconds": elapsed,
        "samples_per_s": seen / elapsed,
        "tokens_per_s": tokens / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
        "final_val_acc": history[-1].get("val_acc") if history else None,
        "env": run_meta(),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    return summary
