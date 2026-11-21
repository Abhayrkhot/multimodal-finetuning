"""Inference benchmark for one precision: weight memory, then per batch size the peak GPU memory,
latency and throughput of the deployed workload (one forward pass that scores the answer
letters). Inputs are preprocessed once, so the timings are GPU work only.

Run each precision in a fresh process: the allocator's peak and cached blocks from one model
would otherwise leak into the next measurement.
"""

from __future__ import annotations

import statistics

import torch

from .batching import Collator, Encoder
from .evaluate import letter_logits


@torch.no_grad()
def bench_model(
    model, encoder: Encoder, items, *, batch_sizes=(1, 2, 4, 8, 16), iters: int = 20, warmup: int = 3
) -> list[dict]:
    dev = next(model.parameters()).device
    letter_ids = torch.tensor(encoder.letter_ids, device=dev)
    collate = Collator(encoder.pad_id, encoder.letter_ids, encoder.eos_id, train=False)
    encoded = [encoder.encode(it) for it in items[: max(batch_sizes)]]
    rows = []
    for bs in batch_sizes:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate(encoded[:bs]).items()}
        for _ in range(warmup):
            letter_logits(model, batch, letter_ids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            letter_logits(model, batch, letter_ids)
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        med = statistics.median(times)
        rows.append(
            {
                "batch_size": bs,
                "seq_len": int(batch["input_ids"].shape[1]),
                "latency_ms_p50": med,
                "latency_ms_p90": sorted(times)[int(0.9 * (len(times) - 1))],
                "throughput_qps": bs / (med / 1000),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        print(rows[-1], flush=True)
    return rows
