"""Calibration metrics over multiple-choice letter logits, and temperature scaling."""

from __future__ import annotations

import numpy as np
import torch


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "n": int(m.sum()),
                    "conf": float(conf[m].mean()),
                    "acc": float(correct[m].mean()),
                }
            )
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (Guo et al., 2017)."""
    n = len(conf)
    return float(sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability_bins(conf, correct, n_bins)))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single temperature T > 0 minimising NLL of softmax(logits / T); logits may hold -inf for
    options a question does not have."""
    x = torch.tensor(np.where(np.isfinite(logits), logits, -1e9), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())
