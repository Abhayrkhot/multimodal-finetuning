"""LoRA adapters written directly in PyTorch (no PEFT).

y = W x + (alpha / r) * B(A(dropout(x)))

A is Kaiming-uniform and B is zero, so a freshly wrapped model computes exactly what the base
model does. Adapter weights stay in fp32 (the optimizer's master copy) and are cast to the
activation dtype in forward, so a bf16 base model trains without bf16 rounding of the updates.
`merge_lora` folds the update into W for inference and leaves plain nn.Linear modules.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.base = base
        self.r, self.scaling = r, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        z = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
        return out + F.linear(z, self.lora_B.to(x.dtype)) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        w = self.base.weight
        out = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=w.device,
            dtype=w.dtype,
        )
        out.weight.copy_((w.float() + self.delta_weight()).to(w.dtype))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def _set_child(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child, module)


def apply_lora(
    model: nn.Module,
    *,
    scope: str,
    targets: tuple[str, ...],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    exclude=None,
) -> list[str]:
    """Wrap every nn.Linear whose qualified name contains `scope` and ends with one of `targets`
    (minus names for which `exclude(name)` is true)."""
    names = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and scope in n
        and n.rsplit(".", 1)[-1] in targets
        and not (exclude and exclude(n))
    ]
    if not names:
        raise ValueError(f"no nn.Linear under {scope!r} matched {targets}")
    for n in names:
        _set_child(model, n, LoRALinear(model.get_submodule(n), r, alpha, dropout))
    return names


def merge_lora(model: nn.Module) -> int:
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in names:
        _set_child(model, n, model.get_submodule(n).merged())
    return len(names)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".lora_A" in n or ".lora_B" in n]
