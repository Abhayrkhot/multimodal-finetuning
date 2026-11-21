from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .model import LM_TARGETS


@dataclass
class TrainConfig:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    data_root: str = "data/visual7w"
    n_train: int | None = None  # None = the whole train split
    n_val: int = 2000  # fixed val subset scored during training
    seed: int = 0
    # adapters
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LM_TARGETS
    vision_lora_targets: tuple[str, ...] = ()  # e.g. VISION_TARGETS to adapt the CLIP encoder too
    train_projector: bool = True
    # optimisation
    epochs: int = 1
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 2e-4
    projector_lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    num_workers: int = 8
    log_every: int = 10
    eval_every: int = 500
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for k in ("lora_targets", "vision_lora_targets"):
            if k in raw:
                raw[k] = tuple(raw[k])
        return cls(**raw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        d["vision_lora_targets"] = list(self.vision_lora_targets)
        return d
