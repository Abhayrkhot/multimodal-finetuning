"""Load LLaVA-1.5 (CLIP ViT-L/14-336 -> 2-layer MLP projector -> Vicuna-7B), attach adapters,
save/load the trainable weights, and merge them for deployment."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .lora import apply_lora, lora_parameters, merge_lora

LM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROJECTOR = "multi_modal_projector"
SKIP_QUANT = ["vision_tower", PROJECTOR, "lm_head"]


def quant_config(quant: str | None):
    """bitsandbytes config for the language model only; CLIP, projector and lm_head stay bf16."""
    if quant in (None, "bf16"):
        return None
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        # LLM.int8 matmuls run in fp16; bitsandbytes warns on every bf16 input it casts.
        warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT)
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=SKIP_QUANT,
        )
    raise ValueError(f"unknown quant {quant!r} (bf16, int8, nf4)")


def load_model(model_id: str, *, quant: str | None = None, device: str = "cuda", attn: str = "sdpa"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    qc = quant_config(quant)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        quantization_config=qc,
        device_map={"": device} if qc is not None else None,
    )
    if qc is None:
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model


def param_count(model: nn.Module, pred=lambda n: True) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if pred(n))


def prepare_for_training(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    targets: tuple[str, ...] = LM_TARGETS,
    train_projector: bool = True,
) -> dict:
    """Freeze everything, add LoRA to the language model's linear layers, unfreeze the projector.

    Trainable weights are kept in fp32; the frozen model stays bf16 and runs under autocast."""
    base_total = param_count(model)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = apply_lora(model, scope="language_model", targets=targets, r=r, alpha=alpha, dropout=dropout)
    if train_projector:
        proj = model.get_submodule(_projector_name(model))
        proj.float()
        for p in proj.parameters():
            p.requires_grad_(True)
    lora = sum(p.numel() for p in lora_parameters(model))
    projector = param_count(model, lambda n: PROJECTOR in n) if train_projector else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == lora + projector
    return {
        "base_params": base_total,
        "trainable_params": trainable,
        "lora_params": lora,
        "projector_params": projector,
        "lora_modules": len(wrapped),
        "trainable_fraction": trainable / base_total,
    }


def _projector_name(model: nn.Module) -> str:
    names = [n for n, _ in model.named_modules() if n.endswith(PROJECTOR)]
    if len(names) != 1:
        raise ValueError(f"expected one {PROJECTOR}, found {names}")
    return names[0]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def save_adapter(model: nn.Module, out: str | Path, meta: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state_dict(model), out / "adapter.safetensors")
    (out / "adapter.json").write_text(json.dumps(meta, indent=2))


def load_adapter(model: nn.Module, path: str | Path) -> dict:
    """Recreate the LoRA modules described in adapter.json and load the trained weights."""
    path = Path(path)
    meta = json.loads((path / "adapter.json").read_text())
    lc = meta["lora"]
    apply_lora(model, scope="language_model", targets=tuple(lc["targets"]), r=lc["r"], alpha=lc["alpha"])
    state = load_file(path / "adapter.safetensors")
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"adapter weights not in model: {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            p = params[n]
            if ".lora_" in n:
                p.copy_(t.to(p.device))
            else:  # projector: keep the model's dtype
                p.copy_(t.to(device=p.device, dtype=p.dtype))
    n_lora = sum(1 for n in state if ".lora_A" in n)
    if n_lora != lc["modules"]:
        raise ValueError(f"adapter has {n_lora} LoRA modules, expected {lc['modules']}")
    return meta


def merge_and_save(model: nn.Module, processor, out: str | Path) -> int:
    n = merge_lora(model)
    out = Path(out)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    return n
