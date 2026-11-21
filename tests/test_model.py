"""End-to-end checks on a tiny random LLaVA with the real processor (CPU, fp32)."""

import dataclasses
import json

import torch

from mmft.batching import Collator, Encoder, chat_text
from mmft.config import TrainConfig
from mmft.evaluate import letter_logits, predict, summarize
from mmft.model import LM_TARGETS, load_adapter, prepare_for_training
from mmft.train import train
from tests.conftest import make_tiny_llava


def test_letter_ids_and_encoding(processor, color_items):
    enc = Encoder(processor)
    tok = processor.tokenizer
    assert enc.letter_ids == tok.convert_tokens_to_ids([f"▁{c}" for c in "ABCDE"])
    out = enc.encode(color_items[0])
    ids = out["input_ids"].tolist()
    assert ids.count(processor.tokenizer.convert_tokens_to_ids("<image>")) == 576
    assert out["pixel_values"].shape == (3, 336, 336)
    assert tok.decode(ids[-6:]).endswith("ASSISTANT:")
    assert chat_text(color_items[0]).count("<image>") == 1


def test_collator_left_pads_and_appends_targets(processor, color_items):
    enc = Encoder(processor)
    a, b = enc.encode(color_items[0]), enc.encode(color_items[1])
    b["input_ids"] = b["input_ids"][:-3]  # make lengths differ
    batch = Collator(enc.pad_id, enc.letter_ids, enc.eos_id, train=True)([a, b])
    la, lb = len(a["input_ids"]) + 2, len(b["input_ids"]) + 2
    assert batch["input_ids"].shape == (2, la)
    assert batch["attention_mask"][1].tolist() == [0] * (la - lb) + [1] * lb
    assert batch["position_ids"][1, -1].item() == lb - 1
    assert batch["input_ids"][0, -2:].tolist() == [enc.letter_ids[a["answer"]], enc.eos_id]


def test_padding_does_not_change_scores(processor, tiny_model, color_items):
    """A question scored alone must get the same letter logits as when left-padded in a batch."""
    enc = Encoder(processor)
    collate = Collator(enc.pad_id, enc.letter_ids, enc.eos_id)
    ids = torch.tensor(enc.letter_ids)
    long_item = dataclasses.replace(color_items[0], question="Look closely. " * 20)
    alone = letter_logits(tiny_model, collate([enc.encode(color_items[1])]), ids)
    batched = letter_logits(tiny_model, collate([enc.encode(long_item), enc.encode(color_items[1])]), ids)
    torch.testing.assert_close(batched[1], alone[0], rtol=1e-4, atol=1e-4)


def test_param_counts(tiny_model):
    total = sum(p.numel() for p in tiny_model.parameters())
    c = prepare_for_training(tiny_model, r=4, alpha=8, dropout=0.0)
    assert c["base_params"] == total
    assert c["lora_modules"] == 2 * 7  # 2 layers x q,k,v,o,gate,up,down
    # q,k,v,o: 64->64; gate,up: 64->128; down: 128->64
    assert c["lora_params"] == 2 * (4 * 4 * 128 + 2 * 4 * 192 + 4 * 192)
    assert c["trainable_params"] == c["lora_params"] + c["projector_params"]
    assert all(p.dtype == torch.float32 for p in tiny_model.parameters() if p.requires_grad)


def test_train_save_load_merge(processor, color_items, tmp_path):
    cfg = TrainConfig(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        batch_size=8,
        grad_accum=1,
        epochs=3,
        lr=5e-3,
        projector_lr=5e-3,
        warmup_ratio=0.1,
        num_workers=0,
        log_every=1,
        eval_every=0,
        gradient_checkpointing=False,
    )
    model = make_tiny_llava()
    summary = train(
        cfg,
        tmp_path / "run",
        device="cpu",
        processor=processor,
        model=model,
        train_items=color_items,
        val_items=color_items[:16],
    )
    log = [
        r
        for r in map(json.loads, (tmp_path / "run" / "train_log.jsonl").read_text().splitlines())
        if "loss" in r
    ]
    assert log[-1]["loss"] < 0.5 * log[0]["loss"]  # learns to answer with a letter
    assert summary["optimizer_steps"] == 3 * 64 // 8

    enc = Encoder(processor)
    ref = predict(model, enc, color_items[:8], batch_size=4, num_workers=0, log_every=0)

    fresh = make_tiny_llava()  # same seed -> same frozen base weights
    meta = load_adapter(fresh, tmp_path / "run")
    assert meta["lora"]["modules"] == 14
    loaded = predict(fresh, enc, color_items[:8], batch_size=4, num_workers=0, log_every=0)
    for r1, r2 in zip(ref, loaded, strict=True):
        assert abs(max(r1["logits"]) - max(r2["logits"])) < 1e-3

    from mmft.lora import merge_lora

    merge_lora(fresh)
    merged = predict(fresh, enc, color_items[:8], batch_size=4, num_workers=0, log_every=0)
    assert summarize(merged)["accuracy"] == summarize(ref)["accuracy"]
    for r1, r2 in zip(ref, merged, strict=True):
        assert max(abs(a - b) for a, b in zip(r1["logits"], r2["logits"], strict=True)) < 1e-2


def test_vision_adapters_skip_unused_clip_layers(processor, color_items, tmp_path):
    """CLIP features come from the penultimate layer, so only the layers before it get adapters,
    and every adapter receives a gradient."""
    from mmft.model import VISION_TARGETS, save_adapter

    model = make_tiny_llava()
    c = prepare_for_training(model, r=4, alpha=8, dropout=0.0, vision_targets=VISION_TARGETS)
    assert c["vision_lora_modules"] == 6  # tiny CLIP: 2 layers, layer 1 unused -> 6 linears in layer 0
    assert not any(
        "layers.1." in n for n, _ in model.named_parameters() if "vision_tower" in n and ".lora_" in n
    )

    enc = Encoder(processor)
    batch = Collator(enc.pad_id, enc.letter_ids, enc.eos_id, train=True)(
        [enc.encode(it) for it in color_items[:2]]
    )
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        pixel_values=batch["pixel_values"],
        logits_to_keep=3,
    )
    out.logits.float().pow(2).mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)

    for n, p in model.named_parameters():
        if ".lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    meta = {
        "lora": {"r": 4, "alpha": 8, "targets": list(LM_TARGETS), "modules": c["lora_modules"]},
        "vision_lora": {"targets": list(VISION_TARGETS), "modules": 6},
    }
    save_adapter(model, tmp_path / "a", meta)
    fresh = make_tiny_llava()
    load_adapter(fresh, tmp_path / "a")
    ids = torch.tensor(enc.letter_ids)
    torch.testing.assert_close(letter_logits(fresh, batch, ids), letter_logits(model, batch, ids))
