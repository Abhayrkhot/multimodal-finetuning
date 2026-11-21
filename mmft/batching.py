"""Turn MCItems into model inputs.

Prompts use LLaVA-1.5's vicuna_v1 conversation template. Images are padded to a square with the
CLIP mean colour before resizing (LLaVA-1.5's `image_aspect_ratio=pad`), so nothing is cropped.

Batches are left-padded with explicit position_ids. The answer letter is then always the last
position, so training and scoring only need logits for the final few positions
(`logits_to_keep`), not a [batch, seq, 32k] tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from .data import LETTERS, MCItem, format_prompt

SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)


def chat_text(item: MCItem) -> str:
    return f"{SYSTEM} USER: <image>\n{format_prompt(item)} ASSISTANT:"


def expand2square(img: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def letter_token_ids(tokenizer) -> list[int]:
    """Token id of each option letter as the model would emit it right after 'ASSISTANT:'."""
    prefix = tokenizer("ASSISTANT:", add_special_tokens=False).input_ids
    ids = []
    for letter in LETTERS:
        full = tokenizer(f"ASSISTANT: {letter}", add_special_tokens=False).input_ids
        if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"letter {letter!r} does not tokenize to one token after the prompt")
        ids.append(full[-1])
    return ids


@dataclass
class Encoder:
    processor: object

    def __post_init__(self):
        tok = self.processor.tokenizer
        self.letter_ids = letter_token_ids(tok)
        self.eos_id = tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ip = self.processor.image_processor
        self.fill = tuple(int(round(x * 255)) for x in ip.image_mean)

    def image(self, item: MCItem, blind: bool = False) -> Image.Image:
        if blind:  # the mean colour normalises to ~0: a "no image" control with the same token count
            return Image.new("RGB", (336, 336), self.fill)
        with Image.open(item.image) as im:
            return expand2square(im.convert("RGB"), self.fill)

    def encode(self, item: MCItem, blind: bool = False, image: Image.Image | None = None) -> dict:
        img = expand2square(image.convert("RGB"), self.fill) if image is not None else self.image(item, blind)
        out = self.processor(text=chat_text(item), images=img, return_tensors="pt")
        return {
            "input_ids": out["input_ids"][0],
            "pixel_values": out["pixel_values"][0],
            "answer": item.answer,
            "n_choices": len(item.choices),
        }


class MCDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[MCItem], encoder: Encoder, blind: bool = False):
        self.items, self.encoder, self.blind = items, encoder, blind

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.encoder.encode(self.items[i], self.blind)


@dataclass
class Collator:
    pad_id: int
    letter_ids: list[int]
    eos_id: int
    train: bool = False

    def __call__(self, batch: list[dict]) -> dict:
        seqs = []
        for b in batch:
            ids = b["input_ids"]
            if self.train:  # target: "<letter></s>" appended after "ASSISTANT:"
                tail = torch.tensor([self.letter_ids[b["answer"]], self.eos_id], dtype=ids.dtype)
                ids = torch.cat([ids, tail])
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, width - len(s) :] = s
            attention_mask[i, width - len(s) :] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "answer": torch.tensor([b["answer"] for b in batch]),
            "n_choices": torch.tensor([b["n_choices"] for b in batch]),
        }


def make_loader(
    items,
    encoder: Encoder,
    *,
    batch_size: int,
    train: bool,
    num_workers: int = 4,
    blind: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        MCDataset(items, encoder, blind),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=Collator(encoder.pad_id, encoder.letter_ids, encoder.eos_id, train),
        generator=g,
    )
