"""Datasets as multiple-choice items: Visual7W "telling" (train/eval) and ScienceQA-IMG (harness check).

Every item becomes the same prompt shape LLaVA-1.5 was evaluated with on multiple-choice
benchmarks: question, lettered options, then "Answer with the option's letter ...". The answer
position is shuffled with a per-question seed, so it is uniform over A-D and identical across runs.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

V7W_QA_URL = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
V7W_QA_SHA256 = "fb7b4c869c9be83e76f7ed24d188f731f642ea25381a0db83cd96e23e8b6953e"
V7W_IMG_URL = "https://vision.stanford.edu/yukezhu/visual7w_images.zip"
V7W_IMG_SHA256 = "PENDING"
V7W_TYPES = ("what", "where", "when", "who", "why", "how")

LETTERS = "ABCDE"
INSTRUCTION = "Answer with the option's letter from the given choices directly."


@dataclass(frozen=True)
class MCItem:
    qid: str
    image: str  # path to the image file
    question: str
    choices: tuple[str, ...]
    answer: int  # index into choices
    group: str  # slice label: Visual7W question type, ScienceQA subject
    context: str = ""


def format_prompt(item: MCItem) -> str:
    lines = [f"Context: {item.context}"] if item.context else []
    lines.append(item.question)
    lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices, strict=False)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def shuffle_choices(qid: str, choices: list[str], answer: int, seed: int) -> tuple[tuple[str, ...], int]:
    order = list(range(len(choices)))
    random.Random(f"{seed}:{qid}").shuffle(order)
    return tuple(choices[i] for i in order), order.index(answer)


def _norm(s: str) -> str:
    return s.strip().lower().rstrip(".")


def load_v7w(root: str | Path, split: str, *, seed: int = 0, limit: int | None = None) -> list[MCItem]:
    """Visual7W telling items for one split ("train", "val", "test").

    `limit` takes a seeded random subset (not the first N, which would cluster by image).
    Items whose four options are not distinct after normalisation are dropped (2 of 139,868).
    """
    root = Path(root)
    meta = json.loads((root / "dataset_v7w_telling.json").read_text())
    items = []
    for im in meta["images"]:
        if im["split"] != split:
            continue
        path = str(root / "images" / im["filename"])
        for qa in im["qa_pairs"]:
            raw = [qa["answer"], *qa["multiple_choices"]]
            if len({_norm(c) for c in raw}) != len(raw):
                continue
            qid = str(qa["qa_id"])
            choices, answer = shuffle_choices(qid, raw, 0, seed)
            items.append(MCItem(qid, path, qa["question"], choices, answer, qa["type"]))
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str) -> None:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        tmp.rename(dest)
    got = _sha256(dest)
    if sha256 != "PENDING" and got != sha256:
        raise RuntimeError(f"{dest.name}: sha256 {got} != pinned {sha256}")
    print(f"{dest.name}: sha256 {got}")


def download_v7w(root: str | Path) -> None:
    """Fetch the Visual7W telling QA (6 MB) and images (1.9 GB) from Stanford, verify, unpack."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    qa_zip, img_zip = root / "dataset_v7w_telling.zip", root / "visual7w_images.zip"
    _fetch(V7W_QA_URL, qa_zip, V7W_QA_SHA256)
    _fetch(V7W_IMG_URL, img_zip, V7W_IMG_SHA256)
    if not (root / "dataset_v7w_telling.json").exists():
        with zipfile.ZipFile(qa_zip) as z:
            z.extract("dataset_v7w_telling.json", root)
    if not (root / "images").is_dir():
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(root)
    n = sum(1 for _ in (root / "images").glob("*.jpg"))
    print(f"visual7w: {n} images under {root / 'images'}")


SQA_REPO = "lmms-lab/ScienceQA-IMG"
SQA_REVISION = "b9984af01798586705cee1fbdfa80a6dcbef04f1"
SQA_TEST_FILE = "data/test-00000-of-00001.parquet"


def prepare_sqa(root: str | Path) -> None:
    """Write ScienceQA-IMG test (2,017 questions with an image) to disk as images + JSONL, from a
    pinned revision of the lmms-eval parquet."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(SQA_REPO, SQA_TEST_FILE, repo_type="dataset", revision=SQA_REVISION)
    ds = Dataset.from_parquet(path)
    with (root / "test.jsonl").open("w") as f:
        for i, ex in enumerate(ds):
            img_path = root / "images" / f"{i:05d}.png"
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path)
            row = {k: ex[k] for k in ("question", "choices", "answer", "hint", "subject", "grade")}
            f.write(json.dumps({"qid": f"sqa-{i:05d}", "image": img_path.name, **row}) + "\n")
    print(f"scienceqa-img: {len(ds)} test questions under {root}")


def load_sqa(root: str | Path, *, seed: int = 0, limit: int | None = None) -> list[MCItem]:
    """ScienceQA-IMG test, used to check the harness against LLaVA-1.5's published 66.8% and to
    measure forgetting after fine-tuning. Options keep their original order, as in LLaVA's eval."""
    root = Path(root)
    items = []
    for line in (root / "test.jsonl").read_text().splitlines():
        r = json.loads(line)
        items.append(
            MCItem(
                r["qid"],
                str(root / "images" / r["image"]),
                r["question"],
                tuple(r["choices"]),
                int(r["answer"]),
                r["subject"],
                context=(r["hint"] or "").strip(),
            )
        )
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items


def load_items(dataset: str, root: str | Path, split: str = "test", **kw) -> list[MCItem]:
    if dataset == "v7w":
        return load_v7w(root, split, **kw)
    if dataset == "sqa":
        return load_sqa(root, **kw)
    raise ValueError(f"unknown dataset {dataset!r}")
