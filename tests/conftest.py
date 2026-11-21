import json
import random

import pytest
import torch
from PIL import Image

from mmft.data import MCItem

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
COLORS = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220), "yellow": (230, 220, 20)}


@pytest.fixture(scope="session")
def processor():
    """The real LLaVA-1.5 processor (tokenizer + CLIP image processor, ~2 MB, no weights)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoProcessor.from_pretrained(MODEL_ID)
    except OSError as e:  # offline
        pytest.skip(f"cannot fetch {MODEL_ID} processor: {e}")


def make_tiny_llava(seed: int = 0):
    """Randomly initialised LLaVA with the real geometry (336px, 14px patches -> 576 image tokens,
    32k vocab) and tiny widths, so the full pipeline runs on a CPU in seconds."""
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    torch.manual_seed(seed)
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=336,
        patch_size=14,
        projection_dim=32,
    )
    text = LlamaConfig(
        vocab_size=32064,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        initializer_range=0.2,
        pad_token_id=32001,
    )
    cfg = LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=32000,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        projector_hidden_act="gelu",
    )
    return LlavaForConditionalGeneration(cfg).eval()


@pytest.fixture
def tiny_model():
    return make_tiny_llava()


@pytest.fixture
def color_items(tmp_path):
    """Solid-colour images with 'What colour is this?' questions and shuffled options."""
    paths = {}
    for name, rgb in COLORS.items():
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (64 + len(name) * 8, 48), rgb).save(p)  # non-square: exercises padding
        paths[name] = str(p)
    rng = random.Random(0)
    items = []
    for i in range(64):
        name = rng.choice(list(COLORS))
        opts = list(COLORS)
        rng.shuffle(opts)
        items.append(
            MCItem(
                f"c{i}",
                paths[name],
                "What colour is this image?",
                tuple(opts),
                opts.index(name),
                "colour" if i % 2 else "hue",
            )
        )
    return items


@pytest.fixture
def v7w_root(tmp_path):
    """A miniature dataset_v7w_telling.json with the real schema."""
    images = []
    k = 0
    for split, n_img in (("train", 6), ("val", 2), ("test", 3)):
        for _ in range(n_img):
            img_id = 1000 + len(images)
            qas = []
            for t in ("what", "where", "why"):
                k += 1
                qas.append(
                    {
                        "image_id": img_id,
                        "qa_id": k,
                        "type": t,
                        "question": f"Q{k}?",
                        "answer": f"right {k}.",
                        "multiple_choices": [f"wrong a{k}.", f"wrong b{k}.", f"wrong c{k}."],
                    }
                )
            images.append(
                {"image_id": img_id, "split": split, "filename": f"v7w_{img_id}.jpg", "qa_pairs": qas}
            )
    # one item whose distractor repeats the answer: must be dropped
    images[0]["qa_pairs"].append(
        {
            "image_id": 1000,
            "qa_id": 999,
            "type": "who",
            "question": "Dup?",
            "answer": "Polar Bear.",
            "multiple_choices": ["Elephant.", "Panther.", "Polar bear."],
        }
    )
    (tmp_path / "dataset_v7w_telling.json").write_text(json.dumps({"images": images}))
    (tmp_path / "images").mkdir()
    for im in images:
        Image.new("RGB", (40, 30), (100, 120, 140)).save(tmp_path / "images" / im["filename"])
    return tmp_path
