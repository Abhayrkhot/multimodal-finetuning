import base64
import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mmft.serve import create_app  # noqa: E402


def _png(rgb=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (80, 50), rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_answer_endpoint(processor, tiny_model):
    client = TestClient(
        create_app("tiny", processor=processor, model=tiny_model, temperature=1.5, abstain_below=0.99)
    )
    assert client.get("/healthz").json()["status"] == "ok"
    r = client.post(
        "/v1/answer",
        json={"image_b64": _png(), "question": "What colour?", "choices": ["red", "green", "blue"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["probs"]) == 3 and abs(sum(body["probs"]) - 1) < 1e-5
    assert body["answer"] == ["red", "green", "blue"][body["index"]] and body["letter"] in "ABC"
    assert body["abstain"] is (body["confidence"] < 0.99)


def test_rejects_bad_input(processor, tiny_model):
    client = TestClient(create_app("tiny", processor=processor, model=tiny_model))
    bad = client.post(
        "/v1/answer", json={"image_b64": "bm90IGFuIGltYWdl", "question": "q", "choices": ["a", "b"]}
    )
    assert bad.status_code == 400
    one = client.post("/v1/answer", json={"image_b64": _png(), "question": "q", "choices": ["a"]})
    assert one.status_code == 422
