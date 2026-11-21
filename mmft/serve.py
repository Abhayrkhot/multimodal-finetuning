"""Minimal HTTP API over a merged model: POST an image and a multiple-choice question, get the
answer with a temperature-scaled confidence, and an abstain flag when that confidence is below a
threshold chosen from validation data."""

from __future__ import annotations

import base64
import io
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .batching import Collator, Encoder
from .data import LETTERS, MCItem
from .evaluate import letter_logits


class AnswerRequest(BaseModel):
    image_b64: str = Field(description="base64-encoded JPEG or PNG")
    question: str = Field(min_length=1, max_length=2000)
    choices: list[str] = Field(min_length=2, max_length=len(LETTERS))
    context: str = ""


class AnswerResponse(BaseModel):
    index: int
    letter: str
    answer: str
    confidence: float
    probs: list[float]
    abstain: bool
    latency_ms: float


def create_app(
    model_dir: str | None = None,
    *,
    quant: str | None = None,
    temperature: float = 1.0,
    abstain_below: float = 0.0,
    processor=None,
    model=None,
) -> FastAPI:
    if model is None:
        from .model import load_model

        processor, model = load_model(model_dir, quant=quant)
    enc = Encoder(processor)
    collate = Collator(enc.pad_id, enc.letter_ids, enc.eos_id, train=False)
    letter_ids = torch.tensor(enc.letter_ids, device=next(model.parameters()).device)
    gpu = threading.Lock()  # FastAPI runs sync handlers in a thread pool; one forward at a time
    app = FastAPI(title="mmft", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "model": model_dir,
            "quant": quant or "bf16",
            "temperature": temperature,
            "abstain_below": abstain_below,
        }

    @app.post("/v1/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest):
        start = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(base64.b64decode(req.image_b64, validate=True)))
            img.load()
        except (ValueError, UnidentifiedImageError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"bad image: {e}") from e
        item = MCItem("request", "", req.question, tuple(req.choices), 0, "", req.context)
        batch = collate([enc.encode(item, image=img)])
        with gpu, torch.no_grad():
            logits = letter_logits(model, batch, letter_ids)[0, : len(req.choices)]
        probs = torch.softmax(logits / temperature, -1).tolist()
        k = max(range(len(probs)), key=probs.__getitem__)
        return AnswerResponse(
            index=k,
            letter=LETTERS[k],
            answer=req.choices[k],
            confidence=probs[k],
            probs=probs,
            abstain=probs[k] < abstain_below,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    return app
