"""Provenance recorded next to every result: code version, library versions, GPU."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

import torch


def git_sha() -> str:
    if os.environ.get("MMFT_GIT_SHA"):
        return os.environ["MMFT_GIT_SHA"]
    stamp = Path(__file__).resolve().parent.parent / "GIT_SHA"
    if stamp.exists():
        return stamp.read_text().strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def run_meta() -> dict:
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {
        "git_sha": git_sha(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if gpu else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
    }
