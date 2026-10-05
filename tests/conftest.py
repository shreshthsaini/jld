"""Shared fixtures and helpers.

The suite needs only the bundled examples and the DINOv2-S weights (downloaded
through timm on first use). It does not need any benchmark dataset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jld import JLD

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXPECTED = json.loads((EXAMPLES / "expected.json").read_text())


def reference_of(name: str) -> Path:
    return EXAMPLES / (name.split("_", 1)[0] + ".png")


@pytest.fixture(scope="session")
def metrics() -> dict[str, JLD]:
    """One metric per variant (full, fast, lens, raw) on the CPU."""
    return {variant: JLD.pretrained(variant, device="cpu") for variant in EXPECTED}


@pytest.fixture(scope="session")
def metric(metrics) -> JLD:
    return metrics["full"]


@pytest.fixture(scope="session")
def fast(metrics) -> JLD:
    return metrics["fast"]


def run(*args: str) -> subprocess.CompletedProcess:
    """Run ``python *args`` from the repository root with the package importable."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
