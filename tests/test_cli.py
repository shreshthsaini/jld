"""Smoke tests of the command line entry points."""
from __future__ import annotations

import re

import pytest
from PIL import Image

from conftest import EXAMPLES, EXPECTED, run
from jld import JLD, Lens

REF, DIST = "examples/parrots.png", "examples/parrots_jpeg_q8.png"


def _score(stdout: str) -> float:
    return float(re.search(r"= ([0-9.]+)", stdout).group(1))


@pytest.mark.parametrize("variant", ["full", "fast"])
def test_jld_command_scores_a_pair(variant):
    done = run("-m", "jld", REF, DIST, "--variant", variant, "--device", "cpu")
    assert done.returncode == 0, done.stderr
    assert f"JLD ({variant})" in done.stdout
    assert _score(done.stdout) == pytest.approx(EXPECTED[variant]["parrots_jpeg_q8.png"], abs=2e-4)


def test_jld_command_writes_the_local_map(tmp_path):
    out = tmp_path / "map.png"
    done = run("-m", "jld", REF, DIST, "--variant", "fast", "--device", "cpu", "--map", str(out))
    assert done.returncode == 0, done.stderr
    assert "21x28 patches" in done.stdout
    with Image.open(out) as image:
        assert image.size == (392, 294) and image.mode == "RGB"


def test_jld_command_needs_two_images():
    done = run("-m", "jld", REF)
    assert done.returncode == 2 and "usage: jld" in done.stderr


@pytest.mark.parametrize("script", ["demo.py", "fit_lens.py", "benchmark.py", "evaluate.py"])
def test_script_help(script):
    if script == "evaluate.py":
        pytest.importorskip("pandas")
        pytest.importorskip("scipy")
    done = run(f"scripts/{script}", "--help")
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_demo_ranks_the_examples():
    done = run("scripts/demo.py", "--variant", "fast", "--device", "cpu")
    assert done.returncode == 0, done.stderr
    rows = [line.split() for line in done.stdout.splitlines() if re.match(r"\s+\d+\s", line)]
    assert len(rows) == len(EXPECTED["fast"])
    scores = [float(row[3]) for row in rows]
    assert scores == sorted(scores)
    assert {f"{row[1]}_{row[2]}.png" for row in rows} == set(EXPECTED["fast"])


def test_fit_lens_script(tmp_path):
    out = tmp_path / "tiny.npz"
    done = run("scripts/fit_lens.py", "--images", str(EXAMPLES), "--out", str(out), "--n-images", "2",
               "--n-crops", "1", "--crop", "112", "--n-probes", "1", "--k", "8", "--batch", "2",
               "--device", "cpu", "--compare", "shipped")
    assert done.returncode == 0, done.stderr
    assert "subspace overlap" in done.stdout
    lens = Lens.load(out)
    assert lens.U.shape == (384, 8)
    assert lens.meta["n_images"] == 2 and "fit_seconds" in lens.meta
    assert JLD.from_lens(out, device="cpu").U.shape == (384, 8)


def test_benchmark_script():
    done = run("scripts/benchmark.py", "--device", "cpu", "--height", "70", "--width", "84",
               "--repeats", "2", "--warmup", "1", "--threads", "1")
    assert done.returncode == 0, done.stderr
    rows = {line[:20].strip(): line[20:].split() for line in done.stdout.splitlines()[2:]}
    assert set(rows) == {"JLD", "JLD-fast"}
    assert float(rows["JLD"][1]) > float(rows["JLD-fast"][1]) > 0     # GFLOPs
    assert float(rows["JLD"][2]) > 0 and float(rows["JLD-fast"][2]) > 0  # ms
