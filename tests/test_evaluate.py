"""The evaluation script, on a tiny stand-in dataset built from the bundled examples."""
from __future__ import annotations

import importlib.util
import json
import shutil

import numpy as np
import pytest

from conftest import EXAMPLES, EXPECTED, ROOT, reference_of

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kadid_split_file():
    split = json.loads((ROOT / "splits" / "kadid10k_split.json").read_text())
    dev, test = set(split["dev"]), set(split["test"])
    assert (len(dev), len(test)) == (16, 65)
    assert not dev & test
    assert dev | test == {f"I{index:02d}.png" for index in range(1, 82)}


def test_main_table_configuration(evaluate):
    assert evaluate.MAIN_TABLE == ["tid2013", "csiq", "live", "kadid10k_test", "pipal_val"]
    assert set(evaluate.MAIN_TABLE) <= set(evaluate.SPLITS)
    for row in evaluate.PAPER_SRCC.values():
        assert set(row) == set(evaluate.MAIN_TABLE)
    rows = {name: evaluate.DATASETS[evaluate.SPLITS[name][0]].rows for name in evaluate.SPLITS}
    assert (rows["tid2013"], rows["csiq"], rows["live"], rows["pipal_val"]) == (3000, 866, 779, 1000)


def test_correlations(evaluate):
    rng = np.random.default_rng(0)
    mos = rng.uniform(1, 5, 200)
    exact = evaluate.correlations(np.exp(-mos), mos)  # monotone decreasing, non-linear
    assert exact["SRCC"] == pytest.approx(1.0) and exact["KRCC"] == pytest.approx(1.0)
    assert exact["PLCC"] > 0.99 and exact["n"] == 200
    noisy = evaluate.correlations(mos + rng.normal(0, 1.0, 200), mos)
    assert 0.5 < noisy["SRCC"] < 1.0
    flipped = evaluate.correlations(-(mos + rng.normal(0, 1.0, 200)), mos)
    assert 0.5 < flipped["SRCC"] < 1.0  # absolute values: sign of the metric does not matter
    assert abs(evaluate.correlations(rng.normal(size=200), mos)["SRCC"]) < 0.25


def _stand_in_csiq(root):
    """The eight example pairs laid out like CSIQ, with made-up scores."""
    names = sorted(EXPECTED["fast"])
    (root / "meta").mkdir(parents=True)
    for folder in ("src_imgs", "dst_imgs"):
        (root / "csiq" / "CSIQ" / folder).mkdir(parents=True)
    for name in names:
        shutil.copy(reference_of(name), root / "csiq" / "CSIQ" / "src_imgs")
        shutil.copy(EXAMPLES / name, root / "csiq" / "CSIQ" / "dst_imgs")
    meta = pd.DataFrame({
        "ref_name": [reference_of(n).name for n in names],
        "dist_name": names,
        "dmos": [EXPECTED["fast"][n] for n in names],  # stand-in scores, not human ratings
    })
    meta.to_csv(root / "meta" / "meta_info_CSIQDataset.csv", index=False)
    return names


def test_end_to_end_on_stand_in_dataset(evaluate, tmp_path, monkeypatch, capsys):
    names = _stand_in_csiq(tmp_path / "data")
    monkeypatch.setitem(evaluate.DATASETS, "csiq", evaluate.DATASETS["csiq"]._replace(rows=len(names)))
    args = ["--datasets", "csiq", "--metrics", "jld_fast", "--data-root", str(tmp_path / "data"),
            "--out", str(tmp_path / "out"), "--device", "cpu", "--workers", "0", "--offline"]
    evaluate.main(args)
    scores = pd.read_csv(tmp_path / "out" / "csiq_scores.csv")
    assert scores.dist.tolist() == [f"csiq/CSIQ/dst_imgs/{n}" for n in names]
    assert np.allclose(scores.jld_fast, [EXPECTED["fast"][n] for n in names], rtol=1e-3, atol=1e-4)
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert summary.loc[0, "SRCC"] == pytest.approx(1.0) and summary.loc[0, "n"] == len(names)

    capsys.readouterr()
    evaluate.main(args)  # second run reuses the saved per-pair scores
    assert "pairs in" not in capsys.readouterr().out


def test_offline_refuses_to_download(evaluate, tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.load_dataset("tid2013", tmp_path, offline=True)
