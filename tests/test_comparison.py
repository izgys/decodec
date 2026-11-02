"""Offline tests for the LoRA vs baselines comparison, on synthetic predictions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_record

from src.config import load_config
from src.visualisation import comparison_figure

REPO = Path(__file__).resolve().parent.parent


def load_module():
    path = REPO / "experiments" / "model_comparison" / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("run_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load_module()


def write_inputs(tmp_path, misaligned=None):
    processed = tmp_path / "processed"
    processed.mkdir()
    test, table = [], []
    for i in range(12):
        uid = f"q{i}"
        test.append(make_record(uid, label=i % 2, cluster_id=f"c{i // 2}"))
        row = {"uniprot_id": uid, "cluster_id": f"c{i // 2}", "n_hits": 1, "best_target": "t"}
        row.update({"best_identity": 25.0 + 5 * i, "max_identity_covered": None})
        table.append(row if i else {"uniprot_id": uid, "cluster_id": "c0", "n_hits": 0})
    (processed / "test.json").write_text(json.dumps(test))
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "3.4": 1}))
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "test_identity.json").write_text(json.dumps(table))

    ids = [r["uniprot_id"] for r in test]
    y_true = np.array([r["label"] for r in test])
    wrong_two = y_true.copy()
    wrong_two[[2, 3]] = 1 - wrong_two[[2, 3]]
    no_hit_wrong = y_true.copy()
    no_hit_wrong[0] = -1
    paths = {
        "lora": (tmp_path / "lora" / "test_predictions.npz", y_true),
        "probe": (tmp_path / "probe" / "test_predictions_mean.npz", wrong_two),
        "neighbour": (tmp_path / "similarity" / "test_predictions_neighbour.npz", no_hit_wrong),
    }
    for name, (path, predicted) in paths.items():
        path.parent.mkdir()
        saved = ids[::-1] if name == misaligned else ids
        np.savez(path, uniprot_ids=np.array(saved), predictions=predicted)

    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{identity_dir.as_posix()}'",
        f"comparison.lora_predictions='{paths['lora'][0].as_posix()}'",
        f"comparison.output_dir='{(tmp_path / 'results').as_posix()}'",
        "comparison.n_bootstrap=20",
        f"probe.output_dir='{paths['probe'][0].parent.as_posix()}'",
        f"similarity.output_dir='{paths['neighbour'][0].parent.as_posix()}'",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])


def test_run_scores_all_models_on_the_same_views(tmp_path):
    comparison.run(write_inputs(tmp_path))
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    everything = results["views"]["all"]
    assert everything["n_sequences"] == 12
    assert everything["models"]["lora"]["accuracy"] == 1.0
    assert everything["models"]["probe"]["accuracy"] == pytest.approx(10 / 12)
    assert everything["models"]["neighbour"]["accuracy"] == pytest.approx(11 / 12)
    assert everything["lora_minus_probe"]["accuracy"]["difference"] == pytest.approx(2 / 12)
    assert everything["lora_minus_neighbour"]["accuracy"]["difference"] == pytest.approx(1 / 12)
    assert results["views"]["no_hit"]["n_sequences"] == 1
    bins = ["no_hit", "<30", "30-40", "40-50", "50-70", ">=70"]
    assert sum(results["views"][b]["n_sequences"] for b in bins) == 12


def test_per_class_table_holds_every_model(tmp_path):
    comparison.run(write_inputs(tmp_path))
    rows = json.loads((tmp_path / "results" / "per_class.json").read_text())
    assert [r["subclass"] for r in rows] == ["1.1", "3.4"]
    assert all(r["lora_f1"] == 1.0 and r["probe_f1"] < 1.0 for r in rows)
    assert rows[0]["support"] == 6 and rows[0]["n_clusters"] == 6


def test_figure_is_drawn_for_each_theme(tmp_path):
    cfg = write_inputs(tmp_path)
    comparison.run(cfg)
    comparison_figure.run(cfg)
    for theme in ["light", "dark"]:
        path = tmp_path / "results" / f"model_comparison_{theme}.png"
        assert path.read_bytes().startswith(b"\x89PNG")


def test_figure_needs_the_comparison_results(tmp_path):
    cfg = write_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="run_comparison"):
        comparison_figure.run(cfg)


@pytest.mark.parametrize("model", ["lora", "probe", "neighbour"])
def test_misaligned_predictions_are_rejected(tmp_path, model):
    with pytest.raises(ValueError, match=f"{model} predictions are not aligned"):
        comparison.run(write_inputs(tmp_path, misaligned=model))
