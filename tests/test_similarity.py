"""Offline tests for the nearest-neighbour baseline, on synthetic predictions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_record

from src import metrics
from src.config import load_config

REPO = Path(__file__).resolve().parent.parent


def load_module():
    path = REPO / "experiments" / "baseline_similarity" / "run_similarity.py"
    spec = importlib.util.spec_from_file_location("run_similarity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


similarity = load_module()


def hit(target, best, covered=None):
    return {
        "n_hits": 1, "best_target": target, "best_identity": best,
        "max_identity": best, "max_identity_covered": covered,
    }


def test_prediction_copies_the_best_hit_label_and_marks_no_hit():
    table = [hit("t1", 40.0), {"n_hits": 0}, hit("t2", 25.0)]
    predicted = similarity.predict_from_best_hit(table, {"t1": 3, "t2": 7})
    assert predicted.tolist() == [3, similarity.NO_PREDICTION, 7]


def test_views_partition_by_best_hit_identity_at_the_edges():
    table = [
        {"n_hits": 0},
        hit("t", 29.9), hit("t", 30.0), hit("t", 39.9), hit("t", 40.0), hit("t", 70.0),
    ]
    views = similarity.build_views(table, 30, [30, 40])
    assert views["no_hit"].tolist() == [True, False, False, False, False, False]
    assert views["<30"].tolist() == [False, True, False, False, False, False]
    assert views["30-40"].tolist() == [False, False, True, True, False, False]
    assert views[">=40"].tolist() == [False, False, False, False, True, True]
    bins = np.array([views[k] for k in ["no_hit", "<30", "30-40", ">=40"]])
    assert (bins.sum(axis=0) == 1).all()


def test_relative_free_uses_covered_identity_only():
    table = [
        hit("t", 80.0, covered=None),   # high identity, partial coverage: no relative
        hit("t", 40.0, covered=29.9),   # covered, below the threshold
        hit("t", 40.0, covered=30.0),   # covered, at the threshold: a relative
        {"n_hits": 0},
    ]
    views = similarity.build_views(table, 30, [30])
    assert views["relative_free"].tolist() == [True, True, False, True]


def test_no_hit_counts_as_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, similarity.NO_PREDICTION, 1, 1])
    clusters = np.array(["a", "b", "c", "d"])
    view = similarity.score_view(
        y_true, {"neighbour": y_pred, "probe": y_true}, clusters, 20, 42
    )
    assert view["models"]["neighbour"]["accuracy"] == 0.75
    assert view["models"]["neighbour"]["macro_f1"] < 1.0
    assert view["models"]["probe"]["macro_f1"] == 1.0
    assert view["n_classes"] == 2


def test_macro_f1_covers_only_classes_in_the_view():
    # Class 5 is predicted once but absent from this view's true labels
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 5])
    view = similarity.score_view(
        y_true, {"neighbour": y_pred, "probe": y_pred}, np.array(list("abcd")), 20, 42
    )
    expected = metrics.macro_f1(y_true, y_pred, [0, 1])
    assert view["models"]["neighbour"]["macro_f1"] == pytest.approx(expected)


def test_accuracy_difference_is_paired():
    y_true, y_pred = np.array([0, 0, 1, 1, 1, 0]), np.array([0, 1, 1, 0, 1, 0])
    clusters = np.array(list("abcdef"))
    same = metrics.paired_accuracy_difference_ci(y_true, y_pred, y_pred, clusters, 50)
    assert same == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    better = metrics.paired_accuracy_difference_ci(y_true, y_true, y_pred, clusters, 200)
    assert better["difference"] == pytest.approx(2 / 6)


def test_hierarchical_metrics_treat_no_prediction_as_wrong():
    subclasses = {0: "1.1", 1: "3.4"}
    result = metrics.hierarchical_metrics(np.array([0, 1]), np.array([-1, 1]), subclasses)
    assert result["top_level_accuracy"] == 0.5
    assert result["errors_within_top_level"] == 0.0


def write_inputs(tmp_path, misaligned=False):
    processed = tmp_path / "processed"
    processed.mkdir()
    train = [make_record("t0", label=0), make_record("t1", label=1)]
    test, table = [], []
    for i in range(12):
        label = i % 2
        uid = f"q{i}"
        test.append(make_record(uid, label=label, cluster_id=f"c{i // 2}"))
        if i == 0:
            table.append({"uniprot_id": uid, "cluster_id": "c0", "n_hits": 0})
        else:
            table.append({
                "uniprot_id": uid, "cluster_id": f"c{i // 2}",
                **hit(f"t{label}", 25.0 + 5 * i, covered=(35.0 if i % 3 == 0 else None)),
            })
    (processed / "train.json").write_text(json.dumps(train))
    (processed / "test.json").write_text(json.dumps(test))
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "3.4": 1}))

    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "test_identity.json").write_text(json.dumps(table))

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    ids = [r["uniprot_id"] for r in test]
    if misaligned:
        ids = ids[::-1]
    np.savez(
        probe_dir / "test_predictions_mean.npz",
        uniprot_ids=np.array(ids), predictions=np.array([r["label"] for r in test]),
    )
    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{identity_dir.as_posix()}'",
        f"probe.output_dir='{probe_dir.as_posix()}'",
        f"similarity.output_dir='{(tmp_path / 'results').as_posix()}'",
        "similarity.n_bootstrap=20",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])


def test_run_end_to_end(tmp_path):
    similarity.run(write_inputs(tmp_path))
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    views = results["views"]
    assert views["all"]["n_sequences"] == 12
    assert views["no_hit"]["n_sequences"] == 1
    # q3, q6, q9 have a covered relative at 35%; the rest are relative-free
    assert views["relative_free"]["n_sequences"] == 9
    assert views["all"]["models"]["probe"]["accuracy"] == 1.0
    assert views["all"]["models"]["neighbour"]["accuracy"] == pytest.approx(11 / 12)
    assert views["all"]["probe_minus_neighbour"]["accuracy"]["difference"] == pytest.approx(1 / 12)
    bins = ["no_hit", "<30", "30-40", "40-50", "50-70", ">=70"]
    assert sum(views[b]["n_sequences"] for b in bins if b in views) == 12
    assert (tmp_path / "results" / "per_class_neighbour.json").exists()


def test_misaligned_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="not aligned"):
        similarity.run(write_inputs(tmp_path, misaligned=True))
