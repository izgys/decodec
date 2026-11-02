"""Offline tests for temperature scaling, on synthetic logits."""

import json

import numpy as np
import pytest
from conftest import make_record
from scipy.special import softmax

from src.calibration import temperature
from src.config import load_config
from src.visualisation import reliability_figure


def overconfident_logits(n=4000, n_classes=5, true_t=2.5, seed=0):
    """Labels drawn from softmax(logits / true_t), so true_t is the correct answer."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(n, n_classes))
    probs = softmax(logits / true_t, axis=1)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    return logits, labels


def test_fit_recovers_the_temperature():
    logits, labels = overconfident_logits(true_t=2.5)
    assert temperature.fit_temperature(logits, labels) == pytest.approx(2.5, abs=0.2)


def test_fit_leaves_a_calibrated_model_alone():
    logits, labels = overconfident_logits(true_t=1.0)
    assert temperature.fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.1)


def test_scaling_never_changes_a_prediction():
    logits, _ = overconfident_logits()
    assert (softmax(logits / 3.7, axis=1).argmax(1) == logits.argmax(1)).all()


def test_scaling_reduces_error_on_overconfident_logits():
    logits, labels = overconfident_logits()
    t = temperature.fit_temperature(logits, labels)
    before = temperature.expected_calibration_error(
        *temperature.confidence_and_correct(logits, labels, 1.0), 15
    )
    after = temperature.expected_calibration_error(
        *temperature.confidence_and_correct(logits, labels, t), 15
    )
    assert after < before / 2


def test_ece_by_hand():
    # Two bins of 10: confidence 0.2 with accuracy 0.0, and 0.9 with accuracy 0.5
    confidence = np.array([0.2, 0.2, 0.9, 0.9])
    correct = np.array([False, False, True, False])
    assert temperature.expected_calibration_error(confidence, correct, 10) == pytest.approx(
        0.5 * 0.2 + 0.5 * 0.4
    )


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert temperature.expected_calibration_error(
        np.array([1.0, 1.0]), np.array([True, True]), 15
    ) == pytest.approx(0.0)


def test_bins_hold_every_sequence_including_the_edges():
    confidence = np.array([0.0, 0.1, 0.100001, 0.5, 1.0])
    bins = temperature.reliability_bins(confidence, np.ones(5, dtype=bool), 10)
    assert sum(b["n_sequences"] for b in bins) == 5
    assert bins[0]["n_sequences"] == 2      # 0.0 and the upper edge 0.1
    assert bins[1]["n_sequences"] == 1
    assert bins[9]["n_sequences"] == 1


def test_cluster_check_finds_a_family_that_fails_with_confidence():
    logits = np.array([[5.0, 0.0]] * 4 + [[0.5, 0.0]] * 4)
    labels = np.array([1] * 4 + [0] * 4)
    clusters = np.array(["bad"] * 4 + ["fine"] * 4)
    records = [{"ec_subclass": "1.1", "protein_name": "x"}] * 8
    check = temperature.cluster_check(logits, labels, clusters, records, 1.0, min_size=3)
    assert check["n_clusters"] == 2
    assert check["clusters"][0]["cluster"] == "bad"
    assert check["clusters"][0]["accuracy"] == 0.0
    assert check["n_mostly_wrong_and_confident"] == 1


def write_inputs(tmp_path, misaligned=False):
    processed = tmp_path / "processed"
    processed.mkdir()
    logits, labels = overconfident_logits(n=240, n_classes=2)
    records, table = [], []
    for i in range(120):
        uid = f"q{i}"
        records.append(make_record(
            uid, label=int(labels[i]), cluster_id=f"c{i % 12}", ec_subclass="1.1",
            protein_name="enzyme",
        ))
        table.append({
            "uniprot_id": uid, "cluster_id": f"c{i % 12}", "n_hits": 1, "best_target": "t",
            "best_identity": 25.0 + i % 50, "max_identity_covered": None,
        })
    (processed / "test.json").write_text(json.dumps(records))
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "test_identity.json").write_text(json.dumps(table))

    predictions = tmp_path / "lora"
    predictions.mkdir()
    ids = [r["uniprot_id"] for r in records]
    np.savez(
        predictions / "test_predictions.npz",
        uniprot_ids=np.array(ids[::-1] if misaligned else ids),
        labels=labels[:120], clusters=np.array([f"c{i % 12}" for i in range(120)]),
        logits=logits[:120],
    )
    np.savez(
        predictions / "val_predictions.npz", uniprot_ids=np.array(ids), labels=labels[120:],
        clusters=np.array([f"v{i % 12}" for i in range(120)]), logits=logits[120:],
    )
    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{identity_dir.as_posix()}'",
        f"calibration.predictions_dir='{predictions.as_posix()}'",
        f"calibration.output_dir='{(tmp_path / 'results').as_posix()}'",
        "calibration.n_bootstrap=20",
        "calibration.min_cluster_size=5",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])


def test_run_end_to_end_and_draws_the_figure(tmp_path):
    cfg = write_inputs(tmp_path)
    temperature.run(cfg)
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results["temperature"] > 1
    view = results["views"]["all"]
    assert view["n_sequences"] == 120
    # Accuracy never moves with temperature; confidence does
    assert view["before"]["accuracy"] == view["after"]["accuracy"]
    assert view["after"]["mean_confidence"] < view["before"]["mean_confidence"]
    assert view["before"]["ci_low"] <= view["before"]["ece"] <= view["before"]["ci_high"]
    change = view["ece_change"]
    assert change["difference"] == pytest.approx(view["after"]["ece"] - view["before"]["ece"])
    assert change["ci_low"] <= change["ci_high"]
    assert results["validation"]["n_sequences"] == 120
    assert results["cluster_check"]["after"]["n_clusters"] == 12

    reliability_figure.run(cfg)
    for theme in ["light", "dark"]:
        figure = tmp_path / "results" / f"reliability_{theme}.png"
        assert figure.read_bytes().startswith(b"\x89PNG")


def test_misaligned_predictions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="not aligned"):
        temperature.run(write_inputs(tmp_path, misaligned=True))


def test_figure_needs_the_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="temperature"):
        reliability_figure.run(write_inputs(tmp_path))
