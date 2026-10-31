"""Tests for src/metrics.py: point metrics and the stratified cluster bootstrap."""

import numpy as np
import pytest

from src import metrics

SUBCLASSES = {0: "1.1", 1: "1.2", 2: "3.4"}


def test_macro_f1_counts_unpredicted_classes_as_zero():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 2, 2])          # class 1 never predicted
    assert metrics.macro_f1(y_true, y_pred, [0, 1, 2]) == pytest.approx((2 / 3 + 0 + 1) / 3)


def test_top_k_accuracy():
    y_true = np.array([0, 1, 2])
    scores = np.array([[0.9, 0.1, 0.0], [0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
    assert metrics.top_k_accuracy(y_true, scores, 1) == pytest.approx(1 / 3)
    assert metrics.top_k_accuracy(y_true, scores, 2) == pytest.approx(2 / 3)
    assert metrics.top_k_accuracy(y_true, scores, 3) == 1.0


def test_per_class_table_reports_support_and_clusters():
    y_true = np.array([0, 0, 0, 2])
    y_pred = np.array([0, 0, 2, 2])
    clusters = ["a", "a", "b", "c"]
    rows = metrics.per_class_table(y_true, y_pred, clusters, SUBCLASSES)
    table = {row["subclass"]: row for row in rows}
    assert table["1.1"]["support"] == 3 and table["1.1"]["n_clusters"] == 2
    assert table["1.2"]["support"] == 0 and table["1.2"]["f1"] == 0.0
    assert table["3.4"]["recall"] == 1.0 and table["3.4"]["precision"] == 0.5


def test_hierarchical_metrics():
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 1, 0, 2])                # 1.1->1.2 near miss, 3.4->1.1 far miss
    result = metrics.hierarchical_metrics(y_true, y_pred, SUBCLASSES)
    assert result["top_level_accuracy"] == pytest.approx(3 / 4)
    assert result["errors_within_top_level"] == pytest.approx(1 / 2)


def make_clustered(n_clusters_per_class=6, size=5, seed=0):
    rng = np.random.default_rng(seed)
    y_true, clusters = [], []
    for label in range(3):
        for c in range(n_clusters_per_class):
            y_true += [label] * size
            clusters += [f"{label}-{c}"] * size
    y_true = np.array(y_true)
    noisy = np.where(rng.random(len(y_true)) < 0.3, rng.integers(0, 3, len(y_true)), y_true)
    return y_true, noisy, clusters


def test_cluster_strata_groups_by_majority_label():
    y_true, _, clusters = make_clustered(n_clusters_per_class=4)
    strata = metrics.cluster_strata(y_true, clusters)
    assert [len(s) for s in strata] == [4, 4, 4]
    assert all(len(set(y_true[idx])) == 1 for s in strata for idx in s)


def test_bootstrap_keeps_cluster_counts_per_class():
    y_true, _, clusters = make_clustered(n_clusters_per_class=4, size=3)
    clusters = np.array(clusters)

    def clusters_per_class(idx):
        # Resampled with replacement: count cluster draws (sequences / cluster size)
        counts = np.bincount(y_true[idx], minlength=3) / 3
        assert (counts == 4).all()
        return 0.0

    metrics.bootstrap_replicates(clusters_per_class, y_true, clusters, n_resamples=20)


def test_perfect_predictions_have_a_zero_width_interval():
    y_true, _, clusters = make_clustered()
    result = metrics.macro_f1_with_ci(y_true, y_true, clusters, [0, 1, 2], n_resamples=50)
    assert result == {"macro_f1": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_interval_contains_the_estimate_and_is_reproducible():
    y_true, y_pred, clusters = make_clustered()
    first = metrics.macro_f1_with_ci(y_true, y_pred, clusters, [0, 1, 2], n_resamples=200, seed=1)
    again = metrics.macro_f1_with_ci(y_true, y_pred, clusters, [0, 1, 2], n_resamples=200, seed=1)
    assert first == again
    assert first["ci_low"] < first["macro_f1"] < first["ci_high"]


def test_paired_difference_of_a_model_with_itself_is_zero():
    y_true, y_pred, clusters = make_clustered()
    result = metrics.paired_difference_ci(
        y_true, y_pred, y_pred, clusters, [0, 1, 2], n_resamples=50
    )
    assert result == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_paired_difference_favours_the_better_model():
    y_true, y_pred, clusters = make_clustered()
    result = metrics.paired_difference_ci(
        y_true, y_true, y_pred, clusters, [0, 1, 2], n_resamples=200
    )
    assert result["difference"] > 0
    assert result["ci_low"] > 0
