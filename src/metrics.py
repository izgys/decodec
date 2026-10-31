"""
src/metrics.py

Evaluation metrics shared by every model (linear probe, nearest neighbour,
LoRA), so all of them are scored by the same code.

What this module does:
    - Macro-F1 over a fixed label set, per-class F1 with support and cluster
      counts, top-k accuracy
    - Hierarchical metrics (D15): top-level EC accuracy derived from
      sub-class predictions, and the share of errors that stay inside the
      correct top-level class
    - Stratified cluster bootstrap (D3): confidence intervals that treat the
      cluster, not the sequence, as the unit of independence; paired across
      models, so a difference between two models gets its own interval

Why a cluster bootstrap?
    Sequences in the same cluster are near-copies of each other. Resampling
    sequences would treat 700 sequences from 6 clusters as 700 independent
    examples and give an interval that is far too narrow.

Why stratified?
    Clusters are resampled within each class, so every class keeps its number
    of clusters in every replicate and no class's F1 becomes undefined.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    """Macro-F1 over a fixed label set; a class never predicted scores 0."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of sequences whose true label is among the k highest scores."""
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return float((top_k == y_true[:, None]).any(axis=1).mean())


def per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: Sequence[str],
    index_to_subclass: dict[int, str],
) -> list[dict]:
    """One row per class: precision, recall, F1, support and number of clusters."""
    labels = sorted(index_to_subclass)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cluster_sets = defaultdict(set)
    for label, cluster in zip(y_true, clusters):
        cluster_sets[int(label)].add(cluster)
    return [
        {
            "subclass": index_to_subclass[label],
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
            "n_clusters": len(cluster_sets[label]),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]


def top_level(subclass: str) -> str:
    return subclass.split(".")[0]


def hierarchical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, index_to_subclass: dict[int, str]
) -> dict[str, float]:
    """
    top_level_accuracy: the predicted sub-class has the right first EC digit.
    errors_within_top_level: of the wrong sub-class predictions, the share
    that still has the right first digit (a "near miss").
    """
    true_top = np.array([top_level(index_to_subclass[int(i)]) for i in y_true])
    pred_top = np.array([top_level(index_to_subclass[int(i)]) for i in y_pred])
    wrong = y_true != y_pred
    return {
        "top_level_accuracy": float((true_top == pred_top).mean()),
        "errors_within_top_level": (
            float((true_top[wrong] == pred_top[wrong]).mean()) if wrong.any() else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Stratified, paired cluster bootstrap
# ---------------------------------------------------------------------------

def cluster_strata(y_true: np.ndarray, clusters: Sequence[str]) -> list[list[np.ndarray]]:
    """
    Group test indices by cluster, and clusters by their majority label.
    Returns one list per class, each holding one index array per cluster.
    """
    members = defaultdict(list)
    for i, cluster in enumerate(clusters):
        members[cluster].append(i)
    by_label = defaultdict(list)
    for cluster in sorted(members):
        idx = np.array(members[cluster])
        label = Counter(y_true[idx].tolist()).most_common(1)[0][0]
        by_label[label].append(idx)
    return [by_label[label] for label in sorted(by_label)]


def bootstrap_replicates(
    metric: Callable[[np.ndarray], float],
    y_true: np.ndarray,
    clusters: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute metric(indices) on n_resamples stratified cluster resamples.
    metric receives the resampled test indices (with repeats). Passing a
    metric that scores several models on the same indices gives a paired
    bootstrap.
    """
    rng = np.random.default_rng(seed)
    strata = cluster_strata(y_true, clusters)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        picked = []
        for stratum in strata:
            choice = rng.integers(0, len(stratum), size=len(stratum))
            picked.extend(stratum[c] for c in choice)
        values[b] = metric(np.concatenate(picked))
    return values


def confidence_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def macro_f1_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: Sequence[str],
    labels: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Macro-F1 and its 95% stratified cluster-bootstrap interval."""
    replicates = bootstrap_replicates(
        lambda idx: macro_f1(y_true[idx], y_pred[idx], labels),
        y_true, clusters, n_resamples, seed,
    )
    low, high = confidence_interval(replicates)
    return {"macro_f1": macro_f1(y_true, y_pred, labels), "ci_low": low, "ci_high": high}


def paired_difference_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    clusters: Sequence[str],
    labels: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Macro-F1 of model A minus model B, with a paired 95% interval."""
    def diff(idx):
        a = macro_f1(y_true[idx], pred_a[idx], labels)
        return a - macro_f1(y_true[idx], pred_b[idx], labels)

    replicates = bootstrap_replicates(diff, y_true, clusters, n_resamples, seed)
    low, high = confidence_interval(replicates)
    point = macro_f1(y_true, pred_a, labels) - macro_f1(y_true, pred_b, labels)
    return {"difference": point, "ci_low": low, "ci_high": high}
