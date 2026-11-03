"""
experiments/baseline_similarity/run_similarity.py

Nearest-neighbour similarity baseline, compared with the linear probe (D5, D14,
experiment E2).

What this script does:
    1. Predicts each test sequence's sub-class by copying the label of its
       best training hit (highest bit score, from src/data/identity.py)
    2. Loads the linear probe's saved test predictions (E1, mean pooling)
    3. Scores both models on the same sequences, in several views:
         all            the full test set
         relative_free  no training hit at or above 30% identity covering at
                        least 80% of both sequences
         no_hit, <30, 30-40, ...  bins of best-hit identity
    4. Reports macro-F1 and accuracy per model, and the probe-minus-neighbour
       difference with a paired cluster-bootstrap interval

Why copy one label and tune nothing?
    Annotation transfer from the closest known protein is what a biologist
    would do with BLAST. With no parameters to tune, it needs no validation
    data and cannot be overfitted.

Why does no hit count as wrong?
    Those sequences are the hardest cases. Dropping them would flatter the
    baseline, so they stay in every view as a wrong prediction (label -1).

Reading across bins:
    The bins hold different sets of classes, so macro-F1 is not strictly
    comparable from one bin to the next. The difference between the two models
    within a bin is the comparison that matters.
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.identity import NO_PREDICTION, build_views, predict_from_best_hit  # noqa: F401
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

def score_view(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Both models on one view. Macro-F1 covers the classes present in this view,
    so a class with no sequences here does not count as F1 = 0.
    """
    labels = sorted(set(y_true.tolist()))
    view = {
        "n_sequences": int(len(y_true)),
        "n_clusters": int(len(set(clusters.tolist()))),
        "n_classes": len(labels),
        "models": {},
    }
    for name, y_pred in predictions.items():
        scores = metrics.macro_f1_with_ci(y_true, y_pred, clusters, labels, n_bootstrap, seed)
        scores["accuracy"] = float((y_pred == y_true).mean())
        view["models"][name] = scores
    view["probe_minus_neighbour"] = {
        "macro_f1": metrics.paired_difference_ci(
            y_true, predictions["probe"], predictions["neighbour"],
            clusters, labels, n_bootstrap, seed,
        ),
        "accuracy": metrics.paired_accuracy_difference_ci(
            y_true, predictions["probe"], predictions["neighbour"],
            clusters, n_bootstrap, seed,
        ),
    }
    return view


def run(cfg: DictConfig) -> None:
    similarity = cfg.similarity
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(similarity.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "train.json") as f:
        train_labels = {r["uniprot_id"]: r["label"] for r in json.load(f)}
    with open(processed_dir / "test.json") as f:
        test = json.load(f)
    with open(processed_dir / "label_map.json") as f:
        index_to_subclass = {v: k for k, v in json.load(f).items()}
    with open(Path(cfg.data.identity.work_dir) / "test_identity.json") as f:
        table = json.load(f)

    probe_path = Path(cfg.probe.output_dir) / f"test_predictions_{cfg.probe.poolings[0]}.npz"
    with np.load(probe_path) as f:
        probe_ids, probe_pred = f["uniprot_ids"], f["predictions"]

    # All three sources must describe the same sequences in the same order
    ids = [r["uniprot_id"] for r in test]
    if ids != [row["uniprot_id"] for row in table] or ids != probe_ids.tolist():
        raise ValueError(
            "Test records, identity table and probe predictions are not aligned. "
            "Rerun src/data/identity.py and run_probe.py on the current splits."
        )

    y_true = np.array([r["label"] for r in test])
    clusters = np.array([r["cluster_id"] for r in test])
    neighbour_pred = predict_from_best_hit(table, train_labels)
    predictions = {"neighbour": neighbour_pred, "probe": probe_pred}

    views = build_views(table, similarity.relative_identity, list(similarity.identity_edges))

    commit, dirty = git_state()
    results = {
        "experiment": "E2 nearest-neighbour similarity vs linear probe",
        "git_commit": commit,
        "dirty": dirty,
        "config": config_as_dict(cfg.similarity),
        "views": {},
    }
    for name, mask in views.items():
        logger.info(f"Scoring view {name}: {mask.sum():,} sequences")
        if not mask.any():
            results["views"][name] = {"n_sequences": 0}
            continue
        results["views"][name] = score_view(
            y_true[mask], {m: p[mask] for m, p in predictions.items()}, clusters[mask],
            similarity.n_bootstrap, similarity.seed,
        )

    results["neighbour_hierarchical"] = metrics.hierarchical_metrics(
        y_true, neighbour_pred, index_to_subclass
    )
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    table_path = output_dir / "per_class_neighbour.json"
    with open(table_path, "w") as f:
        per_class = metrics.per_class_table(y_true, neighbour_pred, clusters, index_to_subclass)
        json.dump(per_class, f, indent=2)
    np.savez(
        output_dir / "test_predictions_neighbour.npz",
        uniprot_ids=np.array(ids), labels=y_true, clusters=clusters, predictions=neighbour_pred,
    )
    logger.info(f"Results saved to {results_path}")

    log_run(
        cfg, "decodec-baselines", "similarity-nearest-neighbour",
        metrics={
            f"{view}/{model}_{key}": results["views"][view]["models"][model][key]
            for view in ["all", "relative_free"]
            for model in predictions
            for key in ["macro_f1", "accuracy"]
        },
        artifacts=[results_path, table_path],
    )
    logger.info("Similarity baseline complete.")


if __name__ == "__main__":
    run(cli_config("Nearest-neighbour similarity baseline vs the linear probe (E2)"))
