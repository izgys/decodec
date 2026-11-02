"""
experiments/model_comparison/run_comparison.py

Fine-tuned model against both baselines on the same test sequences (D14,
experiment E3). It trains nothing and only reads saved predictions.

What this script does:
    1. Loads the saved test predictions of ESM-2 + LoRA, the linear probe (E1,
       mean pooling) and the nearest neighbour (E2)
    2. Scores all three in the views of E2: the full test set, the sequences
       with no training relative, and the bins of best-hit identity
    3. Reports macro-F1 and accuracy per model, and LoRA minus each baseline
       with a paired cluster-bootstrap interval
    4. Saves results.json and a per-class table, and logs the run to MLflow

Why paired differences?
    All models are scored on the same resampled clusters, so the interval
    covers the gap between them. Two separate intervals that overlap would say
    little about that gap.

Reading across bins:
    Six bins mean six comparisons, with no correction for multiplicity. An
    interval that barely excludes zero in one bin is weak evidence.
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.identity import build_views
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

BASELINES = ["probe", "neighbour"]


def load_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the experiment that produces it first.")
    with np.load(path) as f:
        return f["uniprot_ids"].tolist(), f["predictions"]


def score_view(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """All models on one view. Macro-F1 covers the classes present in this view."""
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
    for baseline in BASELINES:
        view[f"lora_minus_{baseline}"] = {
            "macro_f1": metrics.paired_difference_ci(
                y_true, predictions["lora"], predictions[baseline],
                clusters, labels, n_bootstrap, seed,
            ),
            "accuracy": metrics.paired_accuracy_difference_ci(
                y_true, predictions["lora"], predictions[baseline],
                clusters, n_bootstrap, seed,
            ),
        }
    return view


def per_class_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    index_to_subclass: dict[int, str],
) -> list[dict]:
    """One row per class: support, clusters and each model's F1."""
    tables = {
        name: metrics.per_class_table(y_true, y_pred, clusters, index_to_subclass)
        for name, y_pred in predictions.items()
    }
    rows = []
    for i, row in enumerate(tables["lora"]):
        rows.append({
            "subclass": row["subclass"],
            "support": row["support"],
            "n_clusters": row["n_clusters"],
            **{f"{name}_f1": table[i]["f1"] for name, table in tables.items()},
        })
    return rows


def run(cfg: DictConfig) -> None:
    comparison = cfg.comparison
    output_dir = Path(comparison.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg.data.output.processed_dir) / "test.json") as f:
        test = json.load(f)
    with open(Path(cfg.data.output.processed_dir) / "label_map.json") as f:
        index_to_subclass = {v: k for k, v in json.load(f).items()}
    with open(Path(cfg.data.identity.work_dir) / "test_identity.json") as f:
        table = json.load(f)

    sources = {
        "lora": Path(comparison.lora_predictions),
        "probe": Path(cfg.probe.output_dir) / f"test_predictions_{cfg.probe.poolings[0]}.npz",
        "neighbour": Path(cfg.similarity.output_dir) / "test_predictions_neighbour.npz",
    }
    ids = [r["uniprot_id"] for r in test]
    if ids != [row["uniprot_id"] for row in table]:
        raise ValueError("Test records and identity table are not aligned.")
    predictions = {}
    for name, path in sources.items():
        model_ids, predictions[name] = load_predictions(path)
        if model_ids != ids:
            raise ValueError(f"Test records and {name} predictions are not aligned.")

    y_true = np.array([r["label"] for r in test])
    clusters = np.array([r["cluster_id"] for r in test])
    views = build_views(
        table, cfg.similarity.relative_identity, list(cfg.similarity.identity_edges)
    )

    commit, dirty = git_state()
    results = {
        "experiment": "E3 ESM-2 + LoRA vs linear probe and nearest neighbour",
        "git_commit": commit,
        "dirty": dirty,
        "sources": {name: path.as_posix() for name, path in sources.items()},
        "config": config_as_dict(cfg.comparison),
        "views": {},
    }
    for name, mask in views.items():
        logger.info(f"Scoring view {name}: {mask.sum():,} sequences")
        if not mask.any():
            results["views"][name] = {"n_sequences": 0}
            continue
        results["views"][name] = score_view(
            y_true[mask], {m: p[mask] for m, p in predictions.items()}, clusters[mask],
            comparison.n_bootstrap, comparison.seed,
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    table_path = output_dir / "per_class.json"
    with open(table_path, "w") as f:
        rows = per_class_comparison(y_true, predictions, clusters, index_to_subclass)
        json.dump(rows, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    log_run(
        cfg, "decodec-lora", "model-comparison",
        metrics={
            f"{view}/{model}_{key}": results["views"][view]["models"][model][key]
            for view in ["all", "relative_free"]
            for model in predictions
            for key in ["macro_f1", "accuracy"]
        },
        artifacts=[results_path, table_path],
    )
    logger.info("Model comparison complete.")


if __name__ == "__main__":
    run(cli_config("ESM-2 + LoRA vs linear probe and nearest neighbour (E3)"))
