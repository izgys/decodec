"""
experiments/baseline_linear_probe/run_probe.py

Corrected linear-probe baseline on frozen ESM-2 embeddings (D2, experiment E1).

What this script does:
    1. Loads the embeddings saved by src/embeddings.py (run that first)
    2. For each pooling (mean = primary result, cls = comparison):
         - standardises features (scaler fitted on train only)
         - fits a class-weighted logistic regression for every C in the grid
         - picks C by validation macro-F1
         - evaluates the chosen model once on test
    3. Reports macro-F1 with a stratified cluster-bootstrap interval (D3),
       micro-F1, top-3 accuracy, hierarchical metrics (D15) and per-class F1
       with support and cluster counts
    4. Saves results.json, per-class reports and test predictions, and logs
       the run to MLflow

Why a linear probe?
    A linear model measures what is already in the representations, not what
    a network can learn on top of them. It is the reference for how much
    fine-tuning adds.

Why tune C on validation and refit nothing on validation?
    Validation is also used later to fit the calibration temperature (D16).
    The probe is trained on train only, so val stays unseen by the weights.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src import metrics
from src.config import cli_config, config_as_dict
from src.embeddings import SPLITS, embeddings_path
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_split(cfg: DictConfig, split: str) -> dict[str, np.ndarray]:
    path = embeddings_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python -m src.embeddings")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def fit_and_select(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    labels: list[int], probe: DictConfig,
) -> tuple[LogisticRegression, list[dict]]:
    """Fit one probe per C on train; return the best by val macro-F1, and the sweep."""
    sweep, best, best_f1 = [], None, -1.0
    for c in probe.C_grid:
        start = time.time()
        clf = LogisticRegression(
            C=float(c),
            class_weight=probe.class_weight,
            max_iter=probe.max_iter,
            random_state=probe.seed,
        )
        clf.fit(x_train, y_train)
        val_f1 = metrics.macro_f1(y_val, clf.predict(x_val), labels)
        sweep.append({
            "C": float(c),
            "val_macro_f1": val_f1,
            "n_iter": int(np.max(clf.n_iter_)),
            "converged": bool(np.max(clf.n_iter_) < probe.max_iter),
            "fit_seconds": time.time() - start,
        })
        logger.info(f"  C={c:<8} val macro-F1 {val_f1:.4f} ({sweep[-1]['fit_seconds']:.0f} s)")
        if val_f1 > best_f1:
            best, best_f1 = clf, val_f1
    return best, sweep


def evaluate(
    clf: LogisticRegression,
    x_test: np.ndarray, test: dict[str, np.ndarray],
    labels: list[int], index_to_subclass: dict[int, str], probe: DictConfig,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    y_true = test["labels"]
    scores = clf.predict_proba(x_test)
    # predict_proba columns follow clf.classes_; map back to label indices
    full = np.zeros((len(y_true), len(labels)), dtype=np.float32)
    full[:, clf.classes_] = scores
    y_pred = full.argmax(axis=1)

    summary = metrics.macro_f1_with_ci(
        y_true, y_pred, test["clusters"], labels, probe.n_bootstrap, probe.seed
    )
    summary.update({
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "top3_accuracy": metrics.top_k_accuracy(y_true, full, 3),
        **metrics.hierarchical_metrics(y_true, y_pred, index_to_subclass),
    })
    table = metrics.per_class_table(y_true, y_pred, test["clusters"], index_to_subclass)
    return summary, table, y_pred, full


def write_per_class_report(path: Path, table: list[dict]) -> None:
    header = ["precision", "recall", "f1", "support", "clusters"]
    lines = [f"{'subclass':<10}" + "".join(f"{h:>10}" for h in header)]
    for row in sorted(table, key=lambda r: [int(p) for p in r["subclass"].split(".")]):
        lines.append(
            f"{row['subclass']:<10}{row['precision']:>10.3f}{row['recall']:>10.3f}"
            f"{row['f1']:>10.3f}{row['support']:>10}{row['n_clusters']:>10}"
        )
    path.write_text("\n".join(lines) + "\n")


def run(cfg: DictConfig) -> None:
    probe = cfg.probe
    output_dir = Path(probe.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg.data.output.processed_dir) / "label_map.json") as f:
        label_map = json.load(f)
    index_to_subclass = {v: k for k, v in label_map.items()}
    labels = sorted(index_to_subclass)

    data = {split: load_split(cfg, split) for split in SPLITS}
    model_name = str(data["train"]["model_name"])
    for split in SPLITS:
        logger.info(f"{split}: {len(data[split]['labels']):,} embeddings")

    commit, dirty = git_state()
    results = {
        "experiment": "E1 linear probe (corrected)",
        "model": model_name,
        "git_commit": commit,
        "dirty": dirty,
        "n_classes": len(labels),
        "n_train": int(len(data["train"]["labels"])),
        "n_val": int(len(data["val"]["labels"])),
        "n_test": int(len(data["test"]["labels"])),
        "n_test_clusters": int(len(set(data["test"]["clusters"].tolist()))),
        "config": config_as_dict(cfg),
        "poolings": {},
    }

    for pooling in probe.poolings:
        logger.info(f"Pooling: {pooling}")
        scaler = StandardScaler().fit(data["train"][pooling].astype(np.float32))
        x = {s: scaler.transform(data[s][pooling].astype(np.float32)) for s in SPLITS}

        clf, sweep = fit_and_select(
            x["train"], data["train"]["labels"], x["val"], data["val"]["labels"], labels, probe
        )
        summary, table, y_pred, scores = evaluate(
            clf, x["test"], data["test"], labels, index_to_subclass, probe
        )
        summary["best_C"] = float(clf.C)
        logger.info(
            f"  test macro-F1 {summary['macro_f1']:.4f} "
            f"[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}], C={clf.C}"
        )

        results["poolings"][pooling] = {"test": summary, "C_sweep": sweep}
        write_per_class_report(output_dir / f"per_class_{pooling}.txt", table)
        with open(output_dir / f"per_class_{pooling}.json", "w") as f:
            json.dump(table, f, indent=2)
        # Kept for paired comparisons with later models (same test order and IDs)
        np.savez(
            output_dir / f"test_predictions_{pooling}.npz",
            uniprot_ids=data["test"]["uniprot_ids"],
            labels=data["test"]["labels"],
            clusters=data["test"]["clusters"],
            predictions=y_pred,
            probabilities=scores.astype(np.float16),
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    primary = probe.poolings[0]
    log_run(
        cfg, "decodec-baselines", f"linear-probe-{primary}",
        metrics={
            f"{p}/{k}": v
            for p, r in results["poolings"].items()
            for k, v in r["test"].items()
        },
        artifacts=sorted(output_dir.iterdir()),
    )


if __name__ == "__main__":
    run(cli_config("Linear probe on frozen ESM-2 embeddings (E1)"))
