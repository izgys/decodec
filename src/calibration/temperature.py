"""
src/calibration/temperature.py

Temperature scaling for the fine-tuned model (D16). It only reads saved logits.

What this script does:
    1. Fits one number, the temperature T, on validation: logits are divided by
       T before the softmax, with T chosen to minimise the negative log-likelihood
    2. Measures calibration before and after on test: expected calibration error
       (ECE) with a cluster-bootstrap interval, log-likelihood, and how many
       predictions clear each confidence threshold and how accurate they are
    3. Repeats this per identity view, because a global T can hide a model that
       is overconfident on distant sequences and underconfident on close ones
    4. Checks large clusters one by one, because ECE averages over sequences and
       a whole family can fail while the average looks good
    5. Saves results.json (with T, for inference) and logs the run to MLflow

Why is using test allowed here, after the model comparison?
    Dividing logits by a positive number keeps the highest score the highest, so
    no prediction, accuracy or macro-F1 changes. T comes from validation only.

ECE in one line:
    Bin predictions by confidence; ECE is the average gap between confidence and
    accuracy in a bin, weighted by the share of predictions in that bin.

Why not weight the classes when fitting T?
    The training loss weights classes, which shifts the model towards rare
    classes. T only rescales confidence, and the question it answers is the one a
    user asks of a single prediction: how often is this confidence right?
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.identity import build_views
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

LOG_T_BOUNDS = (-3.0, 3.0)      # T between about 0.05 and 20


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    log_probs = log_softmax(logits / temperature, axis=1)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """The temperature that minimises the negative log-likelihood. Searched in log T."""
    logits = logits.astype(np.float64)
    result = minimize_scalar(
        lambda log_t: negative_log_likelihood(logits, labels, float(np.exp(log_t))),
        bounds=LOG_T_BOUNDS, method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature search failed: {result.message}")
    return float(np.exp(result.x))


def confidence_and_correct(
    logits: np.ndarray, labels: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """The probability of the predicted class, and whether that prediction is right."""
    probs = softmax(logits.astype(np.float64) / temperature, axis=1)
    return probs.max(axis=1), probs.argmax(axis=1) == labels


def bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> list[dict]:
    """One row per bin (lower edge excluded, upper edge included)."""
    edges = bin_edges(n_bins)
    index = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        members = index == b
        n = int(members.sum())
        rows.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "n_sequences": n,
            "mean_confidence": float(confidence[members].mean()) if n else None,
            "accuracy": float(correct[members].mean()) if n else None,
        })
    return rows


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int
) -> float:
    total = len(confidence)
    return sum(
        row["n_sequences"] / total * abs(row["mean_confidence"] - row["accuracy"])
        for row in reliability_bins(confidence, correct, n_bins)
        if row["n_sequences"]
    )


def score_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    temperature: float,
    cfg: DictConfig,
) -> dict:
    """Calibration at one temperature. All intervals resample clusters, not sequences."""
    logits = logits.astype(np.float64)
    confidence, correct = confidence_and_correct(logits, labels, temperature)
    replicates = metrics.bootstrap_replicates(
        lambda idx: expected_calibration_error(confidence[idx], correct[idx], cfg.n_bins),
        labels, clusters, cfg.n_bootstrap, cfg.seed,
    )
    low, high = metrics.confidence_interval(replicates)
    thresholds = {}
    for threshold in cfg.confidence_thresholds:
        confident = confidence >= threshold
        thresholds[f"{threshold:g}"] = {
            "share_of_sequences": float(confident.mean()),
            "accuracy": float(correct[confident].mean()) if confident.any() else None,
        }
    return {
        "ece": expected_calibration_error(confidence, correct, cfg.n_bins),
        "ci_low": low,
        "ci_high": high,
        "nll": negative_log_likelihood(logits, labels, temperature),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "thresholds": thresholds,
        "bins": reliability_bins(confidence, correct, cfg.n_bins),
    }


def ece_change(
    logits: np.ndarray, labels: np.ndarray, clusters: np.ndarray, temperature: float,
    cfg: DictConfig,
) -> dict:
    """ECE after minus before, with a paired interval: negative means scaling helped."""
    before = confidence_and_correct(logits, labels, 1.0)
    after = confidence_and_correct(logits, labels, temperature)

    def change(idx):
        scaled = expected_calibration_error(after[0][idx], after[1][idx], cfg.n_bins)
        return scaled - expected_calibration_error(before[0][idx], before[1][idx], cfg.n_bins)

    replicates = metrics.bootstrap_replicates(
        change, labels, clusters, cfg.n_bootstrap, cfg.seed
    )
    low, high = metrics.confidence_interval(replicates)
    point = change(np.arange(len(labels)))
    return {"difference": point, "ci_low": low, "ci_high": high}


def score_view(
    logits: np.ndarray, labels: np.ndarray, clusters: np.ndarray, temperature: float,
    cfg: DictConfig,
) -> dict:
    return {
        "n_sequences": int(len(labels)),
        "n_clusters": int(len(set(clusters.tolist()))),
        "n_classes": int(len(set(labels.tolist()))),
        "before": score_calibration(logits, labels, clusters, 1.0, cfg),
        "after": score_calibration(logits, labels, clusters, temperature, cfg),
        "ece_change": ece_change(logits, labels, clusters, temperature, cfg),
    }


def cluster_check(
    logits: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    records: list[dict],
    temperature: float,
    min_size: int,
) -> dict:
    """Confidence against accuracy for each large cluster, worst overconfidence first."""
    confidence, correct = confidence_and_correct(logits, labels, temperature)
    rows = []
    for cluster in sorted(set(clusters.tolist())):
        members = np.flatnonzero(clusters == cluster)
        if len(members) < min_size:
            continue
        rows.append({
            "cluster": cluster,
            "n_sequences": int(len(members)),
            "subclass": records[members[0]]["ec_subclass"],
            "example_name": records[members[0]]["protein_name"],
            "mean_confidence": float(confidence[members].mean()),
            "accuracy": float(correct[members].mean()),
        })
    rows.sort(key=lambda r: r["accuracy"] - r["mean_confidence"])
    gaps = np.array([abs(r["mean_confidence"] - r["accuracy"]) for r in rows])
    return {
        "min_cluster_size": min_size,
        "n_clusters": len(rows),
        "n_sequences": sum(r["n_sequences"] for r in rows),
        "mean_absolute_gap": float(gaps.mean()) if len(rows) else None,
        "n_mostly_wrong_and_confident": sum(
            r["accuracy"] < 0.5 and r["mean_confidence"] >= 0.8 for r in rows
        ),
        "clusters": rows,
    }


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/training/trainer.py to the end first.")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def run(cfg: DictConfig) -> None:
    calibration = cfg.calibration
    output_dir = Path(calibration.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = Path(calibration.predictions_dir)
    val = load_predictions(predictions_dir / "val_predictions.npz")
    test = load_predictions(predictions_dir / "test_predictions.npz")
    with open(Path(cfg.data.output.processed_dir) / "test.json") as f:
        records = json.load(f)
    with open(Path(cfg.data.identity.work_dir) / "test_identity.json") as f:
        table = json.load(f)
    ids = [r["uniprot_id"] for r in records]
    if ids != test["uniprot_ids"].tolist() or ids != [row["uniprot_id"] for row in table]:
        raise ValueError("Test records, identity table and predictions are not aligned.")

    temperature = fit_temperature(val["logits"], val["labels"])
    logger.info(f"Temperature fitted on validation: {temperature:.3f}")

    views = build_views(
        table, cfg.similarity.relative_identity, list(cfg.similarity.identity_edges)
    )
    commit, dirty = git_state()
    results = {
        "experiment": "Temperature scaling of ESM-2 + LoRA",
        "git_commit": commit,
        "dirty": dirty,
        "temperature": temperature,
        "config": config_as_dict(calibration),
        "validation": score_view(
            val["logits"], val["labels"], val["clusters"], temperature, calibration
        ),
        "views": {},
    }
    for name, mask in views.items():
        logger.info(f"Scoring view {name}: {mask.sum():,} sequences")
        if not mask.any():
            results["views"][name] = {"n_sequences": 0}
            continue
        results["views"][name] = score_view(
            test["logits"][mask], test["labels"][mask], test["clusters"][mask],
            temperature, calibration,
        )
    results["cluster_check"] = {
        stage: cluster_check(
            test["logits"], test["labels"], test["clusters"], records, t,
            calibration.min_cluster_size,
        )
        for stage, t in [("before", 1.0), ("after", temperature)]
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    summary = results["views"]["all"]
    logger.info(
        f"Test ECE {summary['before']['ece']:.4f} before, {summary['after']['ece']:.4f} after. "
        f"Results saved to {results_path}"
    )

    log_run(
        cfg, "decodec-lora", "calibration",
        metrics={
            "temperature": temperature,
            **{
                f"{view}/{stage}_{key}": results["views"][view][stage][key]
                for view in ["all", "relative_free"]
                for stage in ["before", "after"]
                for key in ["ece", "nll", "mean_confidence"]
            },
        },
        artifacts=[results_path],
    )


if __name__ == "__main__":
    run(cli_config("Fit a temperature on validation and measure calibration on test (D16)"))
