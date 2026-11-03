"""
experiments/hybrid/run_hybrid.py

Search when a hit is strong, a language model otherwise (D21, experiment E7).

What this script does:
    1. Searches validation against train with MMseqs2 (Docker), as E2 did for test
    2. For each fallback (LoRA, linear probe), scores every E-value threshold on
       validation macro-F1 over all sequences, and keeps the best
    3. Picks the fallback by the same validation score
    4. With evaluate_test on, applies those choices once to test and scores the
       hybrids and the single models in the views of E2, with paired differences

Why choose on all sequences, not only those with no training relative?
    A deployed predictor receives every kind of sequence, so its one setting
    should suit them all. The no-relative view is still reported.

How far can the test score be trusted?
    The threshold and fallback come from validation, but the idea came from
    looking at this test set (E2), so the result stays somewhat optimistic. A
    clean check needs the new holdout planned for v2.

Run it twice:
    --set hybrid.evaluate_test=false   chooses on validation; test is not read
    (no flag)                          reuses the search and scores test
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.clustering import write_fasta
from src.data.identity import (
    build_table,
    build_views,
    predict_from_best_hit,
    run_search,
    summarise_hits,
)
from src.embeddings import embeddings_path
from src.probe import fit_and_select
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

FALLBACKS = ["lora", "probe"]


def predict_hybrid(
    table: list[dict], neighbour: np.ndarray, fallback: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """The hybrid's predictions, and a mask of the sequences answered by search."""
    trusted = np.array(
        [row["n_hits"] > 0 and row["best_evalue"] <= threshold for row in table], dtype=bool
    )
    return np.where(trusted, neighbour, fallback), trusted


def choose_threshold(
    y_true: np.ndarray,
    table: list[dict],
    neighbour: np.ndarray,
    fallback: np.ndarray,
    thresholds: list[float],
    labels: list[int],
) -> tuple[float, list[dict]]:
    """The threshold with the best macro-F1; on a tie, the strictest. Also the full sweep."""
    sweep = []
    for threshold in sorted(float(t) for t in thresholds):
        predicted, trusted = predict_hybrid(table, neighbour, fallback, threshold)
        sweep.append({
            "evalue_threshold": threshold,
            "macro_f1": metrics.macro_f1(y_true, predicted, labels),
            "accuracy": float((predicted == y_true).mean()),
            "share_from_search": float(trusted.mean()),
        })
    best = max(sweep, key=lambda row: row["macro_f1"])
    return best["evalue_threshold"], sweep


def validation_table(cfg: DictConfig, val: list[dict], train: list[dict], work_dir: Path) -> list:
    """The validation search, reusing a finished one."""
    table_path = work_dir / "val_identity.json"
    if table_path.exists():
        table = json.loads(table_path.read_text())
        if [row["uniprot_id"] for row in table] == [r["uniprot_id"] for r in val]:
            logger.info(f"Reusing the search in {table_path}")
            return table
    write_fasta(val, work_dir / "val.fasta")
    write_fasta(train, work_dir / "train.fasta")
    identity = cfg.data.identity
    tsv_path = run_search(
        work_dir / "val.fasta", work_dir / "train.fasta", work_dir,
        identity.sensitivity, identity.max_evalue, identity.max_seqs,
    )
    table = build_table(val, summarise_hits(tsv_path))
    table_path.write_text(json.dumps(table))
    return table


def probe_validation_predictions(
    cfg: DictConfig, val_ids: list[str], labels: list[int]
) -> tuple[np.ndarray, float]:
    """Refit the probe as in E1, and predict validation. Test embeddings are not loaded."""
    data = {}
    for split in ["train", "val"]:
        path = embeddings_path(cfg, split)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: uv run python -m src.embeddings")
        with np.load(path) as f:
            data[split] = {
                "ids": f["uniprot_ids"].tolist(),
                "labels": f["labels"],
                "mean": f["mean"].astype(np.float32),
            }
    if data["val"]["ids"] != val_ids:
        raise ValueError("Validation records and embeddings are not aligned.")
    scaler = StandardScaler().fit(data["train"]["mean"])
    x_val = scaler.transform(data["val"]["mean"])
    clf, _ = fit_and_select(
        scaler.transform(data["train"]["mean"]), data["train"]["labels"],
        x_val, data["val"]["labels"], labels, cfg.probe,
    )
    return clf.predict(x_val), float(clf.C)


def load_saved(path: Path, ids: list[str], name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the experiment that produces it first.")
    with np.load(path) as f:
        if f["uniprot_ids"].tolist() != ids:
            raise ValueError(f"Records and {name} predictions are not aligned.")
        return f["predictions"]


def score_view(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    selected: str,
    trusted: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """All models on one view, with the selected hybrid against each single model."""
    labels = sorted(set(y_true.tolist()))
    view = {
        "n_sequences": int(len(y_true)),
        "n_clusters": int(len(set(clusters.tolist()))),
        "n_classes": len(labels),
        "share_from_search": float(trusted.mean()),
        "models": {},
        "differences": {},
    }
    for name, y_pred in predictions.items():
        scores = metrics.macro_f1_with_ci(y_true, y_pred, clusters, labels, n_bootstrap, seed)
        scores["accuracy"] = float((y_pred == y_true).mean())
        view["models"][name] = scores
    for baseline in ["neighbour", "lora", "probe"]:
        view["differences"][f"hybrid_minus_{baseline}"] = {
            "macro_f1": metrics.paired_difference_ci(
                y_true, predictions[selected], predictions[baseline],
                clusters, labels, n_bootstrap, seed,
            ),
            "accuracy": metrics.paired_accuracy_difference_ci(
                y_true, predictions[selected], predictions[baseline],
                clusters, n_bootstrap, seed,
            ),
        }
    return view


def run(cfg: DictConfig) -> None:
    hybrid = cfg.hybrid
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(hybrid.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(hybrid.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "train.json") as f:
        train = json.load(f)
    with open(processed_dir / "val.json") as f:
        val = json.load(f)
    with open(processed_dir / "label_map.json") as f:
        labels = sorted(json.load(f).values())
    train_labels = {r["uniprot_id"]: r["label"] for r in train}

    # Everything in this block comes from validation
    table = validation_table(cfg, val, train, work_dir)
    val_ids = [r["uniprot_id"] for r in val]
    y_val = np.array([r["label"] for r in val])
    neighbour = predict_from_best_hit(table, train_labels)
    lora_dir = Path(hybrid.lora_predictions_dir)
    probe_pred, best_c = probe_validation_predictions(cfg, val_ids, labels)
    fallbacks = {
        "lora": load_saved(lora_dir / "val_predictions.npz", val_ids, "LoRA"),
        "probe": probe_pred,
    }
    validation = {}
    for name in FALLBACKS:
        threshold, sweep = choose_threshold(
            y_val, table, neighbour, fallbacks[name], hybrid.evalue_thresholds, labels
        )
        best = next(row for row in sweep if row["evalue_threshold"] == threshold)
        validation[name] = {
            "evalue_threshold": threshold,
            "macro_f1": best["macro_f1"],
            "accuracy": best["accuracy"],
            "share_from_search": best["share_from_search"],
            "sweep": sweep,
        }
        logger.info(
            f"Fallback {name}: E-value {threshold:g}, validation macro-F1 {best['macro_f1']:.4f}, "
            f"{best['share_from_search']:.1%} answered by search"
        )
    selected = max(FALLBACKS, key=lambda name: validation[name]["macro_f1"])
    single = {
        "neighbour": metrics.macro_f1(y_val, neighbour, labels),
        "lora": metrics.macro_f1(y_val, fallbacks["lora"], labels),
        "probe": metrics.macro_f1(y_val, fallbacks["probe"], labels),
    }
    logger.info(f"Selected fallback: {selected}. Single models on validation: {single}")

    commit, dirty = git_state()
    results = {
        "experiment": "E7 hybrid of search and language model",
        "git_commit": commit,
        "dirty": dirty,
        "selected_fallback": selected,
        "selected_evalue_threshold": validation[selected]["evalue_threshold"],
        "probe_best_C": best_c,
        "config": config_as_dict(hybrid),
        "validation": {"single_models_macro_f1": single, "hybrids": validation},
        "test": None,
    }

    if hybrid.evaluate_test:
        with open(processed_dir / "test.json") as f:
            test = json.load(f)
        with open(Path(cfg.data.identity.work_dir) / "test_identity.json") as f:
            test_table = json.load(f)
        ids = [r["uniprot_id"] for r in test]
        if ids != [row["uniprot_id"] for row in test_table]:
            raise ValueError("Test records and identity table are not aligned.")
        y_test = np.array([r["label"] for r in test])
        clusters = np.array([r["cluster_id"] for r in test])
        singles = {
            "neighbour": predict_from_best_hit(test_table, train_labels),
            "lora": load_saved(lora_dir / "test_predictions.npz", ids, "LoRA"),
            "probe": load_saved(
                Path(cfg.probe.output_dir) / "test_predictions_mean.npz", ids, "probe"
            ),
        }
        predictions, trusted_masks = dict(singles), {}
        for name in FALLBACKS:
            predictions[f"hybrid_{name}"], trusted_masks[name] = predict_hybrid(
                test_table, singles["neighbour"], singles[name],
                validation[name]["evalue_threshold"],
            )
        similarity = cfg.similarity
        views = build_views(
            test_table, similarity.relative_identity, list(similarity.identity_edges)
        )
        results["test"] = {"views": {}}
        for name, mask in views.items():
            logger.info(f"Scoring view {name}: {mask.sum():,} sequences")
            results["test"]["views"][name] = score_view(
                y_test[mask], {m: p[mask] for m, p in predictions.items()}, clusters[mask],
                f"hybrid_{selected}", trusted_masks[selected][mask],
                hybrid.n_bootstrap, hybrid.seed,
            ) if mask.any() else {"n_sequences": 0}
        np.savez(
            output_dir / "test_predictions.npz",
            uniprot_ids=np.array(ids), labels=y_test, clusters=clusters,
            predictions=predictions[f"hybrid_{selected}"], from_search=trusted_masks[selected],
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logged = {f"validation/{name}": value for name, value in single.items()}
    logged["validation/hybrid"] = validation[selected]["macro_f1"]
    if results["test"]:
        for view in ["all", "relative_free"]:
            logged[f"test/{view}_hybrid"] = (
                results["test"]["views"][view]["models"][f"hybrid_{selected}"]["macro_f1"]
            )
    log_run(cfg, "decodec-baselines", "hybrid", metrics=logged, artifacts=[results_path])


if __name__ == "__main__":
    run(cli_config("Search when a hit is strong, a language model otherwise (E7)"))
