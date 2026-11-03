"""
experiments/random_split/run_random_split.py

How much does a random split inflate scores? (D6, D26, experiment E5)

What this script does:
    1. Splits the same records at random, stratified by class, with the same
       split sizes as the homology-aware split
    2. Refits the linear probe on the existing embeddings, with the same
       procedure as E1 (standardised features, C chosen on validation)
    3. Searches random test against random train with MMseqs2 (Docker) and
       scores the nearest neighbour, with the same settings as E2
    4. Scores both models in the same views on both splits, and reports the gap
       (random minus homology-aware) for the full test set and for sequences
       with no training relative
    5. Saves results.json and the identity report, and logs the run to MLflow

Why reuse the embeddings?
    They come from frozen ESM-2, so they do not depend on the split. Only what
    is fitted on top of them does.

Why independent resamples for the gap?
    The two test sets hold different sequences, so a paired bootstrap is not
    possible. Each is resampled on its own, which gives a wider, honest interval.

Why still resample clusters on a random split?
    Sequences in a cluster stay related wherever they land, so the cluster is
    still closer to the unit of independence than the sequence is.

Interrupted or rerun?
    The split and search are kept per seed in data/random_split/seed_<n>/, so
    only the scoring repeats. Delete that folder to search again.
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.clustering import write_fasta
from src.data.identity import (
    build_report,
    build_table,
    build_views,
    predict_from_best_hit,
    run_search,
    summarise_hits,
)
from src.embeddings import SPLITS, embeddings_path
from src.probe import fit_and_select
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

GAP_VIEWS = ["all", "relative_free"]    # the views compared between the two splits


def make_random_split(
    records: list[dict], n_val: int, n_test: int, seed: int
) -> dict[str, list[dict]]:
    """Random train, val and test of the given sizes, each with the same class mix."""
    labels = [r["label"] for r in records]
    rest, test = train_test_split(records, test_size=n_test, stratify=labels, random_state=seed)
    train, val = train_test_split(
        rest, test_size=n_val, stratify=[r["label"] for r in rest], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def load_mean_embeddings(cfg: DictConfig) -> tuple[dict[str, int], np.ndarray]:
    """The mean-pooled embeddings of all splits, with each protein's row number."""
    ids, arrays = [], []
    for split in SPLITS:
        path = embeddings_path(cfg, split)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: uv run python -m src.embeddings")
        with np.load(path) as f:
            ids.extend(f["uniprot_ids"].tolist())
            arrays.append(f["mean"].astype(np.float32))
    return {uid: i for i, uid in enumerate(ids)}, np.concatenate(arrays)


def score_with_replicates(
    y_true: np.ndarray, y_pred: np.ndarray, clusters: np.ndarray, n_bootstrap: int, seed: int
) -> tuple[dict, np.ndarray]:
    """Macro-F1 over the classes in this view, its interval and its bootstrap replicates."""
    labels = sorted(set(y_true.tolist()))
    replicates = metrics.bootstrap_replicates(
        lambda idx: metrics.macro_f1(y_true[idx], y_pred[idx], labels),
        y_true, clusters, n_bootstrap, seed,
    )
    low, high = metrics.confidence_interval(replicates)
    return {
        "macro_f1": metrics.macro_f1(y_true, y_pred, labels),
        "ci_low": low,
        "ci_high": high,
        "accuracy": float((y_pred == y_true).mean()),
    }, replicates


def score_split(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray,
    views: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> tuple[dict, dict]:
    """Every model in every view. Also returns the replicates kept for the gap."""
    results, replicates = {}, {}
    for name, mask in views.items():
        if not mask.any():
            results[name] = {"n_sequences": 0}
            continue
        view = {
            "n_sequences": int(mask.sum()),
            "n_clusters": int(len(set(clusters[mask].tolist()))),
            "n_classes": int(len(set(y_true[mask].tolist()))),
            "models": {},
        }
        for model, y_pred in predictions.items():
            view["models"][model], values = score_with_replicates(
                y_true[mask], y_pred[mask], clusters[mask], n_bootstrap, seed
            )
            if name in GAP_VIEWS:
                replicates[name, model] = values
        results[name] = view
    return results, replicates


def gap(random_view: dict, homology_view: dict, random_reps, homology_reps) -> dict:
    low, high = metrics.confidence_interval(random_reps - homology_reps)
    return {
        "difference": random_view["macro_f1"] - homology_view["macro_f1"],
        "ci_low": low,
        "ci_high": high,
    }


def searched_table(cfg: DictConfig, test: list[dict], train: list[dict], work_dir: Path) -> list:
    """The identity table for this split, reusing a finished search."""
    table_path = work_dir / "test_identity.json"
    if table_path.exists():
        table = json.loads(table_path.read_text())
        if [row["uniprot_id"] for row in table] == [r["uniprot_id"] for r in test]:
            logger.info(f"Reusing the search in {table_path}")
            return table
    write_fasta(test, work_dir / "test.fasta")
    write_fasta(train, work_dir / "train.fasta")
    identity = cfg.data.identity
    tsv_path = run_search(
        work_dir / "test.fasta", work_dir / "train.fasta", work_dir,
        identity.sensitivity, identity.max_evalue, identity.max_seqs,
    )
    table = build_table(test, summarise_hits(tsv_path))
    table_path.write_text(json.dumps(table))
    return table


def homology_predictions(cfg: DictConfig, ids: list[str]) -> dict[str, np.ndarray]:
    """The saved test predictions of E1 and E2, on the homology-aware split."""
    sources = {
        "probe": Path(cfg.probe.output_dir) / "test_predictions_mean.npz",
        "neighbour": Path(cfg.similarity.output_dir) / "test_predictions_neighbour.npz",
    }
    predictions = {}
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run E1 and E2 first.")
        with np.load(path) as f:
            if f["uniprot_ids"].tolist() != ids:
                raise ValueError(f"Test records and {name} predictions are not aligned.")
            predictions[name] = f["predictions"]
    return predictions


def run(cfg: DictConfig) -> None:
    random_split = cfg.random_split
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(random_split.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(random_split.work_dir) / f"seed_{random_split.seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    homology = {}
    for split in SPLITS:
        with open(processed_dir / f"{split}.json") as f:
            homology[split] = json.load(f)
    with open(processed_dir / "label_map.json") as f:
        labels = sorted(json.load(f).values())
    with open(Path(cfg.data.identity.work_dir) / "test_identity.json") as f:
        homology_table = json.load(f)

    records = [r for split in SPLITS for r in homology[split]]
    split = make_random_split(
        records, len(homology["val"]), len(homology["test"]), random_split.seed
    )
    test, train = split["test"], split["train"]
    logger.info(
        f"Random split: {len(train):,} train, {len(split['val']):,} val, {len(test):,} test"
    )

    # Linear probe, as in E1
    rows, embeddings = load_mean_embeddings(cfg)
    x = {
        name: embeddings[[rows[r["uniprot_id"]] for r in members]]
        for name, members in split.items()
    }
    del embeddings
    y = {name: np.array([r["label"] for r in members]) for name, members in split.items()}
    scaler = StandardScaler().fit(x["train"])
    clf, sweep = fit_and_select(
        scaler.transform(x["train"]), y["train"], scaler.transform(x["val"]), y["val"],
        labels, cfg.probe,
    )
    probe_pred = clf.predict(scaler.transform(x["test"]))
    del x

    # Nearest neighbour, as in E2
    table = searched_table(cfg, test, train, work_dir)
    train_labels = {r["uniprot_id"]: r["label"] for r in train}
    neighbour_pred = predict_from_best_hit(table, train_labels)

    similarity = cfg.similarity
    edges = list(similarity.identity_edges)
    n_bootstrap, seed = random_split.n_bootstrap, random_split.seed
    random_results, random_reps = score_split(
        y["test"], {"probe": probe_pred, "neighbour": neighbour_pred},
        np.array([r["cluster_id"] for r in test]),
        build_views(table, similarity.relative_identity, edges), n_bootstrap, seed,
    )
    # A different seed: the two test sets are resampled independently
    homology_results, homology_reps = score_split(
        np.array([r["label"] for r in homology["test"]]),
        homology_predictions(cfg, [r["uniprot_id"] for r in homology["test"]]),
        np.array([r["cluster_id"] for r in homology["test"]]),
        build_views(homology_table, similarity.relative_identity, edges), n_bootstrap, seed + 1,
    )
    gaps = {
        view: {
            model: gap(
                random_results[view]["models"][model], homology_results[view]["models"][model],
                random_reps[view, model], homology_reps[view, model],
            )
            for model in ["probe", "neighbour"]
        }
        for view in GAP_VIEWS
        if random_results[view]["n_sequences"] and homology_results[view]["n_sequences"]
    }

    report = build_report(table, test)
    report_path = output_dir / "identity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    commit, dirty = git_state()
    results = {
        "experiment": "E5 random vs homology-aware split, linear probe and nearest neighbour",
        "git_commit": commit,
        "dirty": dirty,
        "n_train": len(train),
        "n_val": len(split["val"]),
        "n_test": len(test),
        "n_classes": len(labels),
        "best_C": float(clf.C),
        "C_sweep": sweep,
        "config": config_as_dict(random_split),
        "random": random_results,
        "homology": homology_results,
        "gap_random_minus_homology": gaps,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        output_dir / "test_predictions.npz",
        uniprot_ids=np.array([r["uniprot_id"] for r in test]), labels=y["test"],
        probe=probe_pred, neighbour=neighbour_pred,
    )
    for view, models in gaps.items():
        for model, value in models.items():
            logger.info(
                f"{view}, {model}: gap {value['difference']:+.3f} "
                f"[{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]"
            )
    logger.info(f"Results saved to {results_path}")

    log_run(
        cfg, "decodec-baselines", "random-split",
        metrics={
            f"{view}/{model}_{name}": value[name]
            for view, models in gaps.items()
            for model, value in models.items()
            for name in ["difference", "ci_low", "ci_high"]
        },
        artifacts=[results_path, report_path],
    )


if __name__ == "__main__":
    run(cli_config("Random vs homology-aware split, for the linear probe and nearest neighbour"))
