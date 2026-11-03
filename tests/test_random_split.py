"""
Offline tests for the random split comparison, on synthetic records.

MMseqs2 (Docker) is not run: the search is replaced by a hits TSV written by hand.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_record

from src.config import load_config

REPO = Path(__file__).resolve().parent.parent
N_PER_CLASS = 40


def load_module():
    path = REPO / "experiments" / "random_split" / "run_random_split.py"
    spec = importlib.util.spec_from_file_location("run_random_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


random_split = load_module()


def make_records():
    return [
        make_record(f"p{label}-{i}", label=label, cluster_id=f"c{label}-{i // 4}")
        for label in range(2)
        for i in range(N_PER_CLASS)
    ]


def test_split_has_the_right_sizes_and_no_overlap():
    split = random_split.make_random_split(make_records(), 16, 16, seed=0)
    assert {k: len(v) for k, v in split.items()} == {"train": 48, "val": 16, "test": 16}
    ids = [r["uniprot_id"] for part in split.values() for r in part]
    assert len(ids) == len(set(ids)) == 80


def test_split_keeps_the_class_mix():
    split = random_split.make_random_split(make_records(), 16, 16, seed=0)
    for part in split.values():
        assert sum(r["label"] for r in part) == len(part) // 2


def test_split_depends_only_on_the_seed():
    first = random_split.make_random_split(make_records(), 16, 16, seed=1)
    again = random_split.make_random_split(make_records(), 16, 16, seed=1)
    other = random_split.make_random_split(make_records(), 16, 16, seed=2)
    assert first == again
    assert first["test"] != other["test"]


def test_gap_is_random_minus_homology_with_independent_replicates():
    value = random_split.gap(
        {"macro_f1": 0.9}, {"macro_f1": 0.7}, np.full(10, 0.9), np.full(10, 0.7)
    )
    assert value["difference"] == pytest.approx(0.2)
    assert value["ci_low"] == pytest.approx(0.2)
    assert value["ci_high"] == pytest.approx(0.2)


def write_inputs(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    records = make_records()
    rng = np.random.default_rng(0)
    parts = {"train": records[::3], "val": records[1::3], "test": records[2::3]}
    for name, part in parts.items():
        (processed / f"{name}.json").write_text(json.dumps(part))
        labels = np.array([r["label"] for r in part])
        mean = rng.normal(0, 1, (len(part), 8)) + labels[:, None] * 3
        np.savez(
            embeddings / f"{name}.npz", uniprot_ids=np.array([r["uniprot_id"] for r in part]),
            labels=labels, clusters=np.array([r["cluster_id"] for r in part]),
            mean=mean.astype(np.float16), cls=mean.astype(np.float16),
        )
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "3.4": 1}))

    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    table = [
        {
            "uniprot_id": r["uniprot_id"], "cluster_id": r["cluster_id"], "n_hits": 1,
            "best_target": "t", "best_identity": 35.0, "max_identity": 35.0,
            "max_identity_covered": None,
        }
        for r in parts["test"]
    ]
    (identity_dir / "test_identity.json").write_text(json.dumps(table))

    ids = np.array([r["uniprot_id"] for r in parts["test"]])
    predicted = np.array([r["label"] for r in parts["test"]])
    probe_dir, neighbour_dir = tmp_path / "probe", tmp_path / "similarity"
    probe_dir.mkdir()
    neighbour_dir.mkdir()
    np.savez(probe_dir / "test_predictions_mean.npz", uniprot_ids=ids, predictions=predicted)
    np.savez(
        neighbour_dir / "test_predictions_neighbour.npz", uniprot_ids=ids, predictions=predicted
    )
    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{identity_dir.as_posix()}'",
        f"embeddings.output_dir='{embeddings.as_posix()}'",
        f"probe.output_dir='{probe_dir.as_posix()}'",
        "probe.C_grid=[0.1,1.0]",
        f"similarity.output_dir='{neighbour_dir.as_posix()}'",
        f"random_split.work_dir='{(tmp_path / 'work').as_posix()}'",
        f"random_split.output_dir='{(tmp_path / 'results').as_posix()}'",
        "random_split.n_bootstrap=20",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])


def fake_search(calls):
    """Every test sequence's hit is a training sequence of the same class."""

    def search(query_fasta, target_fasta, work_dir, *args):
        calls.append(query_fasta)
        targets = {}
        for line in target_fasta.read_text().splitlines():
            if line.startswith(">"):
                targets.setdefault(line[1:].split("-")[0], line[1:])
        rows = []
        queries = [ln[1:] for ln in query_fasta.read_text().splitlines() if ln.startswith(">")]
        for i, query in enumerate(queries):
            target = targets[query.split("-")[0]]
            # Every second hit is partial, so those sequences have no covered relative
            coverage = 0.5 if i % 2 else 0.95
            rows.append(f"{query}\t{target}\t95.0\t100\t{coverage}\t{coverage}\t1e-30\t300.0\n")
        path = work_dir / "hits.tsv"
        path.write_text("".join(rows))
        return path

    return search


def test_run_end_to_end_and_reuses_the_search(tmp_path, monkeypatch):
    cfg = write_inputs(tmp_path)
    calls = []
    monkeypatch.setattr(random_split, "run_search", fake_search(calls))

    random_split.run(cfg)
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results["n_test"] == 26 and results["n_val"] == 27 and results["n_train"] == 27
    assert results["random"]["all"]["models"]["neighbour"]["accuracy"] == 1.0
    assert results["random"][">=70"]["n_sequences"] == 26
    assert results["homology"]["30-40"]["n_sequences"] == 26
    gaps = results["gap_random_minus_homology"]
    assert set(gaps) == {"all", "relative_free"}
    assert set(gaps["all"]) == {"probe", "neighbour"}
    assert gaps["all"]["neighbour"]["difference"] == pytest.approx(0.0)
    assert (tmp_path / "results" / "identity_report.json").exists()
    assert (tmp_path / "work" / "seed_42" / "test_identity.json").exists()

    random_split.run(cfg)
    assert len(calls) == 1


def test_missing_embeddings_are_reported(tmp_path):
    cfg = write_inputs(tmp_path)
    (tmp_path / "embeddings" / "val.npz").unlink()
    with pytest.raises(FileNotFoundError, match="src.embeddings"):
        random_split.run(cfg)
