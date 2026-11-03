"""
Offline tests for the hybrid predictor, on synthetic records.

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


def load_module():
    path = REPO / "experiments" / "hybrid" / "run_hybrid.py"
    spec = importlib.util.spec_from_file_location("run_hybrid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hybrid = load_module()


def hit(evalue):
    return {"n_hits": 1, "best_evalue": evalue, "best_target": "t"}


def test_search_answers_only_at_or_below_the_threshold():
    table = [hit(1e-10), hit(1e-3), hit(1.0), {"n_hits": 0}]
    neighbour = np.array([1, 1, 1, -1])
    fallback = np.array([2, 2, 2, 2])
    predicted, trusted = hybrid.predict_hybrid(table, neighbour, fallback, 1e-3)
    assert predicted.tolist() == [1, 1, 2, 2]
    assert trusted.tolist() == [True, True, False, False]


def test_no_hit_never_reaches_the_neighbours_empty_label():
    predicted, _ = hybrid.predict_hybrid([{"n_hits": 0}], np.array([-1]), np.array([3]), 10.0)
    assert predicted.tolist() == [3]


def test_threshold_is_chosen_by_macro_f1():
    # Search is right at strong hits and wrong at weak ones; the fallback is the reverse
    y_true = np.array([0, 1, 0, 1])
    table = [hit(1e-20), hit(1e-20), hit(1.0), hit(1.0)]
    neighbour = np.array([0, 1, 1, 0])
    fallback = np.array([1, 0, 0, 1])
    threshold, sweep = hybrid.choose_threshold(
        y_true, table, neighbour, fallback, [1e-30, 1e-5, 10.0], [0, 1]
    )
    assert threshold == 1e-5
    assert [row["evalue_threshold"] for row in sweep] == [1e-30, 1e-5, 10.0]
    assert sweep[1]["macro_f1"] == 1.0
    assert sweep[1]["share_from_search"] == 0.5


def test_ties_go_to_the_strictest_threshold():
    y_true = np.array([0, 1])
    same = np.array([0, 1])
    threshold, _ = hybrid.choose_threshold(
        y_true, [hit(1e-20), hit(1e-20)], same, same, [10.0, 1e-30], [0, 1]
    )
    assert threshold == 1e-30


def write_inputs(tmp_path, evaluate_test=True, misaligned=False):
    processed = tmp_path / "processed"
    processed.mkdir()
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    rng = np.random.default_rng(0)
    parts = {}
    for name, n in [("train", 30), ("val", 12), ("test", 12)]:
        parts[name] = [
            make_record(f"{name}{label}-{i}", label=label, cluster_id=f"{name}{label}-{i // 2}")
            for label in range(2)
            for i in range(n)
        ]
        (processed / f"{name}.json").write_text(json.dumps(parts[name]))
        y = np.array([r["label"] for r in parts[name]])
        mean = rng.normal(0, 1, (len(y), 8)) + y[:, None] * 3
        np.savez(
            embeddings / f"{name}.npz",
            uniprot_ids=np.array([r["uniprot_id"] for r in parts[name]]),
            labels=y, clusters=np.array([r["cluster_id"] for r in parts[name]]),
            mean=mean.astype(np.float16), cls=mean.astype(np.float16),
        )
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "3.4": 1}))

    lora, probe, identity_dir = tmp_path / "lora", tmp_path / "probe", tmp_path / "identity"
    for folder in (lora, probe, identity_dir):
        folder.mkdir()
    for name in ["val", "test"]:
        ids = [r["uniprot_id"] for r in parts[name]]
        np.savez(
            lora / f"{name}_predictions.npz",
            uniprot_ids=np.array(ids[::-1] if misaligned else ids),
            predictions=np.array([r["label"] for r in parts[name]]),
        )
    test_ids = np.array([r["uniprot_id"] for r in parts["test"]])
    np.savez(
        probe / "test_predictions_mean.npz", uniprot_ids=test_ids,
        predictions=np.array([r["label"] for r in parts["test"]]),
    )
    table = []
    for i, r in enumerate(parts["test"]):
        row = {"uniprot_id": r["uniprot_id"], "cluster_id": r["cluster_id"], "n_hits": 0}
        if i:
            row.update({
                "n_hits": 1, "best_target": f"train{r['label']}-0", "best_identity": 35.0,
                "best_evalue": 1e-20, "max_identity": 35.0, "max_identity_covered": None,
            })
        table.append(row)
    (identity_dir / "test_identity.json").write_text(json.dumps(table))
    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{identity_dir.as_posix()}'",
        f"embeddings.output_dir='{embeddings.as_posix()}'",
        f"probe.output_dir='{probe.as_posix()}'",
        "probe.C_grid=[0.1,1.0]",
        f"hybrid.lora_predictions_dir='{lora.as_posix()}'",
        f"hybrid.work_dir='{(tmp_path / 'work').as_posix()}'",
        f"hybrid.output_dir='{(tmp_path / 'results').as_posix()}'",
        f"hybrid.evaluate_test={str(evaluate_test).lower()}",
        "hybrid.n_bootstrap=20",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])


def fake_search(calls):
    """Each validation sequence hits a training sequence of its own class."""

    def search(query_fasta, target_fasta, work_dir, *args):
        calls.append(query_fasta)
        rows = []
        for line in query_fasta.read_text().splitlines():
            if line.startswith(">"):
                query = line[1:]
                label = query.split("-")[0][-1]
                rows.append(f"{query}\ttrain{label}-0\t40.0\t100\t0.9\t0.9\t1e-20\t300.0\n")
        path = work_dir / "hits.tsv"
        path.write_text("".join(rows))
        return path

    return search


def test_validation_mode_never_reads_test(tmp_path, monkeypatch):
    cfg = write_inputs(tmp_path, evaluate_test=False)
    monkeypatch.setattr(hybrid, "run_search", fake_search([]))
    (tmp_path / "processed" / "test.json").unlink()
    (tmp_path / "embeddings" / "test.npz").unlink()
    (tmp_path / "lora" / "test_predictions.npz").unlink()
    (tmp_path / "probe" / "test_predictions_mean.npz").unlink()
    (tmp_path / "identity" / "test_identity.json").unlink()

    hybrid.run(cfg)
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results["test"] is None
    assert results["selected_fallback"] in {"lora", "probe"}
    assert results["validation"]["single_models_macro_f1"]["neighbour"] == 1.0
    assert not (tmp_path / "results" / "test_predictions.npz").exists()


def test_run_end_to_end_reuses_the_search(tmp_path, monkeypatch):
    cfg = write_inputs(tmp_path)
    calls = []
    monkeypatch.setattr(hybrid, "run_search", fake_search(calls))
    hybrid.run(cfg)
    hybrid.run(cfg)
    assert len(calls) == 1

    results = json.loads((tmp_path / "results" / "results.json").read_text())
    view = results["test"]["views"]["all"]
    assert view["n_sequences"] == 24
    assert view["models"]["neighbour"]["accuracy"] == pytest.approx(23 / 24)
    # The one sequence with no hit goes to the fallback, which is right
    assert view["models"][f"hybrid_{results['selected_fallback']}"]["accuracy"] == 1.0
    assert view["differences"]["hybrid_minus_neighbour"]["accuracy"]["difference"] > 0
    assert (tmp_path / "results" / "test_predictions.npz").exists()


def test_misaligned_predictions_are_rejected(tmp_path, monkeypatch):
    cfg = write_inputs(tmp_path, misaligned=True)
    monkeypatch.setattr(hybrid, "run_search", fake_search([]))
    with pytest.raises(ValueError, match="LoRA predictions are not aligned"):
        hybrid.run(cfg)
