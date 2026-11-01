"""
Offline tests for src/data/identity.py.

MMseqs2 (Docker) is not run: tests write the hits TSV by hand, in the format
the search produces (query, target, pident, alnlen, qcov, tcov, evalue, bits).
"""

import json

import pytest
from conftest import make_record

from src.config import load_config
from src.data import identity


def write_tsv(path, rows):
    path.write_text("".join("\t".join(str(x) for x in row) + "\n" for row in rows))
    return path


def test_best_hit_is_chosen_by_bit_score_not_identity(tmp_path):
    tsv = write_tsv(tmp_path / "hits.tsv", [
        ("q1", "short", 95.0, 30, 0.1, 0.1, 1e-2, 40.0),     # high identity, short
        ("q1", "long", 28.0, 300, 0.9, 0.9, 1e-30, 200.0),   # low identity, strong
    ])
    row = identity.summarise_hits(tsv)["q1"]
    assert row["best_target"] == "long"
    assert row["best_identity"] == 28.0
    assert row["best_bits"] == 200.0
    assert row["n_hits"] == 2


def test_max_identity_with_and_without_coverage(tmp_path):
    tsv = write_tsv(tmp_path / "hits.tsv", [
        ("q1", "domain", 60.0, 80, 0.3, 0.2, 1e-5, 60.0),      # shared domain
        ("q1", "full", 32.0, 300, 0.9, 0.85, 1e-20, 150.0),    # passes clustering coverage
        ("q1", "one_sided", 45.0, 250, 0.9, 0.5, 1e-10, 90.0), # covers only the query
    ])
    row = identity.summarise_hits(tsv)["q1"]
    assert row["max_identity"] == 60.0
    assert row["max_identity_covered"] == 32.0


def test_query_with_only_partial_hits_has_no_covered_identity(tmp_path):
    tsv = write_tsv(tmp_path / "hits.tsv", [("q1", "t", 50.0, 50, 0.2, 0.2, 1e-3, 30.0)])
    assert identity.summarise_hits(tsv)["q1"]["max_identity_covered"] is None


def test_table_keeps_sequences_without_hits(tmp_path):
    records = [
        make_record("q1", cluster_id="c1"),
        make_record("q2", cluster_id="c2"),
    ]
    tsv = write_tsv(tmp_path / "hits.tsv", [("q1", "t", 40.0, 100, 0.9, 0.9, 1e-9, 80.0)])
    table = identity.build_table(records, identity.summarise_hits(tsv))
    assert [row["uniprot_id"] for row in table] == ["q1", "q2"]
    assert table[0]["best_target"] == "t"
    assert table[1]["n_hits"] == 0
    assert "best_target" not in table[1]


def make_table():
    records = [
        make_record("a", ecs=["3.4.1.1"], cluster_id="c1"),
        make_record("b", ecs=["3.4.1.1"], cluster_id="c1"),
        make_record("c", ecs=["2.7.1.1"], cluster_id="c2"),
        make_record("d", ecs=["2.7.1.1"], cluster_id="c3"),
    ]
    def row(n_hits, best, maximum, covered):
        return {
            "n_hits": n_hits, "best_identity": best,
            "max_identity": maximum, "max_identity_covered": covered,
        }

    table = identity.build_table(records, {
        "a": row(2, 22.0, 55.0, 22.0),
        "b": row(1, 24.0, 24.0, None),
        "c": row(5, 100.0, 100.0, 100.0),
    })
    return table, records


def test_histogram_counts_sequences_clusters_and_classes():
    table, records = make_table()
    rows = {r["identity"]: r for r in identity.histogram(table, records, "best_identity")}
    assert rows["20-25"] == {
        "identity": "20-25", "n_sequences": 2, "n_clusters": 1, "n_classes": 1,
    }
    # 100% identity falls in the last bin; a sequence with no hit in "none"
    assert rows["95-100"]["n_sequences"] == 1
    assert rows["none"]["n_sequences"] == 1


def test_report_counts_leakage_and_holds_no_scores():
    table, records = make_table()
    report = identity.build_report(table, records)
    assert report["n_test"] == 4
    assert report["n_with_hit"] == 3
    assert report["n_no_hit"] == 1
    assert report["share_with_hit"] == 0.75
    assert report["n_covered_identity_over_30"] == 1
    assert report["n_any_identity_over_30"] == 2
    assert report["n_any_identity_over_90"] == 1
    # Bins are chosen from this report, so it must not reveal any accuracy
    assert "best_target" not in json.dumps(report)
    assert "label" not in json.dumps(report)


def test_run_end_to_end_with_a_fake_search(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    train = [make_record("t1", sequence="MKT" * 20, label=0, cluster_id="c9")]
    test = [
        make_record("q1", sequence="MKA" * 20, label=0, cluster_id="c1"),
        make_record("q2", sequence="GGG" * 20, label=0, cluster_id="c2"),
    ]
    (processed / "train.json").write_text(json.dumps(train))
    (processed / "test.json").write_text(json.dumps(test))
    work_dir = tmp_path / "identity"

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        write_tsv(work_dir / "test_vs_train.tsv", [
            ("q1", "t1", 66.7, 60, 1.0, 1.0, 1e-10, 70.0),
        ])

        class Done:
            returncode = 0
        return Done()

    monkeypatch.setattr(identity.subprocess, "run", fake_run)
    cfg = load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"data.identity.work_dir='{work_dir.as_posix()}'",
        f"data.identity.report_dir='{(tmp_path / 'results').as_posix()}'",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])
    identity.run(cfg)

    # The search has no coverage cutoff, and asks for the columns it parses
    command = calls[0]
    assert "easy-search" in command
    assert "-c" not in command
    assert command[command.index("--format-output") + 1] == identity.FORMAT_OUTPUT
    assert ">q1" in (work_dir / "test.fasta").read_text()
    assert ">t1" in (work_dir / "train.fasta").read_text()

    table = json.loads((work_dir / "test_identity.json").read_text())
    assert table[0]["best_target"] == "t1"
    assert table[1]["n_hits"] == 0
    report = json.loads((tmp_path / "results" / "identity_report.json").read_text())
    assert report["n_with_hit"] == 1
    assert report["n_covered_identity_over_50"] == 1


def test_failed_search_raises(tmp_path, monkeypatch):
    class Failed:
        returncode = 1

    monkeypatch.setattr(identity.subprocess, "run", lambda cmd, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="MMseqs2 failed"):
        identity.run_search(
            tmp_path / "test.fasta", tmp_path / "train.fasta", tmp_path, 7.5, 10, 300
        )
