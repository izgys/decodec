"""Offline tests for src/data/preprocess.py on synthetic records."""

import json

from conftest import make_record

from src.data import preprocess


def test_get_subclasses():
    record = make_record("P1", ecs=["1.14.11.63", "3.4.-.-", "3.4.21.4"])
    assert preprocess.get_subclasses(record) == {"1.14", "3.4"}


def test_drop_multi_subclass_keeps_same_subclass_ecs():
    records = [
        make_record("single", ecs=["3.4.21.4"]),
        make_record("same_subclass", ecs=["3.4.21.4", "3.4.22.1"]),
        make_record("two_subclasses", ecs=["1.14.11.63", "3.4.-.-"]),
    ]
    kept, n_dropped = preprocess.drop_multi_subclass(records)
    assert [r["uniprot_id"] for r in kept] == ["single", "same_subclass"]
    assert n_dropped == 1


def test_deduplicate_keeps_first_and_drops_conflicts():
    records = [
        make_record("a1", sequence="AAAA", ecs=["3.4.21.4"]),
        make_record("a2", sequence="AAAA", ecs=["3.4.22.1"]),   # same sub-class: duplicate
        make_record("c1", sequence="CCCC", ecs=["3.1.1.1"]),
        make_record("c2", sequence="CCCC", ecs=["3.4.21.4"]),   # conflicting label
        make_record("u1", sequence="DDDD", ecs=["2.7.11.1"]),
    ]
    unique, n_duplicates, n_conflicting = preprocess.deduplicate(records)
    assert [r["uniprot_id"] for r in unique] == ["a1", "u1"]
    assert n_duplicates == 1
    assert n_conflicting == 2
    # The counts account for every record
    assert len(records) == len(unique) + n_duplicates + n_conflicting


def test_filter_rare_classes():
    records = (
        [make_record(f"big{i}", ecs=["2.7.11.1"]) for i in range(5)]
        + [make_record(f"small{i}", ecs=["4.2.1.1"]) for i in range(2)]
    )
    kept, dropped = preprocess.filter_rare_classes(records, min_count=3)
    assert dropped == ["4.2"]
    assert {r["ec_subclass"] for r in kept} == {"2.7"}


def test_run_report_counts_add_up(tmp_path):
    raw = [
        make_record("multi", sequence="MMMM", ecs=["1.1.1.1", "2.7.1.1"]),
        make_record("dup1", sequence="AAAA", ecs=["2.7.1.1"]),
        make_record("dup2", sequence="AAAA", ecs=["2.7.1.2"]),
        make_record("conf1", sequence="CCCC", ecs=["2.7.1.1"]),
        make_record("conf2", sequence="CCCC", ecs=["3.1.1.1"]),
        make_record("rare", sequence="RRRR", ecs=["6.1.1.1"]),
    ] + [make_record(f"keep{i}", sequence="K" * (i + 5), ecs=["2.7.1.1"]) for i in range(3)]

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "raw.json").write_text(json.dumps(raw))
    config = tmp_path / "config.yaml"
    config.write_text(
        "filtering:\n  min_class_count: 2\n"
        "output:\n"
        f"  raw_dir: '{raw_dir.as_posix()}'\n"
        "  raw_filename: raw.json\n"
        f"  processed_dir: '{(tmp_path / 'processed').as_posix()}'\n"
    )

    preprocess.run(str(config))

    report = json.loads((tmp_path / "processed" / "preprocessing_report.json").read_text())
    assert report["n_raw"] == 9
    assert report["n_multi_subclass_dropped"] == 1
    assert report["n_duplicates_removed"] == 1
    assert report["n_conflicting_duplicates_dropped"] == 2
    assert report["n_after_dedup"] == 5
    assert report["dropped_classes"] == ["6.1"]
    assert report["n_rare_class_records_dropped"] == 1
    assert report["n_final_records"] == 4
    assert (
        report["n_raw"]
        - report["n_multi_subclass_dropped"]
        - report["n_duplicates_removed"]
        - report["n_conflicting_duplicates_dropped"]
        - report["n_rare_class_records_dropped"]
        == report["n_final_records"]
    )
    # The label map is built later, by clustering.py
    assert not (tmp_path / "processed" / "label_map.json").exists()
