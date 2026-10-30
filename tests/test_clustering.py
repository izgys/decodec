"""
Offline tests for the split logic in src/data/clustering.py.

MMseqs2 (Docker) is not run: tests build the member -> cluster mapping by hand.
"""

from collections import Counter, defaultdict

import pytest
from conftest import make_record

from src.data import clustering


def make_class(subclass: str, n_clusters: int, per_cluster: int = 2):
    """Records for one sub-class, spread over n_clusters clusters."""
    records, member_to_cluster = [], {}
    for c in range(n_clusters):
        cluster_id = f"{subclass}_c{c}"
        for m in range(per_cluster):
            uid = f"{cluster_id}_m{m}"
            records.append(make_record(uid, ecs=[f"{subclass}.1.1"]))
            member_to_cluster[uid] = cluster_id
    return records, member_to_cluster


def make_dataset(class_sizes: dict[str, int]):
    records, member_to_cluster = [], {}
    for subclass, n_clusters in class_sizes.items():
        r, m = make_class(subclass, n_clusters)
        records += r
        member_to_cluster.update(m)
    return records, member_to_cluster


def test_parse_cluster_tsv(tmp_path):
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("A\tA\nA\tB\nC\tC\n\n")
    assert clustering.parse_cluster_tsv(tsv) == {"A": "A", "B": "A", "C": "C"}


def test_cluster_label_is_majority():
    records = [
        make_record("x1", ecs=["3.4.1.1"]),
        make_record("x2", ecs=["3.4.1.1"]),
        make_record("x3", ecs=["2.7.1.1"]),
    ]
    member_to_cluster = {"x1": "X", "x2": "X", "x3": "X"}
    _, cluster_to_subclass = clustering.get_cluster_subclasses(records, member_to_cluster)
    assert cluster_to_subclass == {"X": "3.4"}


def test_filter_drops_classes_with_too_few_clusters():
    records, member_to_cluster = make_dataset({"2.7": 10, "3.4": 4})
    kept, dropped = clustering.filter_classes_by_clusters(records, member_to_cluster, 5)
    assert dropped == {"3.4": 4}
    assert {r["ec_subclass"] for r in kept} == {"2.7"}


def test_filter_drops_class_present_only_as_minority():
    records, member_to_cluster = make_dataset({"2.7": 10})
    # One 1.1 protein inside a 2.7 cluster: 0 clusters of its own
    records.append(make_record("stray", ecs=["1.1.1.1"]))
    member_to_cluster["stray"] = "2.7_c0"
    kept, dropped = clustering.filter_classes_by_clusters(records, member_to_cluster, 5)
    assert dropped == {"1.1": 0}
    assert "stray" not in {r["uniprot_id"] for r in kept}


def split_cluster_counts(records, member_to_cluster, cluster_to_split):
    _, cluster_to_subclass = clustering.get_cluster_subclasses(records, member_to_cluster)
    counts = defaultdict(Counter)
    for cluster_id, split in cluster_to_split.items():
        counts[cluster_to_subclass[cluster_id]][split] += 1
    return counts


@pytest.mark.parametrize("n_clusters", [7, 8, 20, 21, 100])
def test_assign_splits_guarantees_minimum_val_and_test(n_clusters):
    records, member_to_cluster = make_dataset({"2.7": n_clusters})
    cluster_to_split = clustering.assign_splits(
        records, member_to_cluster, 0.70, 0.15, seed=42, min_eval_clusters=3
    )
    counts = split_cluster_counts(records, member_to_cluster, cluster_to_split)["2.7"]
    assert counts["val"] >= 3
    assert counts["test"] >= 3
    assert counts["train"] >= 1
    assert sum(counts.values()) == n_clusters


def test_assign_splits_keeps_proportions_for_large_classes():
    records, member_to_cluster = make_dataset({"2.7": 200})
    cluster_to_split = clustering.assign_splits(records, member_to_cluster, 0.70, 0.15, 42, 3)
    counts = Counter(cluster_to_split.values())
    assert counts == {"train": 140, "val": 30, "test": 30}


def test_assign_splits_raises_when_class_is_too_small():
    records, member_to_cluster = make_dataset({"2.7": 6})
    with pytest.raises(ValueError, match="2.7"):
        clustering.assign_splits(records, member_to_cluster, 0.70, 0.15, 42, 3)


def test_assign_splits_is_deterministic_and_seed_dependent():
    records, member_to_cluster = make_dataset({"2.7": 30, "3.4": 25})
    first = clustering.assign_splits(records, member_to_cluster, 0.70, 0.15, 42, 3)
    # Same records in a different order give the same split
    second = clustering.assign_splits(records[::-1], member_to_cluster, 0.70, 0.15, 42, 3)
    other_seed = clustering.assign_splits(records, member_to_cluster, 0.70, 0.15, 7, 3)
    assert first == second
    assert first != other_seed


def test_no_cluster_crosses_splits():
    records, member_to_cluster = make_dataset({"2.7": 30, "3.4": 25})
    cluster_to_split = clustering.assign_splits(records, member_to_cluster, 0.70, 0.15, 42, 3)
    train, val, test = clustering.apply_splits_to_records(
        records, member_to_cluster, cluster_to_split
    )
    assert len(train) + len(val) + len(test) == len(records)
    seen = {}
    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        for r in split_records:
            assert r["split"] == split_name
            assert seen.setdefault(r["cluster_id"], split_name) == split_name


def test_label_map_is_sorted_and_applied():
    records = [make_record("a", ecs=["3.4.1.1"]), make_record("b", ecs=["1.10.1.1"])]
    label_map = clustering.build_label_map(records)
    assert label_map == {"1.10": 0, "3.4": 1}
    labelled = clustering.apply_labels(records, label_map)
    assert [r["label"] for r in labelled] == [1, 0]
