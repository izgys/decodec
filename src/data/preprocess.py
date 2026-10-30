"""
src/data/preprocess.py

Preprocesses the raw UniProt download for model training.

What this script does:
    1. Loads the raw JSON from download.py
    2. Drops proteins whose EC numbers span more than one sub-class
    3. Deduplicates at 100% sequence identity, dropping identical sequences
       that carry conflicting labels
    4. Analyses class distribution and drops sub-classes below min_count
    5. Saves the processed dataset ready for clustering

The label map is built in clustering.py, because the final set of classes
is only known after the cluster-count filter.

Why drop multi-sub-class proteins?
    A protein annotated as both 1.14 and 3.4 has no single correct label;
    taking the first EC number would make the label arbitrary. They are
    about 0.4% of the raw data. Multi-label classification is a v2 extension.

Why deduplication before clustering?
    MMseqs2 clustering is the expensive step. Removing exact duplicates first
    reduces that cost and avoids identical sequences appearing in both train
    and test sets even after clustering (100% identity is always in the same
    cluster, but why pay for it).

Why drop rare sub-classes?
    A sub-class with 10 sequences cannot be split into train/val/test and
    evaluated meaningfully. We set a minimum count threshold and either drop
    or merge rare classes. Dropped classes are documented — we never silently
    discard data.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_records(raw_path: Path) -> list[dict]:
    logger.info(f"Loading records from {raw_path}...")
    with open(raw_path) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# Multi-sub-class proteins
# ---------------------------------------------------------------------------

def get_subclasses(record: dict) -> set[str]:
    """All EC sub-classes a record is annotated with, e.g. {"1.14", "3.4"}."""
    return {".".join(ec.split(".")[:2]) for ec in record["ec_numbers"]}


def drop_multi_subclass(records: list[dict]) -> tuple[list[dict], int]:
    """
    Drop records whose EC numbers span more than one sub-class.
    Returns (kept_records, n_dropped).
    """
    kept = [r for r in records if len(get_subclasses(r)) == 1]
    return kept, len(records) - len(kept)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
# We deduplicate on exact sequence identity (100%).
# Identical sequences with the same label: keep the first occurrence.
# Identical sequences with different labels: drop all of them, since the
# same input cannot have two correct answers (one group of 3 in the June
# 2026 download).

def deduplicate(records: list[dict]) -> tuple[list[dict], int, int]:
    """
    Remove records with identical sequences.
    Returns (deduplicated_records, n_duplicates_removed, n_conflicting_dropped).
    """
    groups = {}
    for record in tqdm(records, desc="Deduplicating"):
        groups.setdefault(record["sequence"], []).append(record)

    unique_records = []
    n_conflicting = 0
    for group in groups.values():
        if len({r["ec_subclass"] for r in group}) > 1:
            n_conflicting += len(group)
            continue
        unique_records.append(group[0])

    n_duplicates = len(records) - len(unique_records) - n_conflicting
    return unique_records, n_duplicates, n_conflicting


# ---------------------------------------------------------------------------
# Class distribution analysis
# ---------------------------------------------------------------------------

def analyse_distribution(records: list[dict]) -> Counter:
    return Counter(r["ec_subclass"] for r in records)


def print_distribution(counts: Counter, title: str = "EC sub-class distribution") -> None:
    logger.info(f"\n{title}")
    logger.info(f"{'Sub-class':<12} {'Count':>8} {'%':>6}")
    logger.info("-" * 30)
    total = sum(counts.values())
    for subclass, count in sorted(counts.items()):
        pct = 100 * count / total
        logger.info(f"  EC {subclass:<8} {count:>8,} {pct:>5.1f}%")
    logger.info(f"  {'TOTAL':<8} {total:>8,} 100.0%")
    logger.info(f"  Unique sub-classes: {len(counts)}")


# ---------------------------------------------------------------------------
# Rare class handling
# ---------------------------------------------------------------------------
# We drop sub-classes with fewer than min_count sequences.
#
# Why 100 as default?
#   With homology-aware splitting, effective sequences per class can be
#   much lower than raw counts (recall the aminoacyl-tRNA synthetase insight).
#   A sub-class with 100 raw sequences might have only 20-30 independent
#   clusters — barely enough for train/val/test. Below 100 raw sequences,
#   per-class F1 estimates become unreliable.
#
# This is only the first filter. Whether a class can be evaluated depends on
# its number of clusters, not sequences, so clustering.py applies a second
# filter on clusters per class.
#
# Dropped classes are always logged — never silently discarded.

def filter_rare_classes(
    records: list[dict], min_count: int
) -> tuple[list[dict], list[str]]:
    """
    Drop records from sub-classes with fewer than min_count sequences.
    Returns (filtered_records, list_of_dropped_subclasses).
    """
    counts = analyse_distribution(records)
    rare = {sc for sc, n in counts.items() if n < min_count}

    if rare:
        logger.warning(
            f"Dropping {len(rare)} sub-classes with fewer than {min_count} sequences:"
        )
        for sc in sorted(rare):
            logger.warning(f"  EC {sc}: {counts[sc]} sequences")

    filtered = [r for r in records if r["ec_subclass"] not in rare]
    return filtered, sorted(rare)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config/data_config.yaml") -> None:
    config = load_config(config_path)
    filtering = config["filtering"]
    output_cfg = config["output"]

    raw_path = Path(output_cfg["raw_dir"]) / output_cfg["raw_filename"]
    processed_dir = Path(output_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load
    records = load_records(raw_path)
    n_raw = len(records)

    # Drop proteins with EC numbers in more than one sub-class
    records, n_multi = drop_multi_subclass(records)
    logger.info(f"Dropped {n_multi:,} multi-sub-class proteins → {len(records):,} records")

    # Deduplicate
    records, n_dupes, n_conflicting = deduplicate(records)
    n_after_dedup = len(records)
    logger.info(
        f"Removed {n_dupes:,} exact duplicates and {n_conflicting:,} identical sequences "
        f"with conflicting labels → {n_after_dedup:,} unique sequences"
    )

    # Distribution before filtering
    counts_before = analyse_distribution(records)
    print_distribution(counts_before, "Distribution before rare-class filtering")

    # Drop rare classes
    min_count = filtering.get("min_class_count", 100)
    records, dropped_classes = filter_rare_classes(records, min_count)
    logger.info(f"After rare-class filtering: {len(records):,} records")

    # Distribution after filtering
    counts_after = analyse_distribution(records)
    print_distribution(counts_after, "Distribution after rare-class filtering")

    # Save processed records
    processed_path = processed_dir / "enzymes_processed.json"
    logger.info(f"Saving processed records to {processed_path}...")
    with open(processed_path, "w") as f:
        json.dump(records, f, indent=2)

    # Save preprocessing report — every count is taken where it happens,
    # so the numbers always add up from n_raw to n_final_records
    report = {
        "n_raw": n_raw,
        "n_multi_subclass_dropped": n_multi,
        "n_duplicates_removed": n_dupes,
        "n_conflicting_duplicates_dropped": n_conflicting,
        "n_after_dedup": n_after_dedup,
        "n_rare_classes_dropped": len(dropped_classes),
        "n_rare_class_records_dropped": n_after_dedup - len(records),
        "dropped_classes": dropped_classes,
        "n_final_records": len(records),
        "n_classes": len(counts_after),
        "min_class_count": min_count,
        "class_counts": dict(counts_after.most_common()),
    }
    report_path = processed_dir / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess UniProt enzyme records")
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()
    run(args.config)