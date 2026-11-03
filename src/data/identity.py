"""
src/data/identity.py

Measures how close each test sequence is to the training set (D14), with an
MMseqs2 search of test against train, run via Docker.

What this script does:
    1. Writes the test and train sequences to FASTA
    2. Runs MMseqs2 easy-search, sensitive and with no coverage cutoff
    3. Summarises the hits per test sequence: the best hit by bit score, and
       the highest identity at each coverage level
    4. Saves the per-sequence table (data/identity/) and a report with the
       identity distribution and the leakage counts

Why search when the split is already homology-aware?
    The split guarantees that no cluster crosses splits, not that sequences
    from different clusters are unrelated. Clustering is greedy (members are
    compared with a representative, not with each other) and requires 80%
    coverage, so a shared domain or a pair that never met can still link test
    to train. This search measures that instead of assuming it away.

Why does the report contain no labels or scores?
    The identity bins are chosen from this report, before any per-bin score
    exists, so that bins cannot be picked to flatter a model. The nearest
    neighbour's accuracy is computed later, from the saved best hits.
"""

import json
import logging
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.config import cli_config
from src.data.clustering import MMSEQS2_IMAGE, write_fasta
from src.tracking import log_run, numeric_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Output columns, in order. pident is a percentage; qcov and tcov are fractions.
FORMAT_OUTPUT = "query,target,pident,alnlen,qcov,tcov,evalue,bits"

# The coverage the clustering requires of both sequences (-c 0.8 --cov-mode 0)
CLUSTER_COVERAGE = 0.8
HISTOGRAM_WIDTH = 5     # identity histogram bin width, in percentage points
NO_PREDICTION = -1      # nearest neighbour's label for a sequence with no hit


# ---------------------------------------------------------------------------
# MMseqs2 via Docker
# ---------------------------------------------------------------------------
# Key parameters:
#   -s 7.5                  → highest standard sensitivity, for remote homologues
#   -e 10                   → permissive: keep weak hits, so the nearest
#                             neighbour gets its best chance
#   --max-seqs              → prefilter hits kept per query
#   --alignment-mode 3      → compute exact identity and coverage for each hit
#   (no -c)                 → no coverage cutoff, so local matches stay visible

def run_search(
    query_fasta: Path,
    target_fasta: Path,
    work_dir: Path,
    sensitivity: float,
    max_evalue: float,
    max_seqs: int,
) -> Path:
    """Run MMseqs2 easy-search via Docker. Returns the path to the hits TSV."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{query_fasta.stem}_vs_{target_fasta.stem}.tsv"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir.resolve()}:/data",
        MMSEQS2_IMAGE,
        "easy-search",
        f"/data/{query_fasta.name}",
        f"/data/{target_fasta.name}",
        f"/data/{output_name}",
        "/data/tmp",
        "-s", str(sensitivity),
        "-e", str(max_evalue),
        "--max-seqs", str(max_seqs),
        "--alignment-mode", "3",
        "--format-output", FORMAT_OUTPUT,
        "-v", "1",
    ]
    logger.info(f"Running MMseqs2 search: sensitivity {sensitivity}, E-value {max_evalue}")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 failed with return code {result.returncode}")

    tsv_path = work_dir / output_name
    if not tsv_path.exists():
        raise FileNotFoundError(f"MMseqs2 output not found at {tsv_path}")
    return tsv_path


# ---------------------------------------------------------------------------
# Per-sequence summary
# ---------------------------------------------------------------------------

def summarise_hits(tsv_path: Path) -> dict[str, dict]:
    """
    Reduce the hits TSV to one row per query, reading it line by line (the
    file can hold millions of hits).

    best_*: the hit with the highest bit score, which the nearest-neighbour
    baseline will use.
    max_identity: highest identity of any hit, whatever its coverage.
    max_identity_covered: highest identity among hits covering at least 80% of
    both sequences, the coverage the clustering requires.
    """
    summary = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                continue
            query, target = parts[0], parts[1]
            identity, alnlen = float(parts[2]), int(parts[3])
            qcov, tcov, evalue, bits = (float(p) for p in parts[4:])

            row = summary.get(query)
            if row is None:
                row = summary[query] = {
                    "n_hits": 0, "best_bits": float("-inf"),
                    "max_identity": 0.0, "max_identity_covered": None,
                }
            row["n_hits"] += 1
            row["max_identity"] = max(row["max_identity"], identity)
            if qcov >= CLUSTER_COVERAGE and tcov >= CLUSTER_COVERAGE:
                previous = row["max_identity_covered"]
                row["max_identity_covered"] = (
                    identity if previous is None else max(previous, identity)
                )
            if bits > row["best_bits"]:
                row.update({
                    "best_target": target, "best_identity": identity,
                    "best_alnlen": alnlen, "best_qcov": qcov, "best_tcov": tcov,
                    "best_evalue": evalue, "best_bits": bits,
                })
    return summary


def build_table(test_records: list[dict], summary: dict[str, dict]) -> list[dict]:
    """One row per test sequence; a sequence with no hit keeps n_hits = 0."""
    table = []
    for record in test_records:
        row = {"uniprot_id": record["uniprot_id"], "cluster_id": record["cluster_id"]}
        row.update(summary.get(record["uniprot_id"], {"n_hits": 0}))
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def histogram(table: list[dict], records: list[dict], key: str) -> list[dict]:
    """
    Sequences, clusters and classes per identity bin. Clusters and classes
    show whether a bin could support a macro-F1. Rows with no value for key go
    in a "none" row.
    """
    groups = defaultdict(list)
    for row, record in zip(table, records):
        value = row.get(key)
        if value is None:
            groups[None].append((row, record))
        else:
            # Identity 100 goes in the last bin, 95-100
            low = min(int(value // HISTOGRAM_WIDTH) * HISTOGRAM_WIDTH, 100 - HISTOGRAM_WIDTH)
            groups[low].append((row, record))

    rows = []
    for low in sorted(groups, key=lambda x: -1 if x is None else x):
        members = groups[low]
        rows.append({
            "identity": "none" if low is None else f"{low}-{low + HISTOGRAM_WIDTH}",
            "n_sequences": len(members),
            "n_clusters": len({row["cluster_id"] for row, _ in members}),
            "n_classes": len({record["ec_subclass"] for _, record in members}),
        })
    return rows


def build_report(table: list[dict], records: list[dict]) -> dict:
    n = len(table)
    with_hit = [row for row in table if row["n_hits"] > 0]

    def over(key: str, threshold: float) -> int:
        return sum(1 for row in table if (row.get(key) or 0) >= threshold)

    return {
        "n_test": n,
        "n_with_hit": len(with_hit),
        "n_no_hit": n - len(with_hit),
        "share_with_hit": len(with_hit) / n if n else 0.0,
        "n_clusters_with_hit": len({row["cluster_id"] for row in with_hit}),
        "n_clusters": len({row["cluster_id"] for row in table}),
        # Leakage: pairs the clustering would have grouped, had it compared them
        "n_covered_identity_over_30": over("max_identity_covered", 30),
        "n_covered_identity_over_50": over("max_identity_covered", 50),
        "n_covered_identity_over_90": over("max_identity_covered", 90),
        # Local matches at any coverage, for example a shared domain
        "n_any_identity_over_30": over("max_identity", 30),
        "n_any_identity_over_50": over("max_identity", 50),
        "n_any_identity_over_90": over("max_identity", 90),
        "n_hits_per_query_histogram": dict(
            sorted(Counter(min(row["n_hits"] // 100 * 100, 300) for row in table).items())
        ),
        "best_hit_identity": histogram(table, records, "best_identity"),
        "max_identity": histogram(table, records, "max_identity"),
        "max_identity_covered": histogram(table, records, "max_identity_covered"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict_from_best_hit(table: list[dict], train_labels: dict[str, int]) -> np.ndarray:
    """The label of each sequence's best training hit; NO_PREDICTION if it has none."""
    return np.array(
        [train_labels[row["best_target"]] if row["n_hits"] else NO_PREDICTION for row in table]
    )


def build_views(
    table: list[dict], relative_identity: float, edges: list[float]
) -> dict[str, np.ndarray]:
    """Boolean mask per view. The identity bins partition the test set."""
    best = np.array([row.get("best_identity", np.nan) for row in table], dtype=float)
    covered = np.array(
        [row.get("max_identity_covered") or 0.0 for row in table], dtype=float
    )
    has_hit = ~np.isnan(best)

    views = {
        "all": np.ones(len(table), dtype=bool),
        "relative_free": covered < relative_identity,
        "no_hit": ~has_hit,
    }
    bounds = [-np.inf, *edges, np.inf]
    for low, high in zip(bounds[:-1], bounds[1:]):
        if low == -np.inf:
            name = f"<{high:g}"
        elif high == np.inf:
            name = f">={low:g}"
        else:
            name = f"{low:g}-{high:g}"
        views[name] = has_hit & (best >= low) & (best < high)
    return views


def run(cfg: DictConfig) -> None:
    identity = cfg.data.identity
    processed_dir = Path(cfg.data.output.processed_dir)
    work_dir = Path(identity.work_dir)
    report_dir = Path(identity.report_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "train.json") as f:
        train = json.load(f)
    with open(processed_dir / "test.json") as f:
        test = json.load(f)
    logger.info(f"Searching {len(test):,} test sequences against {len(train):,} training sequences")

    write_fasta(test, work_dir / "test.fasta")
    write_fasta(train, work_dir / "train.fasta")

    tsv_path = run_search(
        work_dir / "test.fasta", work_dir / "train.fasta", work_dir,
        identity.sensitivity, identity.max_evalue, identity.max_seqs,
    )

    table = build_table(test, summarise_hits(tsv_path))
    report = build_report(table, test)

    table_path = work_dir / "test_identity.json"
    with open(table_path, "w") as f:
        json.dump(table, f)
    report_path = report_dir / "identity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"With a hit: {report['n_with_hit']:,} of {report['n_test']:,} test sequences")
    logger.info(
        f"Covered identity over 30%: {report['n_covered_identity_over_30']:,}; "
        f"any coverage over 30%: {report['n_any_identity_over_30']:,}"
    )
    logger.info(f"Report saved to {report_path}")

    log_run(cfg, "decodec-data", "identity-search", numeric_metrics(report), [report_path])
    logger.info("Identity search complete.")


if __name__ == "__main__":
    run(cli_config("Search test against train and measure identity to the training set"))
