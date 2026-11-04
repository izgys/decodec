"""
src/interpretability/conservation.py

Per-residue conservation of one sequence across its family.

How:
    Each homolog is aligned to the query on its own (local alignment, BLOSUM62),
    and the aligned residues are stacked over the query's positions. Insertions
    relative to the query are dropped, so the columns match the query's residues.

Score of a column:
    (1 - entropy / log 20) x (share of homologs that are not a gap). 1 means every
    homolog has the same residue here; 0 means all gaps or a uniform mix.

Limits:
    Pairwise alignments can slip at distant homologs, and sequences are not
    weighted, so an over-sampled subfamily counts more than a rare one.
"""

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

GAP = "-"
STANDARD = "ACDEFGHIKLMNPQRSTVWY"
CODES = {aa: i for i, aa in enumerate(STANDARD)}


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def clean(sequence: str) -> str:
    """BLOSUM62 has no column for U, O or J."""
    return "".join(aa if aa in CODES else "X" for aa in sequence)


def align_to_query(query: str, homologs: list[str]) -> list[str]:
    """One string per homolog, as long as the query: its residue at each position, or a gap."""
    aligner = make_aligner()
    query = clean(query)
    rows = []
    for homolog in homologs:
        homolog = clean(homolog)
        row = [GAP] * len(query)
        alignments = aligner.align(query, homolog)
        if len(alignments):
            query_blocks, homolog_blocks = alignments[0].aligned
            for (q_start, q_end), (h_start, _) in zip(query_blocks, homolog_blocks):
                row[q_start:q_end] = homolog[h_start:h_start + q_end - q_start]
        rows.append("".join(row))
    return rows


def conservation_scores(rows: list[str]) -> np.ndarray:
    """The score of every column. Anything outside the 20 standard residues counts as a gap."""
    if not rows:
        raise ValueError("Conservation needs at least one row.")
    counts = np.zeros((len(rows[0]), len(STANDARD)))
    for row in rows:
        for position, aa in enumerate(row):
            code = CODES.get(aa)
            if code is not None:
                counts[position, code] += 1
    present = counts.sum(axis=1)
    freq = counts / np.maximum(present, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(freq > 0, freq * np.log(freq), 0.0), axis=1)
    return (1 - entropy / np.log(len(STANDARD))) * (present / len(rows))
