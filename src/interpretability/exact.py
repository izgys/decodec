"""
src/interpretability/exact.py

The one attribution in this project that needs no approximation.

The probe's margin is a direction dotted with a mean over residue positions, and
a dot product with a mean is a sum of per-position terms:

    margin = direction . mean(h) + offset = (1 / L) * sum_i direction . h_i + offset

so position i's share is exactly direction . h_i / L. No baseline, no path, no
step count, one forward pass. The shares add up to the margin by construction,
and split_margin() returns the residual as proof rather than as a hope.

What it does and does not say:
    h_i is the final hidden state at position i, after every attention layer has
    mixed the sequence, so a share says how much that position's representation
    carries the decision. It does not say the input residue at that position is
    needed: masking that residue often changes very little, because its
    neighbours supply the same information. On the enzymes tested here the two
    views barely agree at all, which is a fact about reading a transformer rather
    than a fault in either method, so the run reports both and their agreement.

This cannot be done for the fine-tuned model: its head reads the same mean, but
the adapters make h_i depend on the whole sequence in a way the split would hide
behind one position. The honest version there is masking (occlusion.py).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExactSplit:
    scores: np.ndarray          # one share per residue, no [CLS] or [EOS]
    score: float                # the margin at this sequence
    offset: float               # the part of the margin that belongs to no position
    gap: float                  # |sum(scores) + offset - score|, which should be ~0


def split_margin(
    hidden: torch.Tensor, direction: torch.Tensor, offset: float
) -> ExactSplit:
    """
    Split one sequence's margin over its residues.

    hidden is (1, tokens, dim), the final hidden states of a single unpadded
    sequence, so position 0 is [CLS] and the last position is [EOS].
    """
    if hidden.shape[0] != 1:
        raise ValueError("One sequence at a time.")
    residues = hidden[0, 1:-1].float()
    if not len(residues):
        raise ValueError("A sequence needs at least one residue.")
    shares = (residues @ direction / len(residues)).double().cpu().numpy()
    score = float(residues.mean(dim=0) @ direction) + offset
    return ExactSplit(
        scores=shares,
        score=score,
        offset=float(offset),
        gap=abs(float(shares.sum()) + float(offset) - score),
    )
