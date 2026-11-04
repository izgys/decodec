"""
src/interpretability/occlusion.py

Per-residue attribution by masking one residue at a time.

The idea in one line:
    Replace one residue with the mask token, see how much the margin falls, and
    call that fall the residue's contribution.

Why this and not integrated gradients, for the fine-tuned model:
    Integrated gradients needs a path the gradient can be integrated along, and
    the fine-tuned model does not provide one (see integrated_gradients.py). This
    method asks the model only about inputs it could actually receive: one
    sequence with one position unknown. It needs one forward pass per residue,
    no path, no step count, and repeats to the same number every time.

What it gives up:
    Completeness. The scores do not add up to anything in particular, because
    masking two residues is not masking each of them separately. A high score
    means the prediction depends on this position; it does not partition the
    margin into shares. It also cannot see a pair of residues that only matter
    together, since every position is masked on its own.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.interpretability.scoring import Scorer, score_in_batches


@dataclass
class Occlusion:
    scores: np.ndarray          # the fall in margin when each residue is masked
    score: float                # the margin at the intact sequence


def occlude(
    embed_tokens,
    score: Scorer,
    input_ids: torch.Tensor,
    replacement: torch.Tensor,
    max_tokens: int,
) -> Occlusion:
    """
    Mask each residue of one sequence in turn, input_ids of shape (1, tokens).

    replacement is the embedding a masked residue is given. A positive score
    means the margin falls when the residue is hidden, so the residue supports
    the predicted class over its competitor.
    """
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        tokens = embed_tokens(input_ids)
    n_residues = input_ids.shape[1] - 2
    if n_residues < 1:
        raise ValueError("A sequence needs at least one residue.")

    intact = score_in_batches(score, tokens, mask, max_tokens)[0].item()
    scores = np.zeros(n_residues)
    batch_size = max(1, max_tokens // input_ids.shape[1])
    for start in range(0, n_residues, batch_size):
        positions = range(start, min(start + batch_size, n_residues))
        points = tokens.repeat(len(positions), 1, 1)
        for row, position in enumerate(positions):
            points[row, position + 1] = replacement
        values = score_in_batches(score, points, mask, max_tokens)
        scores[start:start + len(positions)] = intact - values.double().cpu().numpy()
    return Occlusion(scores=scores, score=intact)
