"""
src/interpretability/integrated_gradients.py

Integrated gradients, and the test of whether they mean anything here.

The idea in one line:
    Walk a straight line from a baseline input to the real sequence, add up the
    gradient along the way, and credit each residue with its share of the change.

Completeness, and why it is reported with every result:
    The scores must add up to score(input) - score(baseline). The gap between the
    two is the only honest measure of whether n_steps was enough, so attribute()
    returns it and the caller decides whether to trust the scores.

Why path_variation() exists:
    On the fine-tuned model this method does not converge. The margin swings up
    and down along the path by far more than it changes overall, and refining the
    path makes the total swing larger instead of settling, so no step count buys
    an answer. Switching the LoRA adapters off makes the same path well behaved,
    which is why the probe is attributed this way and the fine-tuned model is not
    (see src/interpretability/occlusion.py). path_variation() measures that, so
    the claim is a number in results.json rather than an opinion.

Baseline (D9):
    Every residue embedding replaced by the mask token's. ESM-2 is pre-trained
    with masked positions, so this is an input the model has seen: "a residue is
    here, but not which one". The mean amino acid embedding is used as a second
    baseline on a few sequences, to show how much the choice matters.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.interpretability.scoring import Scorer, score_in_batches


@dataclass
class Attribution:
    scores: np.ndarray          # one signed score per residue, no [CLS] or [EOS]
    score: float                # the margin at the real input
    baseline_score: float       # the margin at the baseline
    gap: float                  # |sum(scores) - (score - baseline_score)|
    relative_gap: float         # that gap as a share of the change it should explain


def quadrature(n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre points and weights on [0, 1]; far fewer steps than a Riemann sum."""
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    return (
        torch.tensor((nodes + 1) / 2, dtype=torch.float32),
        torch.tensor(weights / 2, dtype=torch.float32),
    )


def path(tokens: torch.Tensor, residue_baseline: torch.Tensor) -> torch.Tensor:
    """The baseline input: every residue replaced, [CLS] and [EOS] kept."""
    start = tokens.clone()
    start[:, 1:-1] = residue_baseline
    return start


def attribute(
    embed_tokens,
    score: Scorer,
    input_ids: torch.Tensor,
    residue_baseline: torch.Tensor,
    n_steps: int,
    max_tokens: int,
) -> Attribution:
    """
    Integrated gradients of one sequence, input_ids of shape (1, tokens).

    max_tokens limits the padded tokens per batch of path points, and so the memory.
    """
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        tokens = embed_tokens(input_ids)
    start = path(tokens, residue_baseline)
    ends = score_in_batches(score, torch.cat([start, tokens]), mask, max_tokens)
    baseline_score, end_score = ends[0].item(), ends[1].item()

    nodes, weights = quadrature(n_steps)
    nodes, weights = nodes.to(input_ids.device), weights.to(input_ids.device)
    direction = tokens - start
    batch_size = max(1, max_tokens // input_ids.shape[1])
    average_gradient = torch.zeros_like(tokens[0])
    for i in range(0, n_steps, batch_size):
        alpha = nodes[i:i + batch_size]
        points = (start + alpha[:, None, None] * direction).requires_grad_(True)
        total = score(points, mask.expand(len(alpha), -1)).sum()
        gradient, = torch.autograd.grad(total, points)
        average_gradient += (gradient * weights[i:i + batch_size, None, None]).sum(0).detach()

    scores = (average_gradient * direction[0]).sum(dim=-1)[1:-1].double().cpu().numpy()
    change = end_score - baseline_score
    gap = abs(float(scores.sum()) - change)
    return Attribution(
        scores=scores,
        score=end_score,
        baseline_score=baseline_score,
        gap=gap,
        relative_gap=gap / max(abs(change), 1e-8),
    )


def path_variation(
    embed_tokens,
    score: Scorer,
    input_ids: torch.Tensor,
    residue_baseline: torch.Tensor,
    n_points: int,
    max_tokens: int,
) -> dict:
    """
    How much the margin moves up and down along the path, sampled at n_points.

    A path integrated gradients can follow has a total variation that settles as
    n_points grows. One that keeps growing cannot be integrated at any step count.
    """
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        tokens = embed_tokens(input_ids)
    start = path(tokens, residue_baseline)
    direction = tokens - start
    alphas = torch.linspace(0, 1, n_points + 1, device=input_ids.device)
    # Built a batch at a time: all the points at once would be gigabytes on a long sequence
    batch_size = max(1, max_tokens // input_ids.shape[1])
    collected = []
    for i in range(0, len(alphas), batch_size):
        chunk = alphas[i:i + batch_size]
        points = start + chunk[:, None, None] * direction
        with torch.no_grad():
            collected.append(score(points, mask.expand(len(chunk), -1)))
    values = torch.cat(collected).double().cpu().numpy()
    steps = np.abs(np.diff(values))
    return {
        "n_points": n_points,
        "total_variation": float(steps.sum()),
        "max_step": float(steps.max()),
        "net_change": float(values[-1] - values[0]),
    }
