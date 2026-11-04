"""
src/interpretability/scoring.py

What the attribution methods explain, and the embeddings they start from.

Both methods in this package need the same two things: a way to turn token ids
into embeddings they can perturb, and one number per sequence to explain. That
number is always a margin, the score of one class minus the score of another,
because a single class score says nothing about the choice between two classes.

Two models are explained:
    classifier_margin  the fine-tuned model (LoRA adapters active)
    probe_margin       the linear probe on frozen ESM-2 mean embeddings

The probe's margin is linear in the mean embedding, so it is one dot product:
    margin = ((w_p - w_q) / scale) . mean + offset
The division undoes the StandardScaler the probe was fitted behind.
"""

from collections.abc import Callable

import numpy as np
import torch

from src.embeddings import pool
from src.model.esm2_lora import EsmLoraClassifier

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

Scorer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def score_in_batches(
    score: Scorer, points: torch.Tensor, mask: torch.Tensor, max_tokens: int
) -> torch.Tensor:
    """score() over many perturbed sequences, in batches that fit the memory budget."""
    size = max(1, max_tokens // points.shape[1])
    out = []
    for i in range(0, len(points), size):
        chunk = points[i:i + size]
        with torch.no_grad():
            out.append(score(chunk, mask.expand(len(chunk), -1)))
    return torch.cat(out)


def token_embedding(model: EsmLoraClassifier, token_id: int, device: torch.device) -> torch.Tensor:
    """One token's embedding, scaled as the model scales it."""
    ids = torch.tensor([[token_id]], device=device)
    with torch.no_grad():
        return model.embed_tokens(ids)[0, 0]


def mean_residue_embedding(
    model: EsmLoraClassifier, token_ids: list[int], device: torch.device
) -> torch.Tensor:
    """The mean embedding over the 20 standard amino acids: the second baseline."""
    ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        return model.embed_tokens(ids)[0].mean(dim=0)


def classifier_margin(model: EsmLoraClassifier, predicted: int, competitor: int) -> Scorer:
    """The fine-tuned model's margin, with the LoRA adapters active."""

    def score(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = model.forward_embedded(tokens, mask)
        return logits[:, predicted] - logits[:, competitor]

    return score


def frozen_hidden(
    model: EsmLoraClassifier, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Final hidden states with the adapters switched off: frozen ESM-2, the model
    whose mean embeddings the probe was fitted on.

    Do not wrap this in gradient checkpointing. Checkpointing re-runs the forward
    pass during the backward pass, by which time this context manager has exited,
    so the adapters would be active on the recomputation and the gradients would
    belong to a different model than the scores.
    """
    with model.esm.disable_adapter():
        return model.esm(inputs_embeds=tokens, attention_mask=mask).last_hidden_state


def probe_margin(
    model: EsmLoraClassifier, direction: torch.Tensor, offset: float
) -> Scorer:
    """The probe's margin: a direction dotted with the frozen mean embedding."""

    def score(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, mean = pool(frozen_hidden(model, tokens, mask), mask)
        return mean @ direction + offset

    return score


def probe_direction(
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
    predicted: int,
    competitor: int,
) -> tuple[np.ndarray, float]:
    """
    The probe margin as a dot product on unscaled mean embeddings: (direction, offset).

    coefficients and intercepts come from the fitted LogisticRegression, centre
    and scale from the StandardScaler it was fitted behind.
    """
    direction = (coefficients[predicted] - coefficients[competitor]) / scale
    offset = float(intercepts[predicted] - intercepts[competitor] - direction @ centre)
    return direction, offset
