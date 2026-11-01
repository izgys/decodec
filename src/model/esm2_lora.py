"""
src/model/esm2_lora.py

ESM-2 with LoRA adapters and a mean-pooling classification head.

What this module does:
    - Freezes all pre-trained ESM-2 weights
    - Adds trainable low-rank matrices to the attention query and value
      projections of every layer (PEFT)
    - Classifies from the mean over residue tokens with one linear layer

Why not EsmForSequenceClassification?
    Its head classifies from the [CLS] token. The linear probe uses mean
    pooling (D2), so the fine-tuned model must too (D8); otherwise the
    comparison would change two things at once.

What a checkpoint holds:
    Only the trainable parameters (adapters and head), a few MB. The frozen
    weights always come from the pre-trained model.
"""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import EsmModel

from src.embeddings import pool


class EsmLoraClassifier(nn.Module):
    def __init__(
        self,
        esm: EsmModel,
        n_classes: int,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.esm = get_peft_model(esm, LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=list(target_modules),
            lora_dropout=dropout,
            bias="none",
        ))
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(esm.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        _, mean = pool(hidden, attention_mask)
        return self.head(self.dropout(mean))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """The parameters that change in training, on the CPU."""
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Checkpoint does not match this model's trainable parameters")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(params[name].device))

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    model_name: str,
    n_classes: int,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    head_dropout: float,
    device: torch.device,
    base_fp16: bool = False,
    gradient_checkpointing: bool = False,
) -> EsmLoraClassifier:
    """
    base_fp16 stores the frozen weights in half precision, which saves about
    half their memory; the adapters and head stay in float32.
    gradient_checkpointing trades compute for memory: activations are
    recomputed in the backward pass instead of being kept.
    """
    dtype = torch.float16 if base_fp16 else torch.float32
    esm = EsmModel.from_pretrained(model_name, add_pooling_layer=False, dtype=dtype)
    if gradient_checkpointing:
        # Non-reentrant: works although the frozen inputs do not require grad
        esm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = EsmLoraClassifier(esm, n_classes, r, alpha, dropout, target_modules, head_dropout)
    return model.to(device)
