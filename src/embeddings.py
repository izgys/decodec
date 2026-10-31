"""
src/embeddings.py

Extracts frozen ESM-2 embeddings for every split, for the linear probe (D2).

What this script does:
    1. Loads ESM-2 with frozen weights (no fine-tuning), in fp16 on the GPU
    2. Runs every sequence at full length (up to 1,026 tokens), in
       length-grouped batches with dynamic padding
    3. Keeps two embeddings per sequence from the last layer:
         cls  — the [CLS] token
         mean — the mean over residue tokens ([CLS], [EOS] and padding excluded)
    4. Saves one .npz per split, keyed by UniProt ID, with labels and cluster IDs

Why both poolings in one pass?
    D2 uses mean pooling: ESM-2 is trained only with masked language
    modelling, so the [CLS] token never learns to summarise the sequence.
    Keeping CLS too makes the CLS-vs-mean comparison free.

Why keyed by UniProt ID?
    Batches are reordered by length, and a future re-split must not force a
    new extraction. Every row carries its ID, label and cluster.

Resuming:
    A split whose .npz already exists for the same model is skipped, so an
    interrupted run continues where it stopped. Delete the file to redo it.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

from src.config import cli_config
from src.data.dataset import EnzymeDataset, LengthGroupedBatchSampler, PaddingCollator
from src.tracking import log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    hidden: (batch, tokens, dim); attention_mask: (batch, tokens), 1 = real token.
    Returns (cls, mean), each (batch, dim), in float32.
    The mean covers residue tokens only: position 0 is [CLS], the last real
    token is [EOS], everything after it is padding.
    """
    hidden = hidden.float()
    lengths = attention_mask.sum(dim=1)                      # tokens incl. [CLS], [EOS]
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    residues = (positions >= 1) & (positions < (lengths - 1)[:, None])
    weights = residues.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    return hidden[:, 0], mean


@torch.inference_mode()
def extract_split(
    model: EsmModel,
    dataset: EnzymeDataset,
    pad_token_id: int,
    batch_size: int,
    max_tokens: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Embeddings for every record of one split, in the dataset's record order."""
    sampler = LengthGroupedBatchSampler(
        dataset.token_lengths(), batch_size=batch_size, max_tokens=max_tokens, shuffle=False
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=PaddingCollator(pad_token_id)
    )
    dim = model.config.hidden_size
    cls = np.zeros((len(dataset), dim), dtype=np.float16)
    mean = np.zeros((len(dataset), dim), dtype=np.float16)

    for batch in tqdm(loader, desc="Embedding", unit="batch"):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        batch_cls, batch_mean = pool(out.last_hidden_state, batch["attention_mask"].to(device))
        # Batches are reordered by length; the index puts each row back in place
        idx = batch["index"].numpy()
        cls[idx] = batch_cls.cpu().numpy()
        mean[idx] = batch_mean.cpu().numpy()

    records = dataset.records
    return {
        "uniprot_ids": np.array([r["uniprot_id"] for r in records]),
        "labels": np.array([r["label"] for r in records], dtype=np.int64),
        "clusters": np.array([r["cluster_id"] for r in records]),
        "cls": cls,
        "mean": mean,
    }


def embeddings_path(cfg: DictConfig, split: str) -> Path:
    return Path(cfg.embeddings.output_dir) / f"{split}.npz"


def run(cfg: DictConfig) -> None:
    emb = cfg.embeddings
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(emb.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (emb.fp16 and device.type == "cuda") else torch.float32
    logger.info(f"Model: {emb.model_name} on {device} ({dtype})")
    if device.type == "cpu":
        logger.warning("No GPU found: extraction will be very slow")

    tokenizer = EsmTokenizer.from_pretrained(emb.model_name)
    model = EsmModel.from_pretrained(emb.model_name, add_pooling_layer=False, dtype=dtype)
    model.to(device).eval()

    timings = {}
    for split in SPLITS:
        path = embeddings_path(cfg, split)
        if path.exists() and str(np.load(path)["model_name"]) == emb.model_name:
            logger.info(f"{split}: {path} exists for this model, skipping")
            continue
        dataset = EnzymeDataset(processed_dir / f"{split}.json", tokenizer)
        start = time.time()
        arrays = extract_split(
            model, dataset, tokenizer.pad_token_id, emb.batch_size, emb.max_tokens, device
        )
        timings[f"seconds_{split}"] = time.time() - start
        np.savez(path, model_name=np.array(emb.model_name), **arrays)
        logger.info(f"{split}: {len(dataset):,} embeddings saved to {path} "
                    f"({timings[f'seconds_{split}'] / 60:.1f} min)")

    if timings:
        log_run(cfg, "decodec-baselines", "embeddings", timings)
    logger.info("Embedding extraction complete.")


if __name__ == "__main__":
    run(cli_config("Extract frozen ESM-2 embeddings (CLS and mean) for every split"))
