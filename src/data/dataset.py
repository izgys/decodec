"""
src/data/dataset.py

PyTorch Dataset class for enzyme family classification with ESM-2.

What this module does:
    - Loads train/val/test split JSON files
    - Tokenises amino acid sequences using ESM-2's tokeniser
    - Returns (input_ids, attention_mask, label, index) for each record
    - Pads each batch to its own longest sequence (dynamic padding)
    - Groups sequences of similar length into the same batch

Key design decisions:
    - Tokenisation happens at __getitem__ time, not at load time
      Reason: loading all 150k sequences pre-tokenised would require
      ~10GB RAM. On-the-fly tokenisation costs negligible compute and
      keeps memory footprint small.
    - Max length 1026 tokens = 1024 residues + [CLS] + [EOS]
      ESM-2 has 1026 position embeddings. Sequences longer than 1024
      residues were already filtered in preprocessing, so nothing is
      truncated; truncation stays on defensively.
    - Dynamic padding instead of padding everything to the max length
      Most sequences are a few hundred residues long. Padding all of them
      to 1026 tokens would spend most of the compute on padding.
    - Length-grouped batches
      Dynamic padding only helps if a batch holds sequences of similar
      length. One 1000-residue sequence in a batch of 200-residue ones
      pads the whole batch to 1000.
    - Each item returns its index
      Length grouping reorders val and test, so predictions are mapped back
      to records by index, never by position in the loader.
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from transformers import EsmTokenizer

logger = logging.getLogger(__name__)

# 1024 residues + [CLS] + [EOS]; ESM-2 supports 1026 positions
MAX_TOKENS_PER_SEQUENCE = 1026


class EnzymeDataset(Dataset):
    """
    PyTorch Dataset for enzyme EC sub-class classification.

    Args:
        split_path:   Path to train.json, val.json, or test.json
        tokenizer:    ESM-2 tokeniser (EsmTokenizer)
        max_length:   Maximum tokens per sequence, including [CLS] and [EOS]
    """

    def __init__(
        self,
        split_path: str | Path,
        tokenizer: EsmTokenizer,
        max_length: int = MAX_TOKENS_PER_SEQUENCE,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        logger.info(f"Loading dataset from {split_path}...")
        with open(split_path) as f:
            self.records = json.load(f)
        logger.info(f"Loaded {len(self.records):,} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        sequence = record["sequence"]
        label = record["label"]

        # Tokenise the sequence
        # What the tokeniser does:
        #   1. Maps each amino acid character to an integer ID
        #   2. Prepends [CLS] token (id=0 in ESM-2 vocabulary)
        #   3. Appends [EOS] token
        #   4. Truncates to max_length (no padding here; the collator pads
        #      each batch to its own longest sequence)
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            # input_ids: integer token IDs, shape (n_tokens,)
            "input_ids": encoding["input_ids"].squeeze(0),
            # attention_mask: all 1 before padding, shape (n_tokens,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # label: integer class index
            "label": torch.tensor(label, dtype=torch.long),
            # index: position in self.records, to map predictions back
            "index": torch.tensor(idx, dtype=torch.long),
        }

    def token_lengths(self) -> list[int]:
        """Number of tokens per record, without tokenising ([CLS] + residues + [EOS])."""
        return [min(len(r["sequence"]) + 2, self.max_length) for r in self.records]

    def get_class_counts(self) -> dict[int, int]:
        """
        Count sequences per class label.
        Used for computing class weights for weighted sampling or loss.
        """
        return dict(Counter(r["label"] for r in self.records))

    def get_label_to_subclass(self) -> dict[int, str]:
        """
        Build {integer_label: ec_subclass_string} mapping from records.
        Useful for converting model predictions back to EC sub-class strings.
        """
        return {
            r["label"]: r["ec_subclass"]
            for r in self.records
        }


# ---------------------------------------------------------------------------
# Dynamic padding
# ---------------------------------------------------------------------------

class PaddingCollator:
    """
    Pads a list of items to the longest sequence in the batch.
    Padded positions get the tokeniser's pad ID and attention_mask 0.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "label": torch.stack([item["label"] for item in batch]),
            "index": torch.stack([item["index"] for item in batch]),
        }


# ---------------------------------------------------------------------------
# Length-grouped batching
# ---------------------------------------------------------------------------
# Training: shuffle all indices, cut them into "mega-batches" of
# batch_size * mega_batch_factor, sort each mega-batch by length, slice it
# into batches, then shuffle the order of the batches. Batches hold similar
# lengths, while which sequences meet in a batch still changes every epoch.
#
# Evaluation: sort everything by length, no shuffling, so the batches are
# the same in every run.
#
# max_tokens (optional) caps padded tokens per batch (batch rows x longest
# sequence). Batches of short sequences then hold more rows than batches of
# long ones, which keeps GPU memory use roughly constant.

class LengthGroupedBatchSampler(Sampler[list[int]]):
    """
    Yields lists of dataset indices, each a batch of similar-length sequences.

    Args:
        lengths:           Token length of each item (EnzymeDataset.token_lengths())
        batch_size:        Maximum sequences per batch
        max_tokens:        Optional cap on batch rows x longest sequence
        shuffle:           True for training, False for evaluation
        seed:              Base seed; the epoch is added (see set_epoch)
        mega_batch_factor: Mega-batch size in batches, when shuffling
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        max_tokens: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        mega_batch_factor: int = 50,
    ) -> None:
        if max_tokens is not None and max_tokens < max(lengths):
            raise ValueError(
                f"max_tokens ({max_tokens}) is smaller than the longest "
                f"sequence ({max(lengths)} tokens)"
            )
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.mega_batch_factor = mega_batch_factor
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch so every epoch gets new batches."""
        self.epoch = epoch

    def _split(self, sorted_indices: list[int]) -> list[list[int]]:
        """Cut length-sorted indices into batches under both limits."""
        batches, batch, longest = [], [], 0
        for idx in sorted_indices:
            new_longest = max(longest, self.lengths[idx])
            too_many = len(batch) >= self.batch_size
            too_long = (
                self.max_tokens is not None
                and new_longest * (len(batch) + 1) > self.max_tokens
            )
            if batch and (too_many or too_long):
                batches.append(batch)
                batch, new_longest = [], self.lengths[idx]
            batch.append(idx)
            longest = new_longest
        if batch:
            batches.append(batch)
        return batches

    def _batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            indices.sort(key=lambda i: self.lengths[i])
            return self._split(indices)

        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)
        mega = self.batch_size * self.mega_batch_factor
        batches = []
        for start in range(0, len(indices), mega):
            chunk = sorted(indices[start:start + mega], key=lambda i: self.lengths[i])
            batches.extend(self._split(chunk))
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def create_dataloaders(
    data_dir: str | Path,
    tokenizer: EsmTokenizer,
    batch_size: int = 16,
    max_length: int = MAX_TOKENS_PER_SEQUENCE,
    max_tokens: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple:
    """
    Create train, val, and test DataLoaders.

    Args:
        data_dir:    Directory containing train.json, val.json, test.json
        tokenizer:   ESM-2 tokeniser
        batch_size:  Maximum sequences per batch — keep low (8-16) for 8GB VRAM
        max_length:  Maximum tokens per sequence (1026 for ESM-2)
        max_tokens:  Optional cap on padded tokens per batch
        num_workers: Parallel data loading workers
                     Note: set to 0 on Windows — multiprocessing with
                     PyTorch DataLoader has known issues on Windows
        seed:        Seed for the training batch order

    Returns:
        (train_loader, val_loader, test_loader, train_dataset)
        We return train_dataset separately for class weight computation.
        Call train_loader.batch_sampler.set_epoch(epoch) at the start of
        each epoch.
    """
    data_dir = Path(data_dir)
    collator = PaddingCollator(tokenizer.pad_token_id)

    loaders = []
    train_dataset = None
    for split in ["train", "val", "test"]:
        dataset = EnzymeDataset(data_dir / f"{split}.json", tokenizer, max_length)
        # Why shuffle for train only?
        # Training: shuffling prevents the model from learning order-based
        # patterns and ensures each batch has a mix of classes.
        # Val/test: we never shuffle — results must be deterministic and
        # reproducible for fair comparison across runs.
        sampler = LengthGroupedBatchSampler(
            dataset.token_lengths(),
            batch_size=batch_size,
            max_tokens=max_tokens,
            shuffle=(split == "train"),
            seed=seed,
        )
        loaders.append(torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,   # faster GPU transfer
        ))
        if split == "train":
            train_dataset = dataset

    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader, train_dataset
