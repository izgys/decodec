"""Offline tests for src/data/dataset.py: tokenisation, padding and batching."""

import json
import random

import pytest
import torch
from conftest import make_record

from src.data.dataset import (
    MAX_TOKENS_PER_SEQUENCE,
    EnzymeDataset,
    LengthGroupedBatchSampler,
    PaddingCollator,
    create_dataloaders,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def write_split(path, n: int, seed: int = 0, max_residues: int = 300):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        length = rng.randint(50, max_residues)
        seq = "".join(rng.choice(AMINO_ACIDS) for _ in range(length))
        records.append(make_record(f"P{i}", sequence=seq, label=i % 3))
    path.write_text(json.dumps(records))
    return records


@pytest.fixture
def split_file(tmp_path):
    path = tmp_path / "train.json"
    write_split(path, 40)
    return path


def test_item_has_cls_eos_and_no_padding(split_file, tokenizer):
    dataset = EnzymeDataset(split_file, tokenizer)
    item = dataset[5]
    n_residues = len(dataset.records[5]["sequence"])
    assert item["input_ids"].shape == (n_residues + 2,)
    assert item["input_ids"][0] == tokenizer.cls_token_id
    assert item["input_ids"][-1] == tokenizer.eos_token_id
    assert item["attention_mask"].sum() == n_residues + 2
    assert item["index"] == 5
    assert item["label"] == dataset.records[5]["label"]


def test_token_lengths_match_tokenizer(split_file, tokenizer):
    dataset = EnzymeDataset(split_file, tokenizer)
    lengths = dataset.token_lengths()
    assert lengths == [dataset[i]["input_ids"].shape[0] for i in range(len(dataset))]


def test_full_length_sequence_is_not_truncated(tmp_path, tokenizer):
    path = tmp_path / "long.json"
    path.write_text(json.dumps([make_record("L", sequence="A" * 1024, label=0)]))
    dataset = EnzymeDataset(path, tokenizer)
    assert MAX_TOKENS_PER_SEQUENCE == 1026
    assert dataset[0]["input_ids"].shape == (1026,)
    assert dataset.token_lengths() == [1026]


def test_collator_pads_to_longest_in_batch(split_file, tokenizer):
    dataset = EnzymeDataset(split_file, tokenizer)
    items = [dataset[i] for i in range(4)]
    lengths = [item["input_ids"].shape[0] for item in items]
    batch = PaddingCollator(tokenizer.pad_token_id)(items)

    assert batch["input_ids"].shape == (4, max(lengths))
    assert batch["attention_mask"].sum(dim=1).tolist() == lengths
    padded = batch["attention_mask"] == 0
    assert (batch["input_ids"][padded] == tokenizer.pad_token_id).all()
    assert batch["index"].tolist() == [0, 1, 2, 3]
    assert batch["label"].dtype == torch.long


@pytest.fixture
def lengths():
    rng = random.Random(1)
    return [rng.randint(52, 1026) for _ in range(500)]


@pytest.mark.parametrize("shuffle", [True, False])
def test_sampler_yields_every_index_once(lengths, shuffle):
    sampler = LengthGroupedBatchSampler(lengths, batch_size=16, shuffle=shuffle)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    assert len(batches) == len(sampler)
    assert all(len(batch) <= 16 for batch in batches)


def test_sampler_respects_max_tokens(lengths):
    sampler = LengthGroupedBatchSampler(lengths, batch_size=64, max_tokens=4096)
    for batch in sampler:
        assert len(batch) * max(lengths[i] for i in batch) <= 4096


def test_sampler_rejects_max_tokens_below_longest_sequence(lengths):
    with pytest.raises(ValueError, match="max_tokens"):
        LengthGroupedBatchSampler(lengths, batch_size=16, max_tokens=500)


def test_sampler_groups_similar_lengths(lengths):
    def padded_tokens(batches):
        return sum(len(b) * max(lengths[i] for i in b) for b in batches)

    grouped = list(LengthGroupedBatchSampler(lengths, batch_size=16, shuffle=True))
    order = list(range(len(lengths)))
    random.Random(0).shuffle(order)
    unsorted = [order[i:i + 16] for i in range(0, len(order), 16)]
    assert padded_tokens(grouped) < 0.8 * padded_tokens(unsorted)


def test_training_batches_change_per_epoch_but_repeat_per_seed(lengths):
    sampler = LengthGroupedBatchSampler(lengths, batch_size=16, shuffle=True, seed=42)
    epoch0 = list(sampler)
    assert list(sampler) == epoch0
    sampler.set_epoch(1)
    assert list(sampler) != epoch0
    again = LengthGroupedBatchSampler(lengths, batch_size=16, shuffle=True, seed=42)
    assert list(again) == epoch0


def test_evaluation_batches_are_sorted_and_fixed(lengths):
    sampler = LengthGroupedBatchSampler(lengths, batch_size=16, shuffle=False)
    flat = [i for batch in sampler for i in batch]
    assert [lengths[i] for i in flat] == sorted(lengths)
    sampler.set_epoch(3)
    assert [i for batch in sampler for i in batch] == flat


def test_create_dataloaders_end_to_end(tmp_path, tokenizer):
    for i, split in enumerate(["train", "val", "test"]):
        write_split(tmp_path / f"{split}.json", 30, seed=i)
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        tmp_path, tokenizer, batch_size=8
    )
    assert len(train_dataset) == 30

    seen = []
    for batch in val_loader:
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape[0] <= 8
        seen += batch["index"].tolist()
    # Reordered by length, but every record exactly once and traceable by index
    assert sorted(seen) == list(range(30))

    train_loader.batch_sampler.set_epoch(1)
    assert sum(len(b["label"]) for b in train_loader) == 30
