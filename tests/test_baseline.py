"""
Tests for src/embeddings.py and the linear probe, end to end, offline.

A tiny randomly initialised ESM model (2 layers, 16 dimensions) stands in for
ESM-2 650M: same architecture and tokeniser, no download, runs on CPU.
"""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_record
from transformers import EsmConfig, EsmModel

from src import embeddings
from src.config import load_config
from src.data.dataset import EnzymeDataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
REPO = Path(__file__).resolve().parent.parent


def test_pool_excludes_special_tokens_and_padding():
    # Two sequences: 3 residues and 1 residue, padded to 5 tokens
    hidden = torch.arange(2 * 5 * 1, dtype=torch.float32).reshape(2, 5, 1)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    cls, mean = embeddings.pool(hidden, mask)
    assert cls.squeeze().tolist() == [0.0, 5.0]
    # Sequence 0: residues at positions 1-3 -> (1+2+3)/3; sequence 1: position 1 only
    assert mean.squeeze().tolist() == [2.0, 6.0]


@pytest.fixture(scope="module")
def tiny_model(tokenizer, tmp_path_factory):
    config = EsmConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=1026,
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        position_embedding_type="rotary", token_dropout=False,
    )
    torch.manual_seed(0)
    model = EsmModel(config, add_pooling_layer=False).eval()
    path = tmp_path_factory.mktemp("tiny_esm")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, path


def write_splits(processed_dir: Path) -> None:
    """Three classes whose sequences differ in composition, so a probe can learn them."""
    rng = random.Random(0)
    alphabets = {0: "AILMV", 1: "DEKR", 2: "FWYH"}
    processed_dir.mkdir(parents=True)
    for split, n_clusters in [("train", 12), ("val", 4), ("test", 4)]:
        records = []
        for label, alphabet in alphabets.items():
            for c in range(n_clusters):
                for m in range(3):
                    length = rng.randint(50, 120)
                    seq = "".join(rng.choice(alphabet + AMINO_ACIDS[:3]) for _ in range(length))
                    records.append(make_record(
                        f"{split}-{label}-{c}-{m}", sequence=seq, label=label,
                        cluster_id=f"{split}-{label}-{c}", split=split,
                    ))
        (processed_dir / f"{split}.json").write_text(json.dumps(records))
    (processed_dir / "label_map.json").write_text(json.dumps({"1.1": 0, "2.7": 1, "3.4": 2}))


def test_extract_split_restores_record_order(tiny_model, tokenizer, tmp_path):
    model, _ = tiny_model
    write_splits(tmp_path / "processed")
    dataset = EnzymeDataset(tmp_path / "processed" / "val.json", tokenizer)
    arrays = embeddings.extract_split(
        model, dataset, tokenizer.pad_token_id, batch_size=4, max_tokens=400,
        device=torch.device("cpu"),
    )
    assert arrays["mean"].shape == (len(dataset), 16)
    assert arrays["uniprot_ids"].tolist() == [r["uniprot_id"] for r in dataset.records]
    # Batched, length-sorted extraction equals running each sequence alone
    for i in [0, 7, len(dataset) - 1]:
        item = dataset[i]
        with torch.inference_mode():
            out = model(
                input_ids=item["input_ids"][None], attention_mask=item["attention_mask"][None]
            )
        cls, mean = embeddings.pool(out.last_hidden_state, item["attention_mask"][None])
        np.testing.assert_allclose(arrays["mean"][i], mean[0].numpy(), atol=1e-3)
        np.testing.assert_allclose(arrays["cls"][i], cls[0].numpy(), atol=1e-3)


class Crashing(torch.nn.Module):
    """Wraps a model and raises on the n-th forward call, like a killed run."""

    def __init__(self, model, crash_at: int):
        super().__init__()
        self.model, self.crash_at, self.calls = model, crash_at, 0
        self.config = model.config

    def forward(self, **kwargs):
        self.calls += 1
        if self.calls == self.crash_at:
            raise KeyboardInterrupt
        return self.model(**kwargs)


def test_interrupted_extraction_resumes_from_checkpoint(tiny_model, tokenizer, tmp_path):
    model, _ = tiny_model
    write_splits(tmp_path / "processed")
    dataset = EnzymeDataset(tmp_path / "processed" / "train.json", tokenizer)
    args = (dataset, tokenizer.pad_token_id, 4, 400, torch.device("cpu"))
    reference = embeddings.extract_split(model, *args)

    checkpoint = tmp_path / "train.partial.npz"
    crashing = Crashing(model, crash_at=8)
    with pytest.raises(KeyboardInterrupt):
        embeddings.extract_split(crashing, *args, checkpoint_path=checkpoint, checkpoint_every=3)
    with np.load(checkpoint) as ckpt:
        n_saved = int(ckpt["done"].sum())
    assert 0 < n_saved < len(dataset)              # 6 batches saved before the crash

    resumed_model = Crashing(model, crash_at=-1)
    resumed = embeddings.extract_split(
        resumed_model, *args, checkpoint_path=checkpoint, checkpoint_every=3
    )
    n_batches = len(embeddings.LengthGroupedBatchSampler(
        dataset.token_lengths(), batch_size=4, max_tokens=400, shuffle=False
    ))
    assert resumed_model.calls == n_batches - 6      # only the missing batches were run
    np.testing.assert_allclose(resumed["mean"], reference["mean"], atol=1e-3)
    np.testing.assert_allclose(resumed["cls"], reference["cls"], atol=1e-3)


def load_probe_module():
    path = REPO / "experiments" / "baseline_linear_probe" / "run_probe.py"
    spec = importlib.util.spec_from_file_location("run_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embeddings_then_probe_end_to_end(tiny_model, tmp_path):
    _, model_path = tiny_model
    processed = tmp_path / "processed"
    write_splits(processed)
    cfg = load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"embeddings.model_name='{model_path.as_posix()}'",
        f"embeddings.output_dir='{(tmp_path / 'emb').as_posix()}'",
        "embeddings.max_tokens=1100",
        f"probe.output_dir='{(tmp_path / 'results').as_posix()}'",
        "probe.C_grid=[0.1,1.0]",
        "probe.n_bootstrap=50",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])

    embeddings.run(cfg)
    for split in embeddings.SPLITS:
        with np.load(tmp_path / "emb" / f"{split}.npz") as f:
            expected = {"uniprot_ids", "labels", "clusters", "cls", "mean", "model_name"}
            assert set(f.files) >= expected

    # A second run skips the splits that already exist
    mtime = (tmp_path / "emb" / "train.npz").stat().st_mtime
    embeddings.run(cfg)
    assert (tmp_path / "emb" / "train.npz").stat().st_mtime == mtime

    load_probe_module().run(cfg)
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results["n_classes"] == 3
    assert set(results["poolings"]) == {"mean", "cls"}
    mean = results["poolings"]["mean"]
    assert [s["C"] for s in mean["C_sweep"]] == [0.1, 1.0]
    assert mean["test"]["best_C"] in {0.1, 1.0}
    assert mean["test"]["ci_low"] <= mean["test"]["macro_f1"] <= mean["test"]["ci_high"]
    assert 0 <= mean["test"]["top_level_accuracy"] <= 1
    for name in ["per_class_mean.txt", "per_class_cls.json", "test_predictions_mean.npz"]:
        assert (tmp_path / "results" / name).exists()
    with np.load(tmp_path / "results" / "test_predictions_mean.npz") as f:
        assert f["probabilities"].shape == (len(f["labels"]), 3)
