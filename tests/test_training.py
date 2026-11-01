"""
Offline tests for the LoRA model, trainer and benchmark.

A tiny randomly initialised ESM model stands in for ESM-2: same architecture and
tokeniser, no download, runs on CPU. The crash tests check what matters on a
long run: that an interrupted run continues exactly where it stopped.
"""

import json
import random

import numpy as np
import pytest
import torch
from conftest import make_record
from transformers import EsmConfig, EsmModel

from src.config import load_config
from src.model.esm2_lora import EsmLoraClassifier, build_model
from src.training import benchmark, trainer

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_CLASSES = 3


@pytest.fixture(scope="module")
def tiny_model_path(tokenizer, tmp_path_factory):
    config = EsmConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=1026,
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        position_embedding_type="rotary", token_dropout=False,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_esm")
    EsmModel(config, add_pooling_layer=False).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def write_splits(processed):
    rng = random.Random(0)
    alphabets = {0: "AILMV", 1: "DEKR", 2: "FWYH"}
    processed.mkdir(parents=True)
    for split, n_clusters in [("train", 8), ("val", 3), ("test", 3)]:
        records = []
        for label, alphabet in alphabets.items():
            for c in range(n_clusters):
                for m in range(3):
                    length = rng.randint(30, 80)
                    seq = "".join(rng.choice(alphabet + AMINO_ACIDS[:3]) for _ in range(length))
                    records.append(make_record(
                        f"{split}-{label}-{c}-{m}", sequence=seq, label=label,
                        cluster_id=f"{split}-{label}-{c}", split=split,
                    ))
        (processed / f"{split}.json").write_text(json.dumps(records))
    (processed / "label_map.json").write_text(json.dumps({"1.1": 0, "2.7": 1, "3.4": 2}))


def make_cfg(tmp_path, model_path, name="run", **training):
    processed = tmp_path / "processed"
    if not processed.exists():
        write_splits(processed)
    settings = {
        "epochs": 2, "batch_size": 4, "max_tokens": 400, "eval_max_tokens": 400,
        "grad_accum_steps": 2, "checkpoint_minutes": 0, "n_bootstrap": 20,
        "gradient_checkpointing": False, **training,
    }
    return load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"lora.model_name='{model_path.as_posix()}'",
        f"training.output_dir='{(tmp_path / name).as_posix()}'",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
        *[f"training.{k}={v}" for k, v in settings.items()],
    ])


def make_model(path, **kwargs):
    return build_model(
        str(path), N_CLASSES, 2, 4, 0.1, ["query", "value"], 0.1, torch.device("cpu"), **kwargs
    )


def test_only_adapters_and_head_are_trainable(tiny_model_path):
    model = make_model(tiny_model_path)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names
    assert all("lora_" in n or n.startswith("head.") for n in names)
    assert any("query" in n for n in names) and any("value" in n for n in names)
    assert not any(".key." in n for n in names)
    # r=2 on 2 layers x 2 modules x (16 + 16) + head 16x3 + 3
    assert model.n_trainable() == 2 * 2 * 2 * 32 + 16 * 3 + 3


def test_classification_uses_mean_pooling_not_cls(tiny_model_path, tokenizer):
    model = make_model(tiny_model_path).eval()
    ids = tokenizer("MKTAYIAKQR", return_tensors="pt")
    with torch.no_grad():
        logits = model(ids["input_ids"], ids["attention_mask"])
        hidden = model.esm(**ids).last_hidden_state
    expected = model.head(hidden[:, 1:-1].mean(dim=1))
    assert logits.shape == (1, N_CLASSES)
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


def test_padding_does_not_change_the_prediction(tiny_model_path, tokenizer):
    model = make_model(tiny_model_path).eval()
    ids = tokenizer("MKTAYIAKQR", return_tensors="pt")
    pad = torch.full((1, 5), tokenizer.pad_token_id)
    with torch.no_grad():
        alone = model(ids["input_ids"], ids["attention_mask"])
        padded = model(
            torch.cat([ids["input_ids"], pad], dim=1),
            torch.cat([ids["attention_mask"], torch.zeros_like(pad)], dim=1),
        )
    torch.testing.assert_close(alone, padded, atol=1e-4, rtol=1e-4)


def test_trainable_state_round_trip_and_mismatch(tiny_model_path):
    model = make_model(tiny_model_path)
    state = model.trainable_state()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    model.load_trainable_state(state)
    assert all(torch.equal(state[k], v) for k, v in model.trainable_state().items())
    with pytest.raises(ValueError, match="does not match"):
        model.load_trainable_state({"wrong": torch.zeros(1)})


def test_gradient_checkpointing_gives_gradients_to_adapters(tiny_model_path, tokenizer):
    model = make_model(tiny_model_path, gradient_checkpointing=True).train()
    ids = tokenizer("MKTAYIAKQR", return_tensors="pt")
    model(ids["input_ids"], ids["attention_mask"]).sum().backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_balanced_class_weights():
    weights = trainer.balanced_class_weights([0, 0, 0, 1], 2)
    torch.testing.assert_close(weights, torch.tensor([4 / 6, 2.0]))


def test_warmup_cosine_shape():
    factor = trainer.warmup_cosine(10, 110)
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(1.0)
    assert factor(60) == pytest.approx(0.5)
    assert factor(110) == pytest.approx(0.0)
    assert factor(500) == pytest.approx(0.0)        # past the end, no restart


def test_full_run_saves_results_and_touches_test_once(tmp_path, tiny_model_path):
    cfg = make_cfg(tmp_path, tiny_model_path)
    trainer.run(cfg)
    out = tmp_path / "run"
    results = json.loads((out / "results.json").read_text())
    assert results["n_classes"] == N_CLASSES
    assert results["test"]["ci_low"] <= results["test"]["macro_f1"] <= results["test"]["ci_high"]
    assert len(results["history"]) == 2
    assert results["best_epoch"] in {1, 2}
    for name in ["last.pt", "best.pt", "history.json", "per_class.json", "test_predictions.npz"]:
        assert (out / name).exists()
    with np.load(out / "val_predictions.npz") as f:
        assert f["logits"].shape == (len(f["labels"]), N_CLASSES)


def final_weights(out_dir):
    return torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)["model"]


def test_hard_crash_resumes_to_the_same_result(tmp_path, tiny_model_path, monkeypatch):
    """A crash with no chance to save (power cut) loses nothing beyond the last checkpoint."""
    trainer.run(make_cfg(tmp_path, tiny_model_path, name="reference"))
    reference = final_weights(tmp_path / "reference")

    real_forward = EsmLoraClassifier.forward
    calls = {"n": 0}

    def crashing_forward(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 7:
            raise RuntimeError("simulated crash")
        return real_forward(self, *args, **kwargs)

    cfg = make_cfg(tmp_path, tiny_model_path, name="crashed")
    monkeypatch.setattr(EsmLoraClassifier, "forward", crashing_forward)
    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer.run(cfg)
    monkeypatch.setattr(EsmLoraClassifier, "forward", real_forward)

    saved = torch.load(tmp_path / "crashed" / "last.pt", map_location="cpu", weights_only=False)
    assert 0 < saved["progress"]["batches_done"] < 7

    trainer.run(cfg)
    resumed = final_weights(tmp_path / "crashed")
    assert set(resumed) == set(reference)
    for name in reference:
        torch.testing.assert_close(resumed[name], reference[name], atol=1e-5, rtol=1e-4)


def test_ctrl_c_saves_progress_and_the_run_continues(tmp_path, tiny_model_path, monkeypatch):
    # A long checkpoint interval: only the Ctrl+C handler can have saved
    cfg = make_cfg(tmp_path, tiny_model_path, checkpoint_minutes=60)
    real_forward = EsmLoraClassifier.forward
    calls = {"n": 0}

    def interrupted_forward(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 6:
            raise KeyboardInterrupt
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(EsmLoraClassifier, "forward", interrupted_forward)
    with pytest.raises(KeyboardInterrupt):
        trainer.run(cfg)
    monkeypatch.setattr(EsmLoraClassifier, "forward", real_forward)

    saved = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    # Saved on an optimizer-step boundary, with accumulation of 2
    assert saved["progress"]["batches_done"] == 4

    trainer.run(cfg)
    results = json.loads((tmp_path / "run" / "results.json").read_text())
    assert len(results["history"]) == 2


def test_finished_run_only_repeats_the_final_evaluation(tmp_path, tiny_model_path, monkeypatch):
    cfg = make_cfg(tmp_path, tiny_model_path)
    trainer.run(cfg)
    before = final_weights(tmp_path / "run")

    def no_training(*args, **kwargs):
        raise AssertionError("no training step should run")

    monkeypatch.setattr(trainer, "batch_loss", no_training)
    trainer.run(cfg)
    after = final_weights(tmp_path / "run")
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_changed_batching_keeps_weights_and_restarts_the_epoch(
    tmp_path, tiny_model_path, monkeypatch, caplog
):
    real_forward = EsmLoraClassifier.forward
    calls = {"n": 0}

    def crashing_forward(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("simulated crash")
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(EsmLoraClassifier, "forward", crashing_forward)
    with pytest.raises(RuntimeError):
        trainer.run(make_cfg(tmp_path, tiny_model_path))
    monkeypatch.setattr(EsmLoraClassifier, "forward", real_forward)

    trainer.run(make_cfg(tmp_path, tiny_model_path, max_tokens=300))
    assert "Batching settings changed" in caplog.text
    assert (tmp_path / "run" / "results.json").exists()


def test_every_sequence_weighs_the_same_in_the_loss():
    # Summed loss: a sequence in a large batch counts as much as one in a small batch
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    logits = torch.tensor([[2.0, 0.0]] * 4)
    labels = torch.zeros(4, dtype=torch.long)
    one = loss_fn(logits[:1], labels[:1])
    assert loss_fn(logits, labels) == pytest.approx(4 * one.item())


def test_benchmark_reports_and_recommends(tmp_path, tiny_model_path):
    processed = tmp_path / "processed"
    write_splits(processed)
    cfg = load_config([
        f"data.output.processed_dir='{processed.as_posix()}'",
        f"lora.model_name='{tiny_model_path.as_posix()}'",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
        f"training.benchmark.models=['{tiny_model_path.as_posix()}']",
        "training.benchmark.max_tokens=[300,600]",
        "training.benchmark.n_batches=6",
        "training.benchmark.n_longest=2",
        f"training.benchmark.output_dir='{(tmp_path / 'bench').as_posix()}'",
        "training.batch_size=4",
        "training.gradient_checkpointing=false",
    ])
    benchmark.run(cfg)
    report = json.loads((tmp_path / "bench" / "benchmark.json").read_text())
    entry = report["models"][tiny_model_path.as_posix()]
    assert [r["max_tokens"] for r in entry["results"]] == [300, 600]
    assert all(r["status"] == "ok" and r["epoch_hours"] > 0 for r in entry["results"])
    assert entry["recommended_max_tokens"] in {300, 600}
    assert entry["checkpoint"]["weights_restored"] is True
    assert not (tmp_path / "bench" / "benchmark_checkpoint.pt").exists()


def test_recommend_skips_out_of_memory_spill_and_slowdown():
    def ok(tokens, rate, **extra):
        return {
            "max_tokens": tokens, "status": "ok", "train_tokens_per_second": rate,
            "first_half_tokens_per_second": rate, "second_half_tokens_per_second": rate,
            "peak_reserved_gb": 3.0, "total_vram_gb": 8.0, "spill_risk": False, **extra,
        }

    results = [
        ok(2048, 1000),
        ok(4096, 2000),
        ok(8192, 3000, spill_risk=True),
        ok(16384, 4000, second_half_tokens_per_second=100),
        {"max_tokens": 32768, "status": "out of memory"},
    ]
    assert benchmark.recommend(results, 0.8)["max_tokens"] == 4096
    assert benchmark.recommend([{"max_tokens": 1, "status": "out of memory"}], 0.8) is None
