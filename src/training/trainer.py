"""
src/training/trainer.py

Trains the ESM-2 + LoRA classifier (experiment E3/E4), built for long runs on
one 8 GB GPU.

What this script does:
    1. Trains on train with mixed precision, gradient accumulation and
       class-weighted cross-entropy; early stopping on validation macro-F1
    2. Saves progress regularly and resumes from it automatically
    3. At the end, loads the best epoch and evaluates once on test: macro-F1
       with a cluster-bootstrap interval, plus saved logits for calibration
       and paired comparisons

Saving and resuming:
    last.pt     Latest progress: trainable weights, optimizer, scheduler,
                scaler, random states, position within the epoch. Written every
                checkpoint_minutes, at each epoch end, and on Ctrl+C. Each write
                goes to a temp file first, so a crash never corrupts it.
    best.pt     Trainable weights of the best epoch by validation macro-F1.
    history.json  One entry per epoch, readable while the run is going.
    Rerun the same command to continue from last.pt. A finished run only
    repeats the final evaluation.

Why not divide the loss by the batch size?
    Batches are limited by tokens, so a batch of short sequences holds many
    more rows than a batch of long ones. Averaging within each batch would make
    one long protein count several times as much as one short protein. Instead
    the summed loss is divided by a constant (average rows per batch x
    accumulation steps), so every sequence weighs the same.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from transformers import EsmTokenizer

from src import metrics
from src.config import cli_config, config_as_dict
from src.data.dataset import EnzymeDataset, LengthGroupedBatchSampler, PaddingCollator
from src.model.esm2_lora import EsmLoraClassifier, build_model
from src.tracking import git_state, log_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def balanced_class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """n / (k * count), as in the linear probe: rare classes weigh as much as common ones."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def warmup_cosine(warmup_steps: int, total_steps: int):
    """Learning-rate factor: linear warmup, then cosine decay to zero."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
    return factor


def guard_memory(device: torch.device, max_fraction: float) -> None:
    """Empty the CUDA cache before Windows spills GPU memory into system RAM."""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        if torch.cuda.memory_reserved(device) > max_fraction * total:
            torch.cuda.empty_cache()


def batch_loss(
    model: EsmLoraClassifier,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Summed loss over the batch's sequences, under mixed precision."""
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        return loss_fn(logits, batch["label"].to(device))


@torch.inference_mode()
def predict_logits(
    model: EsmLoraClassifier,
    dataset: EnzymeDataset,
    pad_token_id: int,
    batch_size: int,
    max_tokens: int,
    n_classes: int,
    device: torch.device,
    use_amp: bool,
    max_memory_fraction: float = 0.8,
) -> np.ndarray:
    """Logits for every record, in the dataset's record order."""
    was_training = model.training
    model.eval()
    # Longest first: the largest memory blocks are reserved at the start
    batches = list(LengthGroupedBatchSampler(
        dataset.token_lengths(), batch_size=batch_size, max_tokens=max_tokens, shuffle=False
    ))[::-1]
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batches, collate_fn=PaddingCollator(pad_token_id)
    )
    logits = np.zeros((len(dataset), n_classes), dtype=np.float32)
    for batch in tqdm(loader, desc="Evaluating", unit="batch", leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        logits[batch["index"].numpy()] = out.float().cpu().numpy()
        guard_memory(device, max_memory_fraction)
    model.train(was_training)
    return logits


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_atomically(payload: dict, path: Path) -> None:
    """Write to a temp file, then rename, so a crash never leaves a broken file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, progress: dict) -> None:
    save_atomically({
        "model": model.trainable_state(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": progress,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler) -> dict:
    # weights_only=False: the file holds plain progress dicts next to tensors,
    # and it is one this run wrote itself
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_trainable_state(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and ckpt["rng_cuda"]:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    return ckpt["progress"]


def batching_settings(training: DictConfig) -> dict:
    return {
        "batch_size": training.batch_size,
        "max_tokens": training.max_tokens,
        "max_batches_per_epoch": training.max_batches_per_epoch,
        "grad_accum_steps": training.grad_accum_steps,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: DictConfig, model, train_set, val_set, pad_id, labels, output_dir, device) -> None:
    training = cfg.training
    use_amp = bool(training.fp16 and device.type == "cuda")
    n_classes = len(labels)

    sampler = LengthGroupedBatchSampler(
        train_set.token_lengths(), batch_size=training.batch_size,
        max_tokens=training.max_tokens, shuffle=True, seed=training.seed,
    )
    n_batches = len(sampler)
    if training.max_batches_per_epoch:
        n_batches = min(n_batches, training.max_batches_per_epoch)
    accum = training.grad_accum_steps
    steps_per_epoch = int(np.ceil(n_batches / accum))
    total_steps = steps_per_epoch * training.epochs
    # Every sequence weighs the same, whatever the size of its batch
    rows_per_batch = len(train_set) / len(sampler)
    normaliser = rows_per_batch * accum

    weights = None
    if training.balanced_classes:
        weights = balanced_class_weights(
            [r["label"] for r in train_set.records], n_classes
        ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="sum")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(int(training.warmup_fraction * total_steps), total_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    history_path = output_dir / "history.json"
    progress = {
        "epoch": 0, "batches_done": 0, "opt_steps": 0, "best_val": -1.0,
        "epochs_without_improvement": 0, "finished": False,
        "history": [], "batching": batching_settings(training),
    }
    if last_path.exists():
        progress = load_checkpoint(last_path, model, optimizer, scheduler, scaler)
        logger.info(
            f"Resuming from {last_path}: epoch {progress['epoch'] + 1}, "
            f"{progress['batches_done']} batches done, {progress['opt_steps']} optimizer steps"
        )
        if progress["batching"] != batching_settings(training):
            # The batch list no longer matches the saved position within the epoch
            logger.warning(
                "Batching settings changed since the checkpoint: keeping the weights "
                "and restarting the current epoch's batches"
            )
            progress["batches_done"] = 0
            progress["batching"] = batching_settings(training)

    if progress["finished"]:
        logger.info("Training already finished, going to the final evaluation")
        return

    def save_progress() -> None:
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, progress)
        with open(history_path, "w") as f:
            json.dump(progress["history"], f, indent=2)

    last_save = time.time()
    model.train()
    try:
        for epoch in range(progress["epoch"], training.epochs):
            progress["epoch"] = epoch
            sampler.set_epoch(epoch)
            batches = list(sampler)[:n_batches]
            todo = batches[progress["batches_done"]:]
            loader = torch.utils.data.DataLoader(
                train_set, batch_sampler=todo, collate_fn=PaddingCollator(pad_id)
            )
            bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{training.epochs}", unit="batch",
                       initial=progress["batches_done"], total=n_batches)

            epoch_start = time.time()
            window_tokens, window_start = 0, time.time()
            running_loss, running_rows = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for position, batch in enumerate(bar, start=progress["batches_done"] + 1):
                loss = batch_loss(model, batch, loss_fn, device, use_amp)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss became {loss.item()} at epoch {epoch + 1}")
                scaler.scale(loss / normaliser).backward()

                running_loss += loss.item()
                running_rows += len(batch["label"])
                window_tokens += int(batch["attention_mask"].numel())

                if position % accum == 0 or position == len(batches):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    guard_memory(device, training.max_memory_fraction)
                    # A checkpoint only ever sits on an optimizer-step boundary,
                    # so no half-accumulated gradient is lost
                    progress["batches_done"] = position
                    progress["opt_steps"] += 1

                    if time.time() - last_save >= training.checkpoint_minutes * 60:
                        save_progress()
                        last_save = time.time()

                    if progress["opt_steps"] % 10 == 0:
                        postfix = {
                            "loss": f"{running_loss / max(1, running_rows):.3f}",
                            "tok/s": f"{window_tokens / (time.time() - window_start):,.0f}",
                        }
                        if device.type == "cuda":
                            gb = torch.cuda.memory_reserved(device) / 2**30
                            postfix["gpu_GB"] = f"{gb:.1f}"
                        bar.set_postfix(postfix)
                        window_tokens, window_start = 0, time.time()

            # End of epoch: validate, keep the best weights
            val_logits = predict_logits(
                model, val_set, pad_id, training.batch_size, training.eval_max_tokens,
                n_classes, device, use_amp, training.max_memory_fraction,
            )
            y_val = np.array([r["label"] for r in val_set.records])
            val_f1 = metrics.macro_f1(y_val, val_logits.argmax(axis=1), labels)
            entry = {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(1, running_rows),
                "val_macro_f1": val_f1,
                "opt_steps": progress["opt_steps"],
                "epoch_minutes": (time.time() - epoch_start) / 60,
            }
            progress["history"].append(entry)
            logger.info(f"Epoch {epoch + 1}: val macro-F1 {val_f1:.4f}, {entry}")

            if val_f1 > progress["best_val"]:
                progress["best_val"] = val_f1
                progress["epochs_without_improvement"] = 0
                save_atomically({"model": model.trainable_state(), "epoch": epoch + 1}, best_path)
            else:
                progress["epochs_without_improvement"] += 1

            progress["epoch"] = epoch + 1
            progress["batches_done"] = 0
            stop = progress["epochs_without_improvement"] > training.patience
            progress["finished"] = stop or epoch + 1 == training.epochs
            save_progress()
            last_save = time.time()
            if stop:
                logger.info(f"Early stopping after epoch {epoch + 1}")
                break
    except KeyboardInterrupt:
        logger.warning("Interrupted: saving progress. Rerun the same command to continue.")
        save_progress()
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    lora, training = cfg.lora, cfg.training
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(training.seed)

    device = get_device()
    use_amp = bool(training.fp16 and device.type == "cuda")
    logger.info(f"Model: {lora.model_name} on {device}, mixed precision: {use_amp}")
    if device.type == "cpu":
        logger.warning("No GPU found: training will be very slow")

    with open(processed_dir / "label_map.json") as f:
        index_to_subclass = {v: k for k, v in json.load(f).items()}
    labels = sorted(index_to_subclass)

    tokenizer = EsmTokenizer.from_pretrained(lora.model_name)
    pad_id = tokenizer.pad_token_id
    train_set = EnzymeDataset(processed_dir / "train.json", tokenizer)
    val_set = EnzymeDataset(processed_dir / "val.json", tokenizer)
    test_set = EnzymeDataset(processed_dir / "test.json", tokenizer)

    model = build_model(
        lora.model_name, len(labels), lora.r, lora.alpha, lora.dropout, lora.target_modules,
        lora.head_dropout, device,
        base_fp16=bool(training.base_fp16 and device.type == "cuda"),
        gradient_checkpointing=training.gradient_checkpointing,
    )
    logger.info(f"Trainable parameters: {model.n_trainable():,}")

    train(cfg, model, train_set, val_set, pad_id, labels, output_dir, device)

    # Final evaluation with the best epoch: the only time test is touched
    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_trainable_state(best["model"])
    arrays = {}
    for name, dataset in [("val", val_set), ("test", test_set)]:
        logits = predict_logits(
            model, dataset, pad_id, training.batch_size, training.eval_max_tokens,
            len(labels), device, use_amp, training.max_memory_fraction,
        )
        np.savez(
            output_dir / f"{name}_predictions.npz",
            uniprot_ids=np.array([r["uniprot_id"] for r in dataset.records]),
            labels=np.array([r["label"] for r in dataset.records]),
            clusters=np.array([r["cluster_id"] for r in dataset.records]),
            predictions=logits.argmax(axis=1),
            logits=logits,
        )
        arrays[name] = logits

    y_test = np.array([r["label"] for r in test_set.records])
    clusters = np.array([r["cluster_id"] for r in test_set.records])
    y_pred = arrays["test"].argmax(axis=1)
    summary = metrics.macro_f1_with_ci(
        y_test, y_pred, clusters, labels, training.n_bootstrap, training.seed
    )
    summary.update({
        "accuracy": float((y_pred == y_test).mean()),
        "top3_accuracy": metrics.top_k_accuracy(y_test, arrays["test"], 3),
        **metrics.hierarchical_metrics(y_test, y_pred, index_to_subclass),
    })

    history_path = output_dir / "history.json"
    commit, dirty = git_state()
    results = {
        "experiment": f"LoRA r={lora.r} on {lora.model_name}",
        "git_commit": commit,
        "dirty": dirty,
        "n_classes": len(labels),
        "n_trainable_parameters": model.n_trainable(),
        "best_epoch": best["epoch"],
        "test": summary,
        "history": json.loads(history_path.read_text()),
        "config": config_as_dict(cfg),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    per_class_path = output_dir / "per_class.json"
    with open(per_class_path, "w") as f:
        json.dump(metrics.per_class_table(y_test, y_pred, clusters, index_to_subclass), f, indent=2)

    logger.info(
        f"Test macro-F1 {summary['macro_f1']:.4f} "
        f"[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}], saved to {results_path}"
    )
    log_run(
        cfg, "decodec-lora", output_dir.name, summary,
        artifacts=[results_path, per_class_path, history_path],
    )


if __name__ == "__main__":
    run(cli_config("Train ESM-2 + LoRA (rerun the same command to resume)"))
