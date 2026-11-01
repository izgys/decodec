"""
src/training/benchmark.py

Hardware test for LoRA training: run it before any long run (T3.4). It takes
a few minutes per configuration and trains nothing worth keeping.

What this script does, for each model and token budget per batch:
    1. Runs the longest batches first, the worst case for GPU memory
    2. Runs randomly chosen real training batches and times them, with the
       same mixed precision, accumulation and loss as the real run
    3. Times inference, for the validation and test passes
    4. Reports peak GPU memory, throughput and the estimated run time
    5. Saves and reloads a checkpoint, and checks the weights come back intact

What to look for:
    - "out of memory" or "spill risk": that budget is too large for this GPU
    - slowdown: the second half of the timed batches running well below the
      first half, the sign of GPU memory spilling into system RAM
"""

import json
import logging
import random
import subprocess
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from transformers import EsmTokenizer

from src.config import cli_config
from src.data.dataset import EnzymeDataset, LengthGroupedBatchSampler, PaddingCollator
from src.model.esm2_lora import build_model
from src.tracking import log_run
from src.training.trainer import (
    batch_loss,
    get_device,
    guard_memory,
    load_checkpoint,
    save_checkpoint,
    warmup_cosine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

SPILL_FRACTION = 0.9        # dedicated VRAM use above this risks system-RAM spill


def vram_used_mib() -> int | None:
    """Dedicated VRAM in use by everything on the GPU, including other programs."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def padded_tokens(batches: list[list[int]], lengths: list[int]) -> int:
    return sum(len(b) * max(lengths[i] for i in b) for b in batches)


def run_config(
    model, train_set, pad_id: int, max_tokens: int, bench: DictConfig, training: DictConfig,
    device: torch.device, use_amp: bool, n_classes: int,
) -> dict:
    """One token budget on one model. Raises torch.cuda.OutOfMemoryError if it does not fit."""
    lengths = train_set.token_lengths()
    train_batches = list(LengthGroupedBatchSampler(
        lengths, batch_size=training.batch_size, max_tokens=max_tokens, shuffle=True,
        seed=training.seed,
    ))
    longest = sorted(train_batches, key=lambda b: max(lengths[i] for i in b), reverse=True)
    rng = random.Random(training.seed)
    timed = rng.sample(train_batches, min(bench.n_batches, len(train_batches)))
    collator = PaddingCollator(pad_id)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=training.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    accum = training.grad_accum_steps
    rows_per_batch = len(train_set) / len(train_batches)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def train_on(batch_list, desc):
        """Real training steps; returns the seconds each batch took."""
        seconds = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, indices in enumerate(tqdm(batch_list, desc=desc, leave=False), start=1):
            start = time.time()
            batch = collator([train_set[i] for i in indices])
            loss = batch_loss(model, batch, loss_fn, device, use_amp)
            scaler.scale(loss / (rows_per_batch * accum)).backward()
            if step % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                guard_memory(device, training.max_memory_fraction)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.time() - start)
        return seconds

    train_on(longest[:bench.n_longest], "Longest batches")
    seconds = train_on(timed, "Timed batches")

    # Skip the first batches, which include allocation and warm-up
    skip = min(2, len(timed) - 1)
    tokens = [len(b) * max(lengths[i] for i in b) for b in timed]
    rate = sum(tokens[skip:]) / sum(seconds[skip:])
    middle = skip + (len(timed) - skip) // 2
    first_half = sum(tokens[skip:middle]) / max(sum(seconds[skip:middle]), 1e-9)
    second_half = sum(tokens[middle:]) / max(sum(seconds[middle:]), 1e-9)

    # Inference speed, for the validation and test passes
    model.eval()
    eval_tokens, start = 0, time.time()
    with torch.inference_mode():
        for indices in timed[: max(3, bench.n_batches // 3)]:
            batch = collator([train_set[i] for i in indices])
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            eval_tokens += int(batch["attention_mask"].numel())
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_rate = eval_tokens / (time.time() - start)
    model.train()

    epoch_hours = padded_tokens(train_batches, lengths) / rate / 3600
    result = {
        "max_tokens": max_tokens,
        "status": "ok",
        "n_batches_per_epoch": len(train_batches),
        "train_tokens_per_second": rate,
        "first_half_tokens_per_second": first_half,
        "second_half_tokens_per_second": second_half,
        "eval_tokens_per_second": eval_rate,
        "epoch_hours": epoch_hours,
    }
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        result.update({
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
            "total_vram_gb": total / 2**30,
        })
        used = vram_used_mib()
        result["vram_used_gb_all_programs"] = None if used is None else used / 1024
        result["spill_risk"] = bool(
            torch.cuda.max_memory_reserved(device) > SPILL_FRACTION * total
            or (used is not None and used * 2**20 > SPILL_FRACTION * total)
        )
    return result


def check_checkpoint(model, output_dir: Path, training: DictConfig, device, use_amp) -> dict:
    """Save a real checkpoint, disturb the weights, reload, and compare."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=training.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(10, 100))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    path = output_dir / "benchmark_checkpoint.pt"

    before = model.trainable_state()
    start = time.time()
    save_checkpoint(path, model, optimizer, scheduler, scaler, {"test": True})
    save_seconds = time.time() - start
    size_mb = path.stat().st_size / 2**20
    with torch.no_grad():
        for p in trainable:
            p.add_(1.0)
    start = time.time()
    load_checkpoint(path, model, optimizer, scheduler, scaler)
    load_seconds = time.time() - start
    restored = all(torch.equal(before[k], v) for k, v in model.trainable_state().items())
    path.unlink()
    return {
        "size_mb": size_mb,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "weights_restored": restored,
    }


def recommend(results: list[dict], max_fraction: float) -> dict | None:
    """The fastest budget that fits with margin and shows no slowdown."""
    good = []
    for r in results:
        if r["status"] != "ok" or r.get("spill_risk"):
            continue
        if r.get("peak_reserved_gb", 0) > max_fraction * r.get("total_vram_gb", float("inf")):
            continue
        if r["second_half_tokens_per_second"] < 0.5 * r["first_half_tokens_per_second"]:
            continue
        good.append(r)
    return max(good, key=lambda r: r["train_tokens_per_second"], default=None)


def run(cfg: DictConfig) -> None:
    lora, training, bench = cfg.lora, cfg.training, cfg.training.benchmark
    processed_dir = Path(cfg.data.output.processed_dir)
    output_dir = Path(bench.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    use_amp = bool(training.fp16 and device.type == "cuda")
    with open(processed_dir / "label_map.json") as f:
        n_classes = len(json.load(f))
    logger.info(f"Device: {device}, VRAM in use before starting: {vram_used_mib()} MiB")

    report = {"device": str(device), "models": {}}
    for model_name in bench.models:
        logger.info(f"Model: {model_name} (first use downloads it from Hugging Face)")
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        train_set = EnzymeDataset(processed_dir / "train.json", tokenizer)
        model = build_model(
            model_name, n_classes, lora.r, lora.alpha, lora.dropout, lora.target_modules,
            lora.head_dropout, device,
            base_fp16=bool(training.base_fp16 and device.type == "cuda"),
            gradient_checkpointing=training.gradient_checkpointing,
        )
        results = []
        for max_tokens in bench.max_tokens:
            logger.info(f"Token budget per batch: {max_tokens}")
            try:
                result = run_config(
                    model, train_set, tokenizer.pad_token_id, max_tokens, bench, training,
                    device, use_amp, n_classes,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Out of memory at {max_tokens} tokens per batch")
                results.append({"max_tokens": max_tokens, "status": "out of memory"})
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break               # a larger budget cannot fit either
            results.append(result)
            logger.info(json.dumps(result))

        best = recommend(results, training.max_memory_fraction)
        report["models"][model_name] = {
            "n_trainable_parameters": model.n_trainable(),
            "checkpoint": check_checkpoint(model, output_dir, training, device, use_amp),
            "results": results,
            "recommended_max_tokens": None if best is None else best["max_tokens"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report_path = output_dir / "benchmark.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'model':<34}{'tokens':>8}{'tok/s':>9}{'peak GB':>9}{'epoch h':>9}  note")
    for name, entry in report["models"].items():
        for r in entry["results"]:
            if r["status"] != "ok":
                print(f"{name:<34}{r['max_tokens']:>8}{'':>9}{'':>9}{'':>9}  {r['status']}")
                continue
            note = "spill risk" if r.get("spill_risk") else ""
            if r["max_tokens"] == entry["recommended_max_tokens"]:
                note = "recommended"
            print(
                f"{name:<34}{r['max_tokens']:>8}{r['train_tokens_per_second']:>9,.0f}"
                f"{r.get('peak_reserved_gb', 0):>9.1f}{r['epoch_hours']:>9.1f}  {note}"
            )
        ckpt = entry["checkpoint"]
        print(f"  trainable parameters: {entry['n_trainable_parameters']:,}; checkpoint "
              f"{ckpt['size_mb']:.0f} MB, saved in {ckpt['save_seconds']:.1f} s, "
              f"restored correctly: {ckpt['weights_restored']}")
    print(f"\nFull report: {report_path}")

    log_run(cfg, "decodec-lora", "benchmark", {}, artifacts=[report_path])


if __name__ == "__main__":
    run(cli_config("Hardware test for LoRA training: memory, speed and estimated run time"))
