"""The training loop.

Single process or ``torchrun``-launched DDP, mixed precision, gradient
accumulation, cosine LR decay, periodic evaluation and resumable checkpoints.

    python -m motherbrain.train --config configs/small_124m.yaml
    torchrun --nproc_per_node=8 -m motherbrain.train --config configs/small_124m.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from .config import Config, load_config
from .data import BatchLoader, ShardIndex, TokenDataset
from .distributed import all_reduce_mean, cleanup_distributed, init_distributed
from .logging_utils import MetricLogger, format_count, format_duration, setup_logging
from .model import Transformer
from .optim import build_optimizer, get_lr

__all__ = ["train", "main", "resolve_device", "resolve_dtype"]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Pick the autocast dtype, falling back when the hardware cannot do bf16."""
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        requested
    ]
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return dtype


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: BatchLoader,
    steps: int,
    autocast: contextlib.AbstractContextManager,
) -> float:
    """Mean loss over ``steps`` batches. Restores training mode on the way out."""
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = next(loader)
        with autocast:
            _, loss = model(x, targets=y)
        total += loss.item()
    model.train(was_training)
    return total / max(1, steps)


def train(cfg: Config, resume: str | None = None) -> dict[str, float]:
    """Run training to completion. Returns a small summary dict."""
    dist_info = init_distributed()
    logger = setup_logging(dist_info.rank)

    device = resolve_device(cfg.train.device)
    if dist_info.enabled and device.type == "cuda":
        device = torch.device(f"cuda:{dist_info.local_rank}")
    dtype = resolve_dtype(cfg.train.dtype, device)

    # Offset the seed per rank so dropout and sampling differ, but keep the data
    # order tied to the shared base seed inside BatchLoader.
    torch.manual_seed(cfg.train.seed + dist_info.rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.train.seed + dist_info.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.train.out_dir)
    if dist_info.is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------------
    index = ShardIndex.load(cfg.data.data_dir)
    if index.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"config vocab_size={cfg.model.vocab_size} does not match the tokenized "
            f"data (vocab_size={index.vocab_size} in {cfg.data.data_dir}/meta.json)"
        )

    seq_len = cfg.model.max_seq_len
    train_ds = TokenDataset(cfg.data.data_dir, cfg.data.train_split, seq_len)
    train_loader = BatchLoader(
        train_ds,
        cfg.train.batch_size,
        shuffle=cfg.data.shuffle,
        seed=cfg.train.seed,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
        device=device,
    )
    val_loader = None
    if cfg.data.val_split in index.splits:
        val_ds = TokenDataset(cfg.data.data_dir, cfg.data.val_split, seq_len)
        val_loader = BatchLoader(
            val_ds,
            cfg.train.batch_size,
            shuffle=False,
            seed=cfg.train.seed,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            device=device,
        )
    else:
        logger.warning("no %r split found; skipping evaluation", cfg.data.val_split)

    # ---- model -----------------------------------------------------------------
    model = Transformer(cfg.model).to(device)
    raw_model = model
    optimizer = build_optimizer(model, cfg.optim, device.type)

    start_step = 0
    best_val_loss = float("inf")
    resume_path = resume
    if resume_path == "auto":
        found = latest_checkpoint(out_dir)
        resume_path = str(found) if found else None
        if found:
            logger.info("auto-resuming from %s", found)
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=model, optimizer=optimizer, map_location=device)
        start_step = int(ckpt.get("step", 0))
        if ckpt.get("best_val_loss") is not None:
            best_val_loss = float(ckpt["best_val_loss"])
        if ckpt.get("loader_state"):
            train_loader.load_state_dict(ckpt["loader_state"])
        logger.info("resumed at step %d", start_step)

    if cfg.train.compile:
        logger.info("compiling model (first step will be slow)...")
        model = torch.compile(model)
    if dist_info.enabled:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
        )

    # float16 needs loss scaling to keep small gradients from flushing to zero;
    # bfloat16 and float32 do not.
    use_scaler = dtype is torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    autocast: contextlib.AbstractContextManager
    if device.type in ("cuda", "cpu") and dtype is not torch.float32:
        autocast = torch.autocast(device_type=device.type, dtype=dtype)
    else:
        autocast = contextlib.nullcontext()

    tokens_per_step = (
        cfg.train.batch_size * cfg.train.grad_accum_steps * seq_len * dist_info.world_size
    )
    if dist_info.is_master:
        logger.info(
            "model: %s non-embedding params | %s total | %d layers, d_model=%d, %d heads (%d kv)",
            format_count(raw_model.num_params()),
            format_count(raw_model.num_params(non_embedding=False)),
            cfg.model.n_layer,
            cfg.model.d_model,
            cfg.model.n_head,
            cfg.model.n_kv_head or cfg.model.n_head,
        )
        logger.info(
            "device=%s dtype=%s world_size=%d | %s tokens/step | %s tokens total | "
            "train split: %s tokens, %s windows",
            device,
            str(dtype).replace("torch.", ""),
            dist_info.world_size,
            format_count(tokens_per_step),
            format_count(tokens_per_step * cfg.train.max_steps),
            format_count(index.tokens_in(cfg.data.train_split)),
            format_count(train_ds.num_windows),
        )

    metrics = MetricLogger(
        out_dir,
        is_master=dist_info.is_master,
        wandb_project=cfg.train.wandb_project,
        wandb_run_name=cfg.train.wandb_run_name,
        config=cfg.to_dict(),
    )

    # ---- loop ------------------------------------------------------------------
    model.train()
    run_start = time.time()
    step_start = time.time()
    last_loss = float("nan")

    try:
        if val_loader is not None and cfg.train.eval_at_start and start_step == 0:
            val_loss = all_reduce_mean(
                evaluate(model, val_loader, cfg.train.eval_steps, autocast), dist_info, device
            )
            logger.info("step 0 | initial val loss %.4f", val_loss)
            metrics.log(0, {"val/loss": val_loss})

        for step in range(start_step, cfg.train.max_steps):
            lr = get_lr(step, cfg.optim, cfg.train.max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for micro in range(cfg.train.grad_accum_steps):
                x, y = next(train_loader)
                # Only sync gradients on the last micro-batch of the accumulation.
                sync = (micro == cfg.train.grad_accum_steps - 1) or not dist_info.enabled
                ctx = contextlib.nullcontext() if sync else model.no_sync()
                with ctx, autocast:
                    _, loss = model(x, targets=y)
                    loss = loss / cfg.train.grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            grad_norm = float("nan")
            if cfg.optim.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()
            last_loss = accum_loss

            is_last = step == cfg.train.max_steps - 1

            if (step % cfg.train.log_interval == 0 or is_last) and dist_info.is_master:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - step_start
                steps_done = cfg.train.log_interval if step > start_step else 1
                per_step = elapsed / max(1, steps_done)
                tok_per_sec = tokens_per_step / per_step if per_step > 0 else 0.0
                remaining = (cfg.train.max_steps - step - 1) * per_step
                record = {
                    "train/loss": accum_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/step_time_s": per_step,
                    "progress/tokens": tokens_per_step * (step + 1),
                }
                if device.type == "cuda":
                    flops = raw_model.flops_per_token() * tok_per_sec
                    peak = torch.cuda.get_device_properties(device).total_memory
                    record["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    record["perf/tflops"] = flops / 1e12
                    del peak
                metrics.log(step, record)
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | grad %.2f | %s tok/s | eta %s",
                    step,
                    cfg.train.max_steps,
                    accum_loss,
                    lr,
                    grad_norm,
                    format_count(tok_per_sec),
                    format_duration(remaining),
                )
                step_start = time.time()

            should_eval = val_loader is not None and (
                (step + 1) % cfg.train.eval_interval == 0 or is_last
            )
            if should_eval:
                val_loss = all_reduce_mean(
                    evaluate(model, val_loader, cfg.train.eval_steps, autocast), dist_info, device
                )
                if dist_info.is_master:
                    ppl = math.exp(min(val_loss, 20.0))
                    logger.info("step %d | val loss %.4f | ppl %.2f", step, val_loss, ppl)
                    metrics.log(step, {"val/loss": val_loss, "val/ppl": ppl})
                    if cfg.train.save_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            out_dir / "best.pt",
                            model,
                            optimizer,
                            step + 1,
                            cfg,
                            best_val_loss=best_val_loss,
                            loader_state=train_loader.state_dict(),
                        )
                        logger.info("new best val loss %.4f -> best.pt", best_val_loss)
                if dist_info.enabled:
                    best_val_loss = min(best_val_loss, val_loss)
                step_start = time.time()

            should_ckpt = (step + 1) % cfg.train.ckpt_interval == 0 or is_last
            if should_ckpt and dist_info.is_master:
                path = save_checkpoint(
                    out_dir / f"ckpt_{step + 1:07d}.pt",
                    model,
                    optimizer,
                    step + 1,
                    cfg,
                    best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
                    loader_state=train_loader.state_dict(),
                )
                rotate_checkpoints(out_dir, cfg.train.keep_last_n)
                logger.info("saved %s", path.name)
                step_start = time.time()
    finally:
        metrics.close()
        cleanup_distributed(dist_info)

    if dist_info.is_master:
        logger.info(
            "done in %s | final train loss %.4f | best val loss %s",
            format_duration(time.time() - run_start),
            last_loss,
            f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "n/a",
        )
    return {
        "final_train_loss": last_loss,
        "best_val_loss": best_val_loss,
        "steps": float(cfg.train.max_steps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motherbrain-train", description="Train a MotherBrain language model."
    )
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint to resume from, or 'auto' for the latest in out_dir",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="config overrides as section.key=value (e.g. train.max_steps=1000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    train(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
