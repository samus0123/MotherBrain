"""Training loop: AdamW, warmup + decay schedule, grad accumulation, checkpointing."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, RunConfig, TrainConfig
from .data import TokenDataset, load_splits
from .model import MotherBrain

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = DTYPES.get(name, torch.float32)
    if device.type == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    if device.type == "cpu" and dtype is torch.float16:
        # fp16 autocast on CPU is slow and poorly supported; bf16 is the CPU path.
        return torch.bfloat16
    return dtype


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for `step`, following cfg.schedule."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    min_lr = cfg.lr * cfg.min_lr_ratio
    decay_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)

    if cfg.schedule == "constant":
        return cfg.lr
    if cfg.schedule == "linear":
        return min_lr + (cfg.lr - min_lr) * (1.0 - progress)
    if cfg.schedule == "wsd":
        # Warmup-stable-decay: hold, then decay over the final 20% of the run.
        if progress < 0.8:
            return cfg.lr
        tail = (progress - 0.8) / 0.2
        return min_lr + (cfg.lr - min_lr) * (1.0 - tail)
    # cosine (default)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: MotherBrain, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight-decay every matrix; leave norms, biases and 1-D params undecayed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, fused=fused
    )


@torch.no_grad()
def evaluate(
    model: MotherBrain,
    dataset: TokenDataset,
    cfg: TrainConfig,
    device: torch.device,
    autocast_ctx,
    steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    rng = np.random.default_rng(cfg.seed)  # same eval windows every time
    for _ in range(steps or cfg.eval_steps):
        x, y = dataset.sample_batch(cfg.batch_size, cfg.seq_len, device, generator=rng)
        with autocast_ctx:
            out = model(x, targets=y)
        losses.append(out.lm_loss.detach().item())
    model.train()
    mean = sum(losses) / len(losses)
    return {"loss": mean, "ppl": math.exp(min(mean, 20))}


def save_checkpoint(
    path: Path,
    model: MotherBrain,
    optimizer: torch.optim.Optimizer,
    step: int,
    run: RunConfig,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": run.to_dict(),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[MotherBrain, RunConfig]:
    ckpt = load_checkpoint(path, map_location=device)
    run = RunConfig.from_dict(ckpt["config"])
    model = MotherBrain(run.model)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    return model, run


def train(run: RunConfig) -> Path:
    """Run a full training job. Returns the path to the final checkpoint."""
    cfg, mcfg = run.train, run.model
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, meta = load_splits(cfg.data_dir)
    if cfg.train_split not in datasets:
        raise ValueError(f"split '{cfg.train_split}' not found in {cfg.data_dir}")
    if meta["vocab_size"] != mcfg.vocab_size:
        print(
            f"[warn] dataset vocab_size {meta['vocab_size']} != model vocab_size "
            f"{mcfg.vocab_size}; using the dataset's."
        )
        mcfg.vocab_size = meta["vocab_size"]
    train_ds = datasets[cfg.train_split]
    val_ds = datasets.get(cfg.val_split)

    model = MotherBrain(mcfg).to(device)
    if cfg.grad_checkpoint:
        model.set_grad_checkpointing(True)
    total, active = model.num_parameters(), model.num_active_parameters()
    print(f"model: {total:,} parameters ({active:,} active per token)")
    print(f"data:  {len(train_ds):,} train tokens on {device} in {dtype}")

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(enabled=(dtype is torch.float16))
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype is not torch.float32
        else torch.autocast(device_type=device.type, enabled=False)
    )

    start_step, best_val = 0, float("inf")
    resume_path = out_dir / "latest.pt" if cfg.resume == "auto" else Path(cfg.resume or "")
    if cfg.resume and resume_path.exists():
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {resume_path} at step {start_step}")

    if cfg.compile:
        model = torch.compile(model)

    run.save(out_dir / "config.json")
    log_path = out_dir / "log.jsonl"
    rng = np.random.default_rng(cfg.seed + start_step)
    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps

    model.train()
    t0 = time.time()
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_ds.sample_batch(cfg.batch_size, cfg.seq_len, device, generator=rng)
            with autocast_ctx:
                out = model(x, targets=y)
                loss = out.loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += out.lm_loss.detach().item() / cfg.grad_accum_steps

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            elapsed = time.time() - t0
            tok_per_s = tokens_per_step * (step - start_step + 1) / max(elapsed, 1e-6)
            record = {
                "step": step,
                "loss": round(accum_loss, 5),
                "ppl": round(math.exp(min(accum_loss, 20)), 3),
                "lr": round(lr, 8),
                "grad_norm": round(float(grad_norm), 4),
                "tok_per_s": round(tok_per_s, 1),
            }
            if out.aux_loss is not None:
                record["aux_loss"] = round(out.aux_loss.detach().item(), 5)
            print(
                f"step {step:>6} | loss {record['loss']:.4f} | ppl {record['ppl']:>9.2f} "
                f"| lr {lr:.2e} | {record['tok_per_s']:.0f} tok/s"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if val_ds is not None and cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            metrics = evaluate(model, val_ds, cfg, device, autocast_ctx)
            print(f"  eval @ {step}: loss {metrics['loss']:.4f} | ppl {metrics['ppl']:.2f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, step, run, best_val)

        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step, run, best_val)

    final = out_dir / "final.pt"
    save_checkpoint(final, model, optimizer, cfg.max_steps, run, best_val)
    save_checkpoint(out_dir / "latest.pt", model, optimizer, cfg.max_steps, run, best_val)
    print(f"done in {time.time() - t0:.1f}s -> {final}")
    return final
