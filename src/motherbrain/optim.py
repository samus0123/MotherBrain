"""Optimizer construction and learning-rate scheduling."""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from .config import OptimConfig

__all__ = ["build_optimizer", "get_lr", "param_groups"]


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed (matrices) and undecayed (norms, biases).

    Weight decay on 1-D parameters -- RMSNorm gains and any biases -- shrinks
    scale parameters toward zero for no benefit, so they get their own group.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue  # tied weights appear twice; count them once
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module, cfg: OptimConfig, device_type: str = "cpu"
) -> torch.optim.AdamW:
    """Build AdamW, using the fused CUDA kernel when it is available."""
    groups = param_groups(model, cfg.weight_decay)
    kwargs: dict = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": cfg.eps,
    }
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if cfg.fused and supports_fused and device_type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def get_lr(step: int, cfg: OptimConfig, max_steps: int) -> float:
    """Learning rate at ``step`` (0-indexed): linear warmup, then decay to min_lr."""
    min_lr = cfg.lr * cfg.min_lr_ratio
    decay_steps = cfg.decay_steps if cfg.decay_steps is not None else max_steps

    if step < cfg.warmup_steps:
        # +1 so step 0 gets a non-zero LR rather than a wasted update.
        return cfg.lr * (step + 1) / (cfg.warmup_steps + 1)
    if cfg.schedule == "constant":
        return cfg.lr
    if step >= decay_steps:
        return min_lr

    span = max(1, decay_steps - cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / span
    if cfg.schedule == "linear":
        factor = 1.0 - progress
    else:  # cosine
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + factor * (cfg.lr - min_lr)
