"""Atomic checkpoint save/load with rotation.

A checkpoint carries everything needed to resume bit-for-bit: weights,
optimizer state, step counter, the data loader position, RNG state and the
config the run started from. Writes go to a temp file and are renamed into
place, so an interrupted save can never leave a half-written checkpoint behind.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import Config

__all__ = ["save_checkpoint", "load_checkpoint", "unwrap_model", "latest_checkpoint"]

logger = logging.getLogger("motherbrain")

_STEP_RE = re.compile(r"^ckpt_(\d+)\.pt$")
# torch.compile and DDP each add a prefix to every parameter name; strip both so
# a checkpoint is portable between compiled/uncompiled and single/multi-GPU runs.
_PREFIXES = ("_orig_mod.", "module.")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel off DDP / ``torch.compile`` wrappers to reach the real module."""
    while True:
        inner = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if inner is None or inner is model:
            return model
        model = inner


def _strip_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state.items():
        for prefix in _PREFIXES:
            while key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: Config,
    *,
    best_val_loss: float | None = None,
    loader_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": _strip_prefixes(unwrap_model(model).state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config.to_dict(),
        "best_val_loss": best_val_loss,
        "loader_state": loader_state,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model/optimizer/RNG in place."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=False: our payload holds a config dict and RNG tensors, not
    # just weights. Only load checkpoints you produced or otherwise trust.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        unwrap_model(model).load_state_dict(_strip_prefixes(ckpt["model"]), strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore_rng and ckpt.get("rng"):
        torch.set_rng_state(ckpt["rng"]["torch"].cpu().to(torch.uint8))
        if ckpt["rng"].get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            except (RuntimeError, ValueError) as exc:
                # Resuming on a different GPU count is fine; just note it.
                logger.warning("could not restore CUDA RNG state: %s", exc)
    return ckpt


def config_from_checkpoint(path: str | Path, map_location: str = "cpu") -> Config:
    """Read just the config a checkpoint was trained with."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    return Config.from_dict(ckpt["config"])


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``ckpt_*.pt`` in ``out_dir``, if any."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in out_dir.iterdir():
        match = _STEP_RE.match(entry.name)
        if match:
            step = int(match.group(1))
            if best is None or step > best[0]:
                best = (step, entry)
    return best[1] if best else None


def rotate_checkpoints(out_dir: str | Path, keep_last_n: int) -> None:
    """Delete all but the ``keep_last_n`` most recent step checkpoints."""
    if keep_last_n <= 0:
        return
    out_dir = Path(out_dir)
    found: list[tuple[int, Path]] = []
    for entry in out_dir.iterdir():
        match = _STEP_RE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    for _, stale in sorted(found, reverse=True)[keep_last_n:]:
        try:
            stale.unlink()
        except OSError as exc:  # pragma: no cover - filesystem race
            logger.warning("could not remove old checkpoint %s: %s", stale, exc)
