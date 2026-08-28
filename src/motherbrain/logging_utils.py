"""Rank-aware console logging plus a JSONL metric sink (and optional W&B)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["setup_logging", "MetricLogger", "format_count", "format_duration"]


def setup_logging(rank: int = 0, level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger; non-zero ranks are quieted to warnings."""
    logger = logging.getLogger("motherbrain")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt=f"[%(asctime)s][rank{rank}] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level if rank == 0 else logging.WARNING)
    logger.propagate = False
    return logger


def format_count(n: float) -> str:
    """Human-readable magnitude: 1234567 -> '1.23M'."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return f"{n:.0f}"


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class MetricLogger:
    """Append metrics to ``metrics.jsonl`` and mirror them to W&B if configured.

    Only the master rank writes anything; other ranks get a silent no-op object
    so callers do not need rank checks at every log site.
    """

    def __init__(
        self,
        out_dir: str | Path,
        is_master: bool = True,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.is_master = is_master
        self.start_time = time.time()
        self._wandb = None
        self._fh = None

        if not is_master:
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")

        if config is not None:
            with (out_dir / "config.json").open("w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2, default=str)

        if wandb_project:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            except ImportError:
                logging.getLogger("motherbrain").warning(
                    "wandb_project is set but wandb is not installed; skipping"
                )

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.is_master or self._fh is None:
            return
        record = {"step": step, "wall_time": round(time.time() - self.start_time, 3), **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
