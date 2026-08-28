"""Measure loss / perplexity of a checkpoint on a tokenized split.

python -m motherbrain.evaluate --ckpt runs/small/best.pt --split val
"""

from __future__ import annotations

import argparse
import math

import torch

from .data import BatchLoader, TokenDataset
from .sample import load_model
from .train import resolve_device, resolve_dtype

__all__ = ["evaluate_split", "main"]


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    data_dir: str,
    split: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """Stream the split once (no shuffling) and return loss/perplexity."""
    dataset = TokenDataset(data_dir, split, seq_len)
    loader = BatchLoader(dataset, batch_size, shuffle=False, device=device)
    batches = loader.batches_per_epoch
    if max_batches is not None:
        batches = min(batches, max_batches)
    if batches == 0:
        raise ValueError(f"split {split!r} has fewer than one batch at batch_size={batch_size}")

    autocast = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if device.type in ("cuda", "cpu") and dtype is not torch.float32
        else torch.autocast(device_type="cpu", enabled=False)
    )

    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = next(loader)
        with autocast:
            _, loss = model(x, targets=y)
        total += loss.item()

    mean = total / batches
    return {
        "loss": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "batches": float(batches),
        "tokens": float(batches * batch_size * seq_len),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motherbrain-eval", description="Evaluate a MotherBrain checkpoint."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default=None, help="defaults to the checkpoint's data_dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    dtype = resolve_dtype(args.dtype, device)

    stats = evaluate_split(
        model,
        args.data_dir or cfg.data.data_dir,
        args.split,
        cfg.model.max_seq_len,
        args.batch_size,
        device,
        max_batches=args.max_batches,
        dtype=dtype,
    )
    print(
        f"split={args.split} loss={stats['loss']:.4f} ppl={stats['perplexity']:.2f} "
        f"({int(stats['batches'])} batches, {int(stats['tokens'])} tokens)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
