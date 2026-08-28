"""Generate text from a trained checkpoint.

python -m motherbrain.sample --ckpt runs/small/best.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import Config
from .model import Transformer
from .tokenizer import Tokenizer
from .train import resolve_device

__all__ = ["load_model", "main"]


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[Transformer, Config]:
    """Rebuild the model described by a checkpoint and load its weights."""
    ckpt = load_checkpoint(ckpt_path, map_location=device, restore_rng=False)
    cfg = Config.from_dict(ckpt["config"])
    model = Transformer(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motherbrain-sample", description="Sample from a MotherBrain checkpoint."
    )
    parser.add_argument("--ckpt", required=True, help="path to a .pt checkpoint")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json (default: the one beside the training data)",
    )
    parser.add_argument("--prompt", default="", help="prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, cfg = load_model(args.ckpt, device)

    tok_path = args.tokenizer or Path(cfg.data.data_dir) / "tokenizer.json"
    tokenizer = Tokenizer.load(tok_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tokenizer.vocab_size}) does not match the checkpoint "
            f"({cfg.model.vocab_size}); pass the matching --tokenizer"
        )

    ids = tokenizer.encode(args.prompt, add_bos=True)
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    budget = cfg.model.max_seq_len - prompt.shape[1]
    if budget <= 0:
        raise ValueError("prompt fills the whole context window; nothing left to generate")

    for i in range(args.num_samples):
        out = model.generate(
            prompt,
            max_new_tokens=min(args.max_new_tokens, budget),
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p > 0 else None,
            eos_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(out[0].tolist())
        if args.num_samples > 1:
            print(f"--- sample {i + 1}/{args.num_samples} ---")
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
