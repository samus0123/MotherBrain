"""Command line interface: `python -m motherbrain <command>`."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .config import ModelConfig, RunConfig, TrainConfig


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_documents(patterns: list[str], split_on: str | None) -> list[str]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
        if not matched:
            raise SystemExit(f"no files matched '{pattern}'")
        paths.extend(p for p in matched if p.is_file())

    docs: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if split_on:
            docs.extend(chunk for chunk in text.split(split_on) if chunk.strip())
        else:
            docs.append(text)
    return docs


def _load_tokenizer(path: str | None):
    from .tokenizer import ByteTokenizer, Tokenizer

    if not path:
        return ByteTokenizer()
    return Tokenizer.load(path)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_tokenizer(args: argparse.Namespace) -> None:
    from .tokenizer import Tokenizer

    docs = _read_documents(args.input, args.split_on)
    print(f"training BPE on {len(docs)} documents "
          f"({sum(len(d) for d in docs):,} chars) -> vocab {args.vocab_size}")
    tok = Tokenizer.train(docs, vocab_size=args.vocab_size, verbose=True)
    tok.save(args.out)
    sample = docs[0][:2000]
    ids = tok.encode(sample)
    ratio = len(sample.encode()) / max(1, len(ids))
    print(f"saved {args.out}: vocab {tok.vocab_size}, {ratio:.2f} bytes/token")
    if tok.decode(ids) != sample:
        raise SystemExit("tokenizer failed its own roundtrip check")
    print("roundtrip check passed")


def cmd_data(args: argparse.Namespace) -> None:
    from .data import pack_documents, write_meta

    tok = _load_tokenizer(args.tokenizer)
    docs = _read_documents(args.input, args.split_on)
    n_val = max(1, int(len(docs) * args.val_fraction)) if len(docs) > 1 else 0
    train_docs = docs[: len(docs) - n_val] if n_val else docs
    val_docs = docs[len(docs) - n_val :] if n_val else []

    out_dir = Path(args.out_dir)
    splits = {}
    splits["train"] = pack_documents(
        train_docs, tok, out_dir / "train.bin", tok.vocab_size, eot_id=tok.eot_id
    )
    if val_docs:
        splits["val"] = pack_documents(
            val_docs, tok, out_dir / "val.bin", tok.vocab_size, eot_id=tok.eot_id
        )
    write_meta(out_dir, tok.vocab_size, splits)
    for name, n in splits.items():
        print(f"{name}: {n:,} tokens -> {out_dir / (name + '.bin')}")


def cmd_params(args: argparse.Namespace) -> None:
    from .scaling import describe, get_preset, ladder_table

    if args.ladder:
        print(ladder_table())
        return
    cfg = _config_from_args(args)
    print(describe(cfg, args.preset or args.config or "model"))


def cmd_train(args: argparse.Namespace) -> None:
    from .train import train

    run = _run_from_args(args)
    train(run)


def cmd_sample(args: argparse.Namespace) -> None:
    import torch

    from .sample import generate
    from .train import model_from_checkpoint, resolve_device

    device = resolve_device(args.device)
    model, run = model_from_checkpoint(args.checkpoint, device)
    tok = _load_tokenizer(args.tokenizer)

    prompt_ids = tok.encode(args.prompt) if args.prompt else [tok.eot_id]
    idx = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long)
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_id=tok.eot_id if args.stop_at_eos else None,
        seed=args.seed,
    )
    for i, row in enumerate(out):
        text = tok.decode(row.tolist())
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)


def cmd_config(args: argparse.Namespace) -> None:
    run = _run_from_args(args)
    run.save(args.out)
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# config assembly
# --------------------------------------------------------------------------
MODEL_OVERRIDES = (
    "vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len",
    "n_experts", "n_experts_per_tok", "n_shared_experts", "dropout", "tie_embeddings",
)
TRAIN_OVERRIDES = (
    "data_dir", "batch_size", "seq_len", "grad_accum_steps", "max_steps", "warmup_steps",
    "lr", "weight_decay", "grad_clip", "dtype", "compile", "grad_checkpoint", "device",
    "out_dir", "log_every", "eval_every", "save_every", "seed", "resume", "schedule",
)


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    from .scaling import get_preset

    if getattr(args, "config", None):
        cfg = RunConfig.load(args.config).model
    elif getattr(args, "preset", None):
        cfg = get_preset(args.preset)
    else:
        cfg = ModelConfig()

    raw = cfg.to_dict()
    overridden = {
        key for key in MODEL_OVERRIDES if getattr(args, key, None) is not None
    }
    for key in overridden:
        raw[key] = getattr(args, key)

    # `ModelConfig.__post_init__` fills in n_kv_heads / head_dim / ffn_hidden from
    # dim and n_heads, so the base config always carries concrete values. If the
    # user changes what those were derived from without restating them, the stale
    # values are wrong (and `--n-heads 4` against a default of 8 kv heads is not
    # even a valid config). Clear the derived fields so they are recomputed.
    if {"dim", "n_heads"} & overridden:
        if "n_kv_heads" not in overridden and cfg.n_kv_heads == cfg.n_heads:
            raw["n_kv_heads"] = None  # was plain multi-head attention; keep it that way
        raw["head_dim"] = None
    if "dim" in overridden:
        raw["ffn_hidden"] = None
        raw["moe_ffn_hidden"] = None

    try:
        return ModelConfig.from_dict(raw)
    except ValueError as exc:
        # Surface config errors as a clean CLI message rather than a traceback.
        raise SystemExit(f"invalid model configuration: {exc}") from exc


def _run_from_args(args: argparse.Namespace) -> RunConfig:
    run = RunConfig.load(args.config) if getattr(args, "config", None) else RunConfig()
    run.model = _config_from_args(args)
    raw = run.train.to_dict()
    for key in TRAIN_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
    run.train = TrainConfig.from_dict(raw)
    # Keep the context length and the training window consistent.
    if run.train.seq_len > run.model.max_seq_len:
        run.train.seq_len = run.model.max_seq_len
    return run


def _add_model_args(p: argparse.ArgumentParser) -> None:
    from .scaling import PRESETS

    p.add_argument("--preset", choices=list(PRESETS), help="start from a size preset")
    p.add_argument("--config", help="path to a run config JSON")
    p.add_argument("--vocab-size", type=int, dest="vocab_size")
    p.add_argument("--dim", type=int)
    p.add_argument("--n-layers", type=int, dest="n_layers")
    p.add_argument("--n-heads", type=int, dest="n_heads")
    p.add_argument("--n-kv-heads", type=int, dest="n_kv_heads")
    p.add_argument("--max-seq-len", type=int, dest="max_seq_len")
    p.add_argument("--n-experts", type=int, dest="n_experts")
    p.add_argument("--n-experts-per-tok", type=int, dest="n_experts_per_tok")
    p.add_argument("--n-shared-experts", type=int, dest="n_shared_experts")
    p.add_argument("--dropout", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motherbrain", description="Train and run a personal large language model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # tokenizer
    p = sub.add_parser("tokenizer", help="train a byte-level BPE tokenizer")
    p.add_argument("input", nargs="+", help="text files or globs")
    p.add_argument("--out", default="data/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8192, dest="vocab_size")
    p.add_argument("--split-on", default=None, help="split files into documents on this string")
    p.set_defaults(func=cmd_tokenizer)

    # data
    p = sub.add_parser("data", help="tokenize text into packed .bin splits")
    p.add_argument("input", nargs="+", help="text files or globs")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json (omit for raw bytes)")
    p.add_argument("--out-dir", default="data/tokens", dest="out_dir")
    p.add_argument("--val-fraction", type=float, default=0.05, dest="val_fraction")
    p.add_argument("--split-on", default=None)
    p.set_defaults(func=cmd_data)

    # params
    p = sub.add_parser("params", help="report parameter counts for a configuration")
    _add_model_args(p)
    p.add_argument("--ladder", action="store_true", help="print the full size ladder")
    p.set_defaults(func=cmd_params)

    # train
    p = sub.add_parser("train", help="train a model")
    _add_model_args(p)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--seq-len", type=int, dest="seq_len")
    p.add_argument("--grad-accum-steps", type=int, dest="grad_accum_steps")
    p.add_argument("--max-steps", type=int, dest="max_steps")
    p.add_argument("--warmup-steps", type=int, dest="warmup_steps")
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float, dest="weight_decay")
    p.add_argument("--grad-clip", type=float, dest="grad_clip")
    p.add_argument("--schedule", choices=["cosine", "linear", "constant", "wsd"])
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device")
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--grad-checkpoint", action="store_true", default=None, dest="grad_checkpoint")
    p.add_argument("--log-every", type=int, dest="log_every")
    p.add_argument("--eval-every", type=int, dest="eval_every")
    p.add_argument("--save-every", type=int, dest="save_every")
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", nargs="?", const="auto")
    p.set_defaults(func=cmd_train)

    # sample
    p = sub.add_parser("sample", help="generate text from a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--max-new-tokens", type=int, default=200, dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, dest="top_k")
    p.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    p.add_argument("--repetition-penalty", type=float, default=1.0, dest="repetition_penalty")
    p.add_argument("--num-samples", type=int, default=1, dest="num_samples")
    p.add_argument("--stop-at-eos", action="store_true", dest="stop_at_eos")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_sample)

    # config
    p = sub.add_parser("config", help="write a run config JSON you can edit")
    _add_model_args(p)
    p.add_argument("--out", default="configs/run.json")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
