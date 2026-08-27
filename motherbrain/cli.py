"""`mb` — the command line for MotherBrain.

    mb scale   --preset mother        what a configuration would cost
    mb feed    ./notes ./src "text"   put information into the corpus
    mb prepare --preset small         learn a vocabulary, tokenize the corpus
    mb train   --steps 2000           train on everything fed so far
    mb chat                           talk to the checkpoint locally
    mb serve   --host 0.0.0.0         expose it over HTTP from anywhere
"""

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

from motherbrain.config import PRESETS, ModelConfig, human, scale_to

DEFAULT_CORPUS = os.environ.get("MB_CORPUS", "data/corpus")
DEFAULT_RUN = os.environ.get("MB_RUN", "runs/default")


# --------------------------------------------------------------------------
# scale


def cmd_scale(args) -> int:
    if args.params:
        cfg = scale_to(parse_count(args.params), base=args.base)
    else:
        if args.preset not in PRESETS:
            print(f"unknown preset {args.preset!r}; choose from {', '.join(PRESETS)}")
            return 2
        cfg = PRESETS[args.preset]
        if args.experts:
            cfg = ModelConfig.from_dict({**cfg.to_dict(), "n_experts": args.experts})

    print(cfg.summary())
    print()
    print(feasibility(cfg))
    if args.save:
        cfg.save(args.save)
        print(f"\nconfig written to {args.save}")
    return 0


def parse_count(s: str) -> float:
    """Accept 1e12, 175B, 1.5T, 500M."""
    s = s.strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if s and s[-1] in mult:
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


def feasibility(cfg: ModelConfig) -> str:
    """An honest statement of what this configuration actually requires."""
    train_gb = cfg.memory_bytes(optimizer=True) / 1e9
    gpu_gb = 80  # an H100
    n_gpu = max(1, int(train_gb / gpu_gb) + 1)
    # Chinchilla-ish: ~20 tokens per active parameter is the usual target.
    tokens = cfg.n_active_params * 20
    # 6 FLOPs per active parameter per token, at ~400 TFLOP/s sustained.
    flops = 6 * cfg.n_active_params * tokens
    gpu_hours = flops / (400e12 * 3600)

    lines = ["feasibility"]
    if train_gb < 8:
        lines.append("  trains on a laptop CPU or any single GPU.")
    elif train_gb < 80:
        lines.append(f"  needs ~{train_gb:.0f} GB — one datacenter GPU (A100/H100 80GB).")
    else:
        lines.append(f"  weights+optimizer need ~{train_gb:,.0f} GB, so ~{n_gpu:,} "
                     f"80GB GPUs just to hold it.")
    if train_gb >= 8:
        lines.append(f"  a compute-optimal run is ~{human(tokens)} tokens, "
                     f"~{gpu_hours:,.0f} GPU-hours.")
        if gpu_hours > 1e6:
            years = gpu_hours / (n_gpu * 24 * 365)
            lines.append(f"  on {n_gpu:,} GPUs at full utilisation that is ~{years:,.1f} "
                         f"years of wall-clock training.")
    lines.append("  inference cost tracks the active parameters, not the total.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# feed / prepare


def cmd_feed(args) -> int:
    from motherbrain.data import Corpus

    corpus = Corpus(args.corpus)
    files = chars = 0
    for item in args.inputs:
        p = Path(item)
        if p.exists():
            f, c = corpus.add_path(p, recursive=not args.no_recursive)
            files += f
            chars += c
            print(f"  {item}: {f} files, {c:,} chars")
        else:
            n = corpus.add_text(item, source="inline")
            chars += n
            print(f"  inline text: {n:,} chars")
    if not sys.stdin.isatty() and args.stdin:
        text = sys.stdin.read()
        n = corpus.add_text(text, source="stdin")
        chars += n
        print(f"  stdin: {n:,} chars")

    corpus.write_meta()
    print(f"\ncorpus {corpus.root}: {corpus.n_documents} documents, "
          f"{corpus.n_chars:,} chars total")
    print("next: mb prepare")
    return 0


def cmd_prepare(args) -> int:
    from motherbrain.data import Corpus

    corpus = Corpus(args.corpus)
    vocab = args.vocab_size or PRESETS[args.preset].vocab_size
    tok, n = corpus.prepare(vocab_size=vocab)
    print(f"\ncorpus ready: {n:,} tokens, vocab {tok.vocab_size}")
    print("next: mb train")
    return 0


# --------------------------------------------------------------------------
# train


def cmd_train(args) -> int:
    from motherbrain.data import Corpus
    from motherbrain.train import TrainConfig, train

    corpus = Corpus(args.corpus)
    tok = corpus.load_tokenizer()

    if args.config:
        cfg = ModelConfig.load(args.config)
    else:
        cfg = ModelConfig.from_dict(PRESETS[args.preset].to_dict())
    cfg.vocab_size = tok.vocab_size
    if args.experts:
        cfg.n_experts = args.experts
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    tc = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        seq_len=args.seq_len, lr=args.lr, warmup=args.warmup,
        eval_every=args.eval_every, save_every=args.save_every,
        log_every=args.log_every, device=args.device, compile=args.compile,
    )
    train(args.corpus, args.run, cfg, tc, resume=args.resume)
    print("next: mb chat   (or mb serve)")
    return 0


# --------------------------------------------------------------------------
# chat / generate


def load_runtime(run_dir: str, device: str = "auto"):
    from motherbrain.tokenizer import Tokenizer
    from motherbrain.train import load_checkpoint, pick_device

    run = Path(run_dir)
    ckpt = run / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}; run `mb train` first")
    dev = pick_device(device)
    model, meta = load_checkpoint(ckpt, device=dev)
    model.eval()
    tok_path = run / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(DEFAULT_CORPUS) / "tokenizer.json"
    tok = Tokenizer.load(str(tok_path))
    return model, tok, dev, meta


def load_current(run_dir: str, device: str = "auto"):
    """The model as of the current version: base checkpoint + applied patches."""
    from motherbrain.patches import build_version
    from motherbrain.train import pick_device

    model, tok, version = build_version(run_dir, device=device)
    return model, tok, pick_device(device), version


def cmd_chat(args) -> int:
    import torch
    from motherbrain.tokenizer import EOS_ID

    model, tok, device, version = load_current(args.run, args.device)
    print(f"MotherBrain v{version}, {human(model.n_params())} params.")
    print("type a prompt; ctrl-c or empty line to leave.\n")

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = iter(lambda: input("> "), "")

    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=device)
        for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p, eos_id=EOS_ID):
            print(tok.decode([token]), end="", flush=True)
        print("\n")
    return 0


# --------------------------------------------------------------------------
# patches and versions


def cmd_patch(args) -> int:
    """Train the not-yet-learned corpus documents into the next version."""
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchConfig, PatchStore, create_patch

    store = PatchStore(args.run)
    corpus = Corpus(args.corpus)
    pending = corpus.n_documents - store.consumed_docs()
    if pending <= 0:
        print(f"nothing new to learn: all {corpus.n_documents} documents are "
              f"already in v{store.current}")
        return 0

    print(f"patching v{store.current} with {pending} new document(s) ...")
    cfg = PatchConfig(rank=args.rank, steps=args.steps, batch_size=args.batch_size,
                      lr=args.lr, replay_ratio=args.replay, seq_len=args.seq_len)
    version = create_patch(args.run, args.corpus, cfg, note=args.note,
                           device=args.device)
    if version is None:
        print("nothing to do")
        return 0
    print(f"\nv{version.parent} -> v{version.version}   patch {version.patch_id}")
    print(f"  learned      {version.n_documents} docs, {version.n_tokens:,} tokens")
    print(f"  patch size   {version.trainable_params:,} trainable params "
          f"(rank {version.rank})")
    print(f"  loss         {version.loss_before:.4f} -> {version.loss_after:.4f}")
    return 0


def cmd_versions(args) -> int:
    from motherbrain.patches import PatchStore

    store = PatchStore(args.run)
    versions = store.versions()
    current = store.current
    print(f"v0  base checkpoint{'   <- current' if current == 0 else ''}")
    for v in versions:
        mark = "   <- current" if v.version == current else ""
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created_at))
        print(f"v{v.version}  {v.patch_id}  {when}  "
              f"{v.n_documents} docs / {v.n_tokens:,} tokens  "
              f"loss {v.loss_before:.3f}->{v.loss_after:.3f}{mark}")
        if v.note:
            print(f"      note: {v.note}")
        if args.verbose and v.sources:
            for s in v.sources[:5]:
                print(f"      from: {s}")
    if not versions:
        print("(no patches yet — feed information, then run `mb patch`)")
    return 0


def cmd_checkout(args) -> int:
    from motherbrain.patches import PatchStore

    store = PatchStore(args.run)
    target = int(str(args.version).lstrip("vV"))
    store.set_current(target)
    print(f"now serving v{target}")
    return 0


# --------------------------------------------------------------------------
# serve


def cmd_serve(args) -> int:
    import uvicorn

    from motherbrain.server import create_app

    app = create_app(run_dir=args.run, corpus_dir=args.corpus, device=args.device,
                     api_key=args.api_key, auto_patch=not args.no_auto_patch,
                     auto_patch_chars=args.auto_patch_chars,
                     auto_patch_delay=args.auto_patch_delay)
    print(f"MotherBrain serving on http://{args.host}:{args.port}")
    print(f"  OpenAI-compatible  http://{args.host}:{args.port}/v1")
    print(f"  Ollama-compatible  http://{args.host}:{args.port}")
    if not args.no_auto_patch:
        print("  auto-patch on: fed information becomes the next version by itself.")
    if args.host in ("0.0.0.0", "::"):
        print("reachable from any machine that can route to this host.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mb", description="MotherBrain: build, feed, train and serve a language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus directory")
        sp.add_argument("--run", default=DEFAULT_RUN, help="run/checkpoint directory")
        return sp

    s = sub.add_parser("scale", help="price out a configuration")
    s.add_argument("--preset", default="mother", help=f"one of: {', '.join(PRESETS)}")
    s.add_argument("--params", help="instead: smallest config with at least N params (e.g. 2T)")
    s.add_argument("--base", default="titan", help="preset to scale up from with --params")
    s.add_argument("--experts", type=int, help="override the expert count")
    s.add_argument("--save", help="write the resulting config to this path")
    s.set_defaults(func=cmd_scale)

    s = common(sub.add_parser("feed", help="add text, files or directories to the corpus"))
    s.add_argument("inputs", nargs="*", help="paths or literal text")
    s.add_argument("--stdin", action="store_true", help="also read stdin")
    s.add_argument("--no-recursive", action="store_true")
    s.set_defaults(func=cmd_feed)

    s = common(sub.add_parser("prepare", help="learn a vocabulary and tokenize the corpus"))
    s.add_argument("--preset", default="micro")
    s.add_argument("--vocab-size", type=int)
    s.set_defaults(func=cmd_prepare)

    s = common(sub.add_parser("train", help="train on the corpus"))
    s.add_argument("--preset", default="micro")
    s.add_argument("--config", help="a config.json instead of a preset")
    s.add_argument("--steps", type=int, default=500)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--seq-len", type=int)
    s.add_argument("--experts", type=int)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--warmup", type=int, default=50)
    s.add_argument("--eval-every", type=int, default=50)
    s.add_argument("--log-every", type=int, default=10)
    s.add_argument("--save-every", type=int, default=100)
    s.add_argument("--device", default="auto")
    s.add_argument("--compile", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(func=cmd_train)

    s = common(sub.add_parser("chat", help="generate text from a checkpoint"))
    s.add_argument("--prompt")
    s.add_argument("--max-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_chat)

    s = common(sub.add_parser("patch", help="learn new information as the next version"))
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--rank", type=int, default=8)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--replay", type=float, default=0.25,
                   help="share of each batch resampled from older material")
    s.add_argument("--seq-len", type=int)
    s.add_argument("--note", default="")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_patch)

    s = common(sub.add_parser("versions", help="show the model's lineage"))
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_versions)

    s = common(sub.add_parser("checkout", help="serve an earlier version"))
    s.add_argument("version", help="version number, e.g. 3 or v3")
    s.set_defaults(func=cmd_checkout)

    s = common(sub.add_parser("serve", help="expose the model over HTTP"))
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="auto")
    s.add_argument("--api-key", default=os.environ.get("MB_API_KEY"),
                   help="require this key (X-API-Key or Authorization: Bearer)")
    s.add_argument("--no-auto-patch", action="store_true",
                   help="do not learn fed information automatically")
    s.add_argument("--auto-patch-chars", type=int, default=2000,
                   help="learn once this much new text has arrived")
    s.add_argument("--auto-patch-delay", type=float, default=20.0,
                   help="seconds of quiet before learning what was fed")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
