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
import math
import os
import re
import time
import sys
from pathlib import Path

from motherbrain.config import PRESETS, ModelConfig, human, scale_to

def resolve_paths(args) -> None:
    """Fill in --corpus and --run, honouring --workspace when it is given.

    One directory holds everything a MotherBrain needs - corpus, run, models -
    so pointing at a drive should be one flag rather than three paths that can
    disagree with each other. An explicit --corpus or --run still wins.
    """
    workspace = getattr(args, "workspace", None)
    root = Path(workspace).expanduser() if workspace else None

    if getattr(args, "corpus", None) is None:
        args.corpus = str(root / "data" / "corpus") if root else DEFAULT_CORPUS
    if getattr(args, "run", None) is None:
        args.run = str(root / "runs" / "default") if root else DEFAULT_RUN


def project_root() -> Path:
    """Find the MotherBrain workspace, the way git walks up to find .git.

    Once `mb` is installed it can be run from anywhere, but the corpus and the
    checkpoints live in a particular directory. Resolving them against the cwd
    alone would make `mb status` report "no weights" while standing two levels
    inside a workspace that has them.
    """
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "runs").is_dir() or (d / "data" / "corpus").is_dir():
            return d
        if (d / "motherbrain" / "cli.py").is_file():
            return d
    return here


DEFAULT_CORPUS = os.environ.get("MB_CORPUS") or str(project_root() / "data" / "corpus")
DEFAULT_RUN = os.environ.get("MB_RUN") or str(project_root() / "runs" / "default")


# --------------------------------------------------------------------------
# scale


def cmd_scale(args) -> int:
    if args.fit_gpus:
        cfg, note = fit_to_hardware(args.fit_gpus, args.gpu_gb, base=args.base)
        if cfg is None:
            print(note)
            return 1
        print(f"largest configuration that fits on "
              f"{args.fit_gpus} x {args.gpu_gb:g}GB:\n")
        if note:
            print(f"note: {note}\n")
        print(cfg.summary())
        print()
        print(feasibility(cfg))
        if args.save:
            cfg.save(args.save)
            print(f"\nconfig written to {args.save}")
        return 0

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


# Preset shapes ordered small to large; --fit-gpus walks down this ladder when
# the requested shape cannot fit at all.
SHAPE_LADDER = ["micro", "small", "small-moe", "medium", "large", "titan"]


def fit_to_hardware(n_gpus: int, gpu_gb: float,
                    base: str = "titan") -> tuple[ModelConfig | None, str]:
    """The largest configuration that a given cluster can actually hold.

    "As many parameters as possible" only means something once it is bounded by
    memory you have. Experts are the axis grown here: they raise the total
    parameter count without raising per-token compute. If even a single expert
    of the requested shape does not fit, smaller shapes are tried in turn,
    because returning a configuration that does not fit would be worse than
    saying so.
    """
    budget = n_gpus * gpu_gb * 1e9

    candidates = [base]
    if base in SHAPE_LADDER:
        candidates = list(reversed(SHAPE_LADDER[:SHAPE_LADDER.index(base) + 1]))

    for shape in candidates:
        cfg = ModelConfig.from_dict(PRESETS[shape].to_dict())
        if cfg.n_experts == 0:
            cfg.n_experts = 1
            cfg.n_experts_per_token = 1
        cfg.n_experts = 1
        if cfg.memory_bytes(optimizer=True) > budget:
            continue  # this shape cannot fit even at its smallest

        low, high = 1, 1
        while True:  # grow until it no longer fits, then bisect
            cfg.n_experts = high * 2
            if cfg.memory_bytes(optimizer=True) > budget or high > (1 << 24):
                break
            low, high = high * 2, high * 2
        while low < high:
            mid = (low + high + 1) // 2
            cfg.n_experts = mid
            if cfg.memory_bytes(optimizer=True) <= budget:
                low = mid
            else:
                high = mid - 1

        cfg.n_experts = max(low, 1)
        cfg.n_experts_per_token = min(PRESETS[shape].n_experts_per_token or 1,
                                      cfg.n_experts)
        cfg.name = f"{shape}-fit-{n_gpus}x{gpu_gb:g}gb"
        note = "" if shape == base else (
            f"the {base} shape does not fit at any expert count; "
            f"using the {shape} shape instead.")
        return cfg, note

    return None, (
        f"nothing in the preset ladder fits in {n_gpus} x {gpu_gb:g}GB "
        f"({budget / 1e9:,.0f} GB). Even the micro shape needs "
        f"{PRESETS['micro'].memory_bytes(optimizer=True) / 1e9:.2f} GB to train.")


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
        base = shipped_base(run_dir)
        if base is None:
            raise FileNotFoundError(f"no checkpoint at {ckpt}; run `mb train` first")
        model, tok, dev, version, steps = load_exported(base, device)
        if version != 0:
            raise ValueError(
                f"{base} is version {version}, not the base. Patches apply on "
                f"top of the base, so building from a merged model would apply "
                f"them twice. Restore the v0 export, or train a base."
            )
        return model, tok, dev, {"step": steps}
    dev = pick_device(device)
    model, meta = load_checkpoint(ckpt, device=dev)
    model.eval()
    tok_path = run / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(DEFAULT_CORPUS) / "tokenizer.json"
    tok = Tokenizer.load(str(tok_path))
    return model, tok, dev, meta


def _find_model(run_dir: str, names: tuple[str, ...]) -> Path | None:
    """Look for a model beside the run directory first, then beside the code.

    The run directory wins because it is the one the caller named. Point
    `--run` at a drive and its own models/ directory is what should load —
    not whichever checkout the command happened to be typed from.
    """
    root = Path(run_dir).resolve()
    for base in (root.parent.parent, root.parent, project_root(), Path.cwd()):
        for name in names:
            candidate = Path(base) / "models" / name
            if candidate.is_file():
                return candidate
    return None


def shipped_base(run_dir: str) -> Path | None:
    """The committed v0 export that patches are applied on top of.

    A clone carries `models/motherbrain-base.pt` but not a training checkpoint;
    checkpoints are far too large for a repository. The base never changes, so
    committing it once alongside the patches makes the whole lineage
    reproducible from a clone - no training, no download.
    """
    return _find_model(run_dir, ("motherbrain-base.pt",))


def shipped_model(run_dir: str) -> Path | None:
    """The most complete exported model lying beside the code, if any.

    Prefers a merged current-version export, and falls back to the base. This
    is what `mb chat` and `mb console` run when there is nothing else; a fresh
    clone has a model sitting right there, and every command insisting there is
    none is the least helpful thing it could say.
    """
    return _find_model(run_dir, ("motherbrain.pt", "motherbrain-base.pt",
                                 "motherbrain-15m.pt"))


def load_current(run_dir: str, device: str = "auto"):
    """The model as of the current version: base checkpoint + applied patches.

    Falls back to the exported model shipped with the repository when there is
    no trained checkpoint, so a fresh clone runs without training anything.
    """
    from motherbrain.patches import build_version
    from motherbrain.train import pick_device

    if not (Path(run_dir) / "checkpoint.pt").exists() \
            and shipped_base(run_dir) is None:
        # No base to patch. A merged export still runs, at whatever version it
        # was exported at.
        shipped = shipped_model(run_dir)
        if shipped is not None:
            model, tok, dev, version, _steps = load_exported(str(shipped), device)
            return model, tok, dev, version
    model, tok, version = build_version(run_dir, device=device)
    return model, tok, pick_device(device), version


def cmd_chat(args) -> int:
    """Generate text from the current version.

    Output is delimited and counted. An undertrained model emits mostly
    whitespace, and a blank screen is indistinguishable from a command that
    silently failed, so the rules and the token count are what tell you it
    actually ran.
    """
    import time as _time

    import torch

    from motherbrain.tokenizer import EOS_ID

    image = None
    if getattr(args, "image", None):
        from motherbrain.vision import load_image

    if args.model:
        model, tok, device, version, steps = load_exported(args.model, args.device)
    else:
        model, tok, device, version = load_current(args.run, args.device)
        steps = 0
        try:
            ckpt = torch.load(Path(args.run) / "checkpoint.pt", map_location="cpu",
                              weights_only=False)
            steps = ckpt.get("step", 0)
        except Exception:
            pass

    print(f"MotherBrain v{version} · {human(model.n_params())} params · "
          f"trained {steps:,} steps · {device}")
    if steps < 500:
        print(f"note: {steps:,} training steps is very little. Expect mostly "
              f"whitespace and fragments. Train longer with "
              f"`mb train --steps 2000 --resume`.")
    if not args.prompt:
        print("type a prompt; empty line or ctrl-c to leave.")
    print()

    prompts = [args.prompt] if args.prompt else iter(lambda: input("> "), "")
    rule = "─" * 60

    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=device)
        if getattr(args, "image", None) and image is None:
            if model.vision is None:
                print("this model has no vision tower; ignoring --image",
                      file=sys.stderr)
            else:
                image = load_image(args.image, model.cfg.image_size).to(device)
        print(rule)
        t0 = _time.time()
        n = 0
        for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p, eos_id=EOS_ID,
                                    repetition_penalty=args.repetition_penalty,
                                    images=image):
            print(tok.decode([token]), end="", flush=True)
            n += 1
        elapsed = _time.time() - t0
        print(f"\n{rule}")
        print(f"{n} tokens in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.1f} tok/s)\n")
    return 0


# --------------------------------------------------------------------------
# status and bootstrap


def cmd_sight(args) -> int:
    """Give the current version sight, as the next version."""
    from motherbrain.sight import create_sight_patch

    def progress(info):
        if info["step"] % 25 == 0 or info["step"] == info["total"]:
            print(f"  step {info['step']}/{info['total']}  "
                  f"loss {info['loss']:.4f}", flush=True)

    tower = None
    if args.tower:
        import torch

        tower = torch.load(args.tower, map_location="cpu", weights_only=True)
        print(f"loading an already-trained tower from {args.tower}")

    version, result = create_sight_patch(
        args.run, device=args.device, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, layers=args.layers,
        width=args.width, heads=args.heads, image_size=args.image_size,
        patch_size=args.patch_size, n_train=args.n_train, n_eval=args.n_eval,
        progress_cb=progress, tower_state=tower)

    print(f"\nv{version.parent} -> v{version.version}")
    print(f"  gained     sight: a {args.layers}-layer vision tower")
    print(f"  grew       {human(version.params_before)} -> "
          f"{human(version.params_after)} "
          f"(+{version.params_after - version.params_before:,} parameters)")
    print(f"  names      {result['accuracy_after']:.1%} of held-out images "
          f"correctly (chance {result['chance']:.1%})")
    if result["accuracy_after"] < result["chance"] * 2:
        print("  warning    that is not meaningfully above chance. The tower is "
              "attached\n             but has not learned to see; train longer.")
    print(f"  in effect  the model now serving is v{version.version}")

    target = Path(args.export or shipped_model(args.run)
                  or Path("models/motherbrain.pt"))
    try:
        written = export_model(args.run, target, device=args.device,
                               corpus_dir=args.corpus)
        print(f"  exported   {target} ({written / 1e6:,.1f} MB)")
    except Exception as exc:                              # noqa: BLE001
        print(f"  warning: could not export to {target}: {exc}")
    return 0


def cmd_workspace(args) -> int:
    """Copy a complete, runnable MotherBrain onto another disk.

    The point is a directory that does not need this checkout: the base the
    patches apply to, the patches themselves, the manifest, the tokenizer, and
    a merged model of the current version. Point --workspace at it afterwards
    and every command works from there - including `mb serve`, which is what
    hosting from your own drive means.

    The corpus is left behind unless asked for. It is the largest thing here
    by far and is only needed to learn something new, not to run.
    """
    import shutil

    dest = Path(args.dest).expanduser()
    run = Path(args.run)
    (dest / "models").mkdir(parents=True, exist_ok=True)
    (dest / "runs" / "default" / "patches").mkdir(parents=True, exist_ok=True)

    copied: list[tuple[str, int]] = []

    def copy(src: Path, target: Path) -> None:
        if not src.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append((str(target.relative_to(dest)), target.stat().st_size))

    base = shipped_base(args.run)
    if base is None:
        print("error: no committed base to copy (models/motherbrain-base.pt).\n"
              "       run `mb export` from a checkout that has one.",
              file=sys.stderr)
        return 1
    copy(base, dest / "models" / "motherbrain-base.pt")

    for name in ("config.json", "tokenizer.json", "versions.json"):
        copy(run / name, dest / "runs" / "default" / name)
    for patch in sorted((run / "patches").glob("*.pt")):
        copy(patch, dest / "runs" / "default" / "patches" / patch.name)

    if args.with_corpus:
        corpus = Path(args.corpus)
        if corpus.is_dir():
            shutil.copytree(corpus, dest / "data" / "corpus", dirs_exist_ok=True)
            size = sum(f.stat().st_size for f in (dest / "data" / "corpus").rglob("*")
                       if f.is_file())
            copied.append(("data/corpus/", size))

    # The merged model: the whole current version in one file, which is what
    # you actually want a copy of on a disk you own.
    merged = dest / "models" / "motherbrain.pt"
    try:
        written = export_model(args.run, merged, device=args.device,
                               corpus_dir=args.corpus)
        copied.append((str(merged.relative_to(dest)), written))
    except Exception as exc:                              # noqa: BLE001
        print(f"warning: could not export the merged model: {exc}", file=sys.stderr)

    total = sum(size for _, size in copied)
    print(f"workspace  {dest.resolve()}")
    for name, size in copied:
        print(f"  {name:<44} {size / 1e6:>8,.1f} MB")
    print(f"  {'total':<44} {total / 1e6:>8,.1f} MB")
    if not args.with_corpus:
        print("\n  the corpus was not copied (--with-corpus to include it).")
        print("  it is only needed to teach something new, not to run.")

    print(f"\nrun it from there:")
    print(f"  mb gui       --workspace {dest}")
    print(f"  mb console   --workspace {dest}")
    print(f"  mb serve     --workspace {dest} --host 0.0.0.0")
    print(f"\nor set it once:  export MB_WORKSPACE={dest}")
    return 0


def cmd_gui(args) -> int:
    """Open the desktop window."""
    from motherbrain.gui import run

    return run(args.run, args.corpus, args.device, args.max_tokens,
               args.steps, args.grow)


def cmd_status(args) -> int:
    """Report exactly what is on disk and what to run next.

    "How do I load this?" has a different answer depending on what is present,
    and a fresh clone has no weights in it at all: the base checkpoint is far
    too large to commit, so it is either trained locally or fetched from the
    CI artifact. This prints which situation you are in.
    """
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchStore

    corpus = Corpus(args.corpus, create=False)
    store = PatchStore(args.run, create=False)
    run = Path(args.run)
    ckpt = run / "checkpoint.pt"

    def mark(ok: bool) -> str:
        return "yes" if ok else "no "

    has_docs = corpus.n_documents > 0
    has_tok = corpus.tokenizer_path.exists()
    has_tokens = corpus.n_tokens > 0
    has_ckpt = ckpt.exists()

    print(f"workspace  {Path(args.run).resolve().parent.parent}")
    print(f"  corpus   {Path(args.corpus).resolve()}")
    print(f"  run      {Path(args.run).resolve()}")
    print()
    print("corpus")
    print(f"  [{mark(has_docs)}] documents      {corpus.n_documents:,} "
          f"({corpus.n_chars:,} chars)")
    print(f"  [{mark(has_tok)}] tokenizer      {corpus.tokenizer_path}")
    print(f"  [{mark(has_tokens)}] tokenized      {corpus.n_tokens:,} tokens")

    base_export = shipped_base(args.run)

    print("weights")
    if has_ckpt:
        size = ckpt.stat().st_size / 1e6
        print(f"  [yes] base checkpoint  {ckpt}  ({size:,.0f} MB)")
        try:
            cfg = ModelConfig.load(str(run / "config.json"))
            print(f"        {cfg.name} preset, {human(cfg.n_params)} parameters, "
                  f"context {cfg.max_seq_len}")
        except (FileNotFoundError, ValueError):
            pass
    else:
        print(f"  [no ] base checkpoint  missing ({ckpt})")
        print("        this file holds the optimizer state as well, which is far")
        print("        too large to commit. You only need it to train.")
    if base_export is not None:
        size = base_export.stat().st_size / 1e6
        print(f"  [yes] base weights     {base_export}  ({size:,.0f} MB)")
        print("        committed, fp16, and enough to run and to patch.")
    else:
        print("  [no ] base weights     missing (models/motherbrain-base.pt)")

    versions = store.versions()
    print("lineage")
    print(f"  v0 base" + (f" + {len(versions)} patch(es)" if versions else ""))
    grown = [v for v in versions if v.mode == "grow" and v.params_after]
    if grown:
        print(f"  grown {human(grown[0].params_before)} -> "
              f"{human(grown[-1].params_after)} across {len(grown)} patch(es)")
    if versions:
        print(f"  current: v{store.current} of v{store.head}")
    if store.base_fingerprint:
        print(f"  base fingerprint: {store.base_fingerprint}")

    sighted = [v for v in versions if v.mode == "sight"]
    if sighted:
        best = max(sighted, key=lambda v: v.sight_accuracy)
        print(f"  sight      v{best.version}, naming {best.sight_accuracy:.1%} of "
              f"held-out images (chance 3.1%)")
    else:
        print("  sight      none yet (run `mb sight` to add a vision tower)")

    pending = corpus.n_documents - store.consumed_docs()
    if pending > 0:
        print(f"  {pending} document(s) fed but not yet learned "
              f"(run `mb patch`)")

    print()
    # build_version opens a PatchStore, which creates the run directory. Asking
    # what is on disk must never change what is on disk, so a run directory
    # that does not exist yet is reported rather than conjured.
    if (has_ckpt or base_export is not None) and run.exists():
        try:
            from motherbrain.patches import build_version

            _, _, version = build_version(args.run)
            print(f"READY — loaded v{version}.")
            print("  start it with:  mb gui          (windowed)")
            print("                  mb console      (terminal)")
            print("                  mb serve        (HTTP, for IDEs and browsers)")
            return 0
        except (ValueError, FileNotFoundError) as exc:
            print(f"NOT LOADABLE — {exc}")
            return 0

    print("NOT LOADABLE — there are no weights yet.")
    if has_docs:
        print("  you have a corpus, so train a base model:")
        print("    mb prepare && mb train --preset micro --steps 400")
    else:
        print("  fastest path from here:")
        print("    mb bootstrap")
    return 0


def cmd_bootstrap(args) -> int:
    """Go from a fresh clone to a loaded model in one command."""
    from motherbrain.data import Corpus
    from motherbrain.train import TrainConfig, train

    corpus = Corpus(args.corpus)
    run = Path(args.run)

    if corpus.n_documents == 0:
        sources = args.feed or ["motherbrain", "README.md"]
        print(f"feeding {', '.join(sources)} ...")
        for s in sources:
            p = Path(s)
            if p.exists():
                files, chars = corpus.add_path(p)
                print(f"  {s}: {files} files, {chars:,} chars")
        corpus.write_meta()
        if corpus.n_documents == 0:
            print("error: nothing to feed; pass --feed with a path", file=sys.stderr)
            return 1
    else:
        print(f"corpus already holds {corpus.n_documents} documents")

    if not corpus.tokenizer_path.exists() or corpus.n_tokens == 0:
        corpus.prepare(vocab_size=args.vocab_size)
    else:
        print(f"corpus already tokenized: {corpus.n_tokens:,} tokens")

    if (run / "checkpoint.pt").exists() and not args.force:
        print(f"base checkpoint already present at {run / 'checkpoint.pt'}")
    else:
        cfg = ModelConfig.from_dict(PRESETS[args.preset].to_dict())
        cfg.vocab_size = corpus.load_tokenizer().vocab_size
        if args.seq_len:
            cfg.max_seq_len = args.seq_len
        tc = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                         seq_len=args.seq_len, lr=args.lr, device=args.device)
        train(args.corpus, args.run, cfg, tc)

    print("\nbootstrapped. load it with:")
    print("  mb chat")
    print("  mb serve")
    return 0


# --------------------------------------------------------------------------
# console


def cmd_console(args) -> int:
    """An interactive console: tell MotherBrain what to do, one line at a time.

    The same command table the web console uses. Parsing is deterministic; the
    model completes prompts and does not interpret instructions.
    """
    import torch

    from motherbrain.commands import HELP, parse
    from motherbrain.data import TEXT_SUFFIXES, Corpus
    from motherbrain.patches import PatchConfig, PatchStore, create_patch
    from motherbrain.tokenizer import EOS_ID

    from motherbrain.voice import Capability, choose_start, detect, speak

    model = tok = device = None
    version = 0

    def load() -> bool:
        nonlocal model, tok, device, version
        try:
            model, tok, device, version = load_current(args.run, args.device)
            return True
        except FileNotFoundError:
            return False

    if not load():
        print(f"no model in {args.run}; run `mb bootstrap` first", file=sys.stderr)
        return 1

    corpus = Corpus(args.corpus)
    store = PatchStore(args.run, create=False)
    print(f"MotherBrain console — v{version}, {human(model.n_params())} params.")
    print()

    if args.mode == "ask":
        action, mode, cap = choose_start()
    else:
        # --mode names the same four things, plus the older text/voice spellings
        # of option 2 that scripts already pass.
        action = {"feed": "learn", "learn": "learn", "update": "apply",
                  "apply": "apply", "make": "make"}.get(args.mode, "do")
        mode = args.mode if args.mode in ("text", "voice") else "text"
        cap = Capability()
        if mode == "voice":
            cap = detect()
            if not cap.any:
                print(f"voice is unavailable here: {cap.reason}")
                print("using text.")
                print()
                mode = "text"

    def say(text: str) -> None:
        """Read a reply aloud in voice mode. Printing happens either way."""
        if mode == "voice" and cap.speak:
            speak(text, cap)

    def generate_code(want: str) -> str:
        """Write code from a description and return it.

        The description becomes a module docstring and its own words become the
        function name, because that is the shape the model saw in training. A
        base model cannot be told what to write - it continues from context -
        so seeding the signature is what pulls the body towards the subject.
        """
        import torch

        from motherbrain.tokenizer import EOS_ID

        words = [w for w in re.findall(r"[A-Za-z]+", want.lower())
                 if w not in {"a", "an", "the", "that", "to", "and", "of",
                              "for", "in", "it", "with", "program", "script"}]
        slug = "_".join(words[:4]) or "main"
        opener = f"def {slug}("
        head = f'"""{want}"""\n\n\n{opener}'

        rule = "─" * 60
        print(rule)
        print(opener, end="", flush=True)
        ids = torch.tensor([tok.encode(head, bos=True)], device=device)
        produced = [opener]
        for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                    temperature=0.6, top_k=40, top_p=0.95,
                                    repetition_penalty=1.2, eos_id=EOS_ID):
            piece = tok.decode([token])
            produced.append(piece)
            print(piece, end="", flush=True)
        print(f"\n{rule}")
        print(f"{len(produced) - 1} tokens from a {human(model.n_params())} model. "
              f"This is plausible Python, not working Python: a model this size "
              f"reproduces the shape of code, and cannot be told what to write. "
              f"Read it before running it.\n")
        say("program written")
        return f'"""{want}"""\n\n{"".join(produced)}\n'

    def do_make(want: str, path: str | None) -> None:
        """Write a program, save it, and offer to run it."""
        code = generate_code(want)

        if not path:
            words = [w for w in re.findall(r"[A-Za-z]+", want.lower())][:3]
            default = "_".join(words) or "program"
            try:
                path = input(f"save as [{default}.py, blank to skip] ").strip()
            except (EOFError, KeyboardInterrupt, OSError):
                print()
                return
            if not path:
                return
            path = path or f"{default}.py"

        target = Path(path).expanduser()
        if target.suffix == "":
            target = target.with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        print(f"written to {target}")

        try:
            answer = input("run it? it was written by a 25M model [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if answer.startswith("y"):
            do_run(str(target))
        else:
            print()

    def do_see(path: str, prompt: str) -> None:
        """Look at an image and continue from it."""
        import torch

        from motherbrain.tokenizer import EOS_ID
        from motherbrain.vision import load_image

        if model.vision is None:
            print("this model has no vision tower — it was trained text-only.")
            print("build one with vision_layers > 0 and train it before "
                  "asking it to look at anything.\n")
            return

        target = Path(path).expanduser()
        if not target.is_file():
            print(f"no such image: {target}\n")
            return
        try:
            image = load_image(str(target), model.cfg.image_size).to(device)
        except Exception as exc:
            print(f"could not read {target}: {exc}\n")
            return

        print(f"looking at {target} "
              f"({model.cfg.n_image_tokens} patches) ...")
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=device)
        rule = "─" * 60
        print(rule)
        for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    top_k=args.top_k, top_p=args.top_p,
                                    repetition_penalty=args.repetition_penalty,
                                    eos_id=EOS_ID, images=image):
            print(tok.decode([token]), end="", flush=True)
        print(f"\n{rule}\n")

    def do_write(path: str, content: str) -> None:
        """Write a file, asking for the text when it was not given inline."""
        target = Path(path).expanduser()
        if not content:
            print(f"What should go in {target}? Blank line to finish.\n")
            lines: list[str] = []
            while True:
                try:
                    line = input("  ")
                except (EOFError, KeyboardInterrupt, OSError):
                    print()
                    break
                if not line.strip():
                    break
                lines.append(line)
            content = "\n".join(lines)
        if not content:
            print("nothing to write.\n")
            return

        if target.exists():
            try:
                answer = input(f"{target} exists. overwrite? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt, OSError):
                print()
                return
            if not answer.startswith("y"):
                print("left alone.\n")
                return

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {len(content) + 1:,} bytes to {target}\n")

    def do_delete(path: str) -> None:
        """Delete a file, after confirming. Deleting is not undoable."""
        target = Path(path).expanduser()
        if not target.exists():
            print(f"no such file: {target}\n")
            return
        if target.is_dir():
            print(f"{target} is a directory; this only deletes files.\n")
            return
        size = target.stat().st_size
        try:
            answer = input(f"delete {target} ({size:,} bytes)? "
                           f"this cannot be undone [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if not answer.startswith("y"):
            print("kept.\n")
            return
        target.unlink()
        print(f"deleted {target}\n")

    def do_find(pattern: str) -> None:
        """Search the files below here for a pattern."""
        import re as _re

        try:
            rx = _re.compile(pattern)
        except _re.error as exc:
            print(f"not a valid pattern: {exc}\n")
            return

        root = Path.cwd()
        hits = scanned = 0
        for path in sorted(root.rglob("*")):
            if hits >= 40 or scanned >= 4000:
                break
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(p in {".git", "node_modules", "__pycache__", ".venv"}
                   for p in path.parts):
                continue
            scanned += 1
            try:
                for n, line in enumerate(path.read_text(
                        encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = path.relative_to(root)
                        print(f"  {rel}:{n}: {line.strip()[:96]}")
                        hits += 1
                        if hits >= 40:
                            break
            except OSError:
                continue
        print(f"\n{hits} match(es) in {scanned} file(s)"
              + (" (stopped early)" if hits >= 40 else "") + "\n")

    def do_sh(command: str) -> None:
        """Run a shell command and show what it did.

        This is the console reaching the machine directly. It is confined to
        the terminal, where you already have a shell, and refused over HTTP
        where it would be remote code execution.
        """
        import subprocess

        try:
            answer = input(f"run: {command}\nconfirm? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if not answer.startswith("y"):
            print("not run.\n")
            return

        rule = "─" * 60
        print(rule)
        try:
            proc = subprocess.run(command, shell=True, capture_output=True,
                                  text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"{rule}\nstopped after 60 seconds.\n")
            return
        except OSError as exc:
            print(f"{rule}\ncould not run it: {exc}\n")
            return
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        print(rule)
        print(f"exit code {proc.returncode}\n")

    def do_run(path: str) -> None:
        """Run a python file and show what it did."""
        import subprocess

        target = Path(path).expanduser()
        if not target.is_file():
            print(f"no such file: {target}\n")
            return

        print(f"running {target} ...")
        rule = "─" * 60
        print(rule)
        try:
            proc = subprocess.run([sys.executable, str(target)],
                                  capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"{rule}\nstopped after 30 seconds.\n")
            return
        except OSError as exc:
            print(f"{rule}\ncould not run it: {exc}\n")
            return

        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        print(rule)
        print(f"exit code {proc.returncode}"
              + ("" if proc.returncode == 0 else "  (it failed, which is usual "
                                                 "for code this model writes)")
              + "\n")

    def write_program() -> None:
        """The menu's write-a-program entry: describe it, then save and run it."""
        print("Describe the program. One line is enough.")
        print('  e.g. "read a csv file and print the column averages"\n')
        try:
            want = read_line().strip() if mode == "voice" else input("  ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if not want:
            print("nothing to write.\n")
            return
        print(f"\nwriting ({args.max_tokens} tokens)...\n")
        do_make(want, None)

    def learn_new_information() -> None:
        """Option 3: take information in. It is stored, not yet in effect.

        Storing changes nothing about the model. Applying it (option 4) is
        what trains it into the weights, and this says so rather than leaving
        the impression that feeding was enough.
        """
        print("What should MotherBrain learn?")
        print("Type or paste it. A file or directory path works too.")
        print("Blank line when you are done.\n")

        collected: list[str] = []
        while True:
            try:
                entry = read_line() if mode == "voice" else input("  ")
            except (EOFError, KeyboardInterrupt, OSError):
                print()
                break
            if not entry.strip():
                break
            collected.append(entry)

        chars = added = 0
        for entry in collected:
            path = Path(entry.strip()).expanduser()
            if path.exists():
                files, count = corpus.add_path(path)
                added += files
                chars += count
                print(f"  {path}: {files} file(s), {count:,} characters")
            else:
                chars += corpus.add_text(entry, source="console")
                added += 1

        if not added:
            print("nothing learned.\n")
            return

        corpus.write_meta()
        pending = corpus.n_documents - store.consumed_docs()
        print(f"\ntook {added} document(s), {chars:,} characters.")
        print(f"{pending} document(s) now waiting to be applied.")
        print("It is stored, not yet in the model. Choose option 4 to apply it "
              "as a patch\nand ascend to the next version.\n")
        say(f"learned {added} documents. {pending} waiting to be applied.")

    def apply_as_patch() -> None:
        """Option 4: train what was learned into the weights, ascend a version.

        This is where information stops being text on disk and becomes part of
        the model: new experts are trained on it, which grows the parameter
        count and mints the next version.
        """
        pending = corpus.n_documents - store.consumed_docs()
        if pending <= 0:
            print("Nothing waiting to be applied. Learn something first "
                  "(option 3).\n")
            return

        current = model.n_params()
        print(f"{pending} document(s) waiting.")
        print(f"Applying them grows the model from {human(current)} and "
              f"ascends to v{store.current + 1}.")
        try:
            answer = input("apply now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if answer.startswith("n"):
            print("left unapplied; choose option 4 whenever you want it.\n")
            return

        print("\napplying ...")
        v = create_patch(args.run, args.corpus,
                         PatchConfig(mode="grow", grow_experts=args.grow,
                                     steps=args.steps),
                         note="applied", device=args.device)
        if v is None:
            print("nothing to apply.\n")
            return

        print(f"\nv{v.parent} -> v{v.version}")
        print(f"  applied    {v.n_documents} document(s), {v.n_tokens:,} tokens")
        print(f"  grew       {human(v.params_before)} -> {human(v.params_after)} "
              f"(+{v.params_after - v.params_before:,} parameters)")
        print(f"  loss       {v.loss_before:.3f} -> {v.loss_after:.3f} "
              f"on the new material")
        print(f"  in effect  the model now serving is v{v.version}")
        load()

        # The grown model lives in runs/, which is gitignored and does not
        # survive a fresh clone or a wiped machine. Exporting here is what
        # makes the ascent durable: models/motherbrain.pt is the committed
        # artifact, and without this step every applied patch is temporary.
        target = Path(args.export or shipped_model(args.run)
                      or (project_root() / "models" / "motherbrain.pt"))
        try:
            written = export_model(args.run, target, device=args.device)
        except Exception as exc:
            print(f"\n  warning: could not export to {target}: {exc}")
            print("  the patch is applied but only in runs/, which is not "
                  "committed.\n")
        else:
            print(f"  exported   {target} ({written / 1e6:,.1f} MB)")
            print(f"             commit it to keep v{v.version}: "
                  f"git add {target} && git commit\n")
        say(f"applied. now version {v.version}, {human(v.params_after)} parameters")

    def feed_at_startup() -> None:
        """Take information first, then offer to learn it before continuing.

        Feeding only stores text; a patch is what puts it into the weights. The
        offer to grow immediately is here because the gap between the two is
        the thing people most often miss.
        """
        print("Paste or type what MotherBrain should learn.")
        print("A file or directory path works too. Blank line to finish.\n")
        collected: list[str] = []
        while True:
            try:
                entry = input("  ")
            except (EOFError, KeyboardInterrupt, OSError):
                print()
                break
            if not entry.strip():
                break
            collected.append(entry)

        if not collected:
            print("nothing fed.\n")
            return

        chars = files = 0
        for entry in collected:
            path = Path(entry.strip()).expanduser()
            if path.exists():
                f, c = corpus.add_path(path)
                files += f
                chars += c
                print(f"  {path}: {f} file(s), {c:,} characters")
            else:
                chars += corpus.add_text(entry, source="console")
                files += 1
        corpus.write_meta()

        pending = corpus.n_documents - store.consumed_docs()
        print(f"\nadded {files} document(s), {chars:,} characters. "
              f"{pending} waiting to be learned.")

        try:
            answer = input("learn it now? this grows the model [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if answer.startswith("n"):
            print("left in the corpus; run /grow when you want it learned.\n")
            return

        print("growing...")
        v = create_patch(args.run, args.corpus,
                         PatchConfig(mode="grow", grow_experts=1, steps=args.steps),
                         note="startup feed", device=args.device)
        if v is None:
            print("nothing to learn.\n")
        else:
            print(f"v{v.parent} -> v{v.version}: "
                  f"{human(v.params_before)} -> {human(v.params_after)} params, "
                  f"loss {v.loss_before:.3f} -> {v.loss_after:.3f}\n")
            load()

    def read_line(prompt: str = "> ") -> str:
        """One line of input: spoken when that is possible, typed otherwise."""
        if mode == "voice" and cap.listen:
            from motherbrain.voice import listen

            print(prompt.rstrip() + " listening...", end="", flush=True)
            heard = listen(cap)
            print()
            if heard:
                print(f"> {heard}")
                return heard
        return input(prompt)

    if action == "learn":
        learn_new_information()
        mode = "text"
    elif action == "apply":
        apply_as_patch()
        mode = "text"
    elif action == "make":
        # Option 1 goes straight to the question it exists to ask.
        try:
            want = read_line("what kind of program? ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return 0
        if want:
            do_make(want, None)

    print("Tell me what to do. For example:\n")
    print("  make a script that renames files      write code, save it, run it")
    print("  write notes.txt                       create a file")
    print("  find TODO                             search the files here")
    print("  run script.py                         run it and show the output")
    print("  list files                            what is here")
    print("  sh git status                         any shell command")
    print("  <anything else>                       the model continues it")
    print("\n/help for the full list, empty line or ctrl-c to leave.\n")

    while True:
        try:
            line = read_line().strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return 0
        if not line:
            return 0

        cmd = parse(line)

        if cmd.name in ("noop",):
            continue
        if cmd.name == "help":
            print(HELP + "\n")
        elif cmd.name == "error":
            print(f"{cmd.args['message']}\n")
        elif cmd.name == "unknown":
            print(f"unknown command /{cmd.args['command']} — try /help\n")
        elif cmd.name == "version":
            print(f"v{version}\n")
            say(f"version {version}")
        elif cmd.name == "status":
            build_parser().parse_args(
                ["status", "--corpus", args.corpus, "--run", args.run]).func(
                argparse.Namespace(corpus=args.corpus, run=args.run))
            print()
        elif cmd.name == "versions":
            cmd_versions(argparse.Namespace(run=args.run, corpus=args.corpus,
                                            verbose=False))
            print()
        elif cmd.name == "checkout":
            try:
                store.set_current(cmd.args["version"])
                load()
                print(f"now serving v{version}\n")
            except ValueError as exc:
                print(f"{exc}\n")
        elif cmd.name == "scale":
            name = cmd.args.get("preset", "mother")
            if name in PRESETS:
                print(PRESETS[name].summary() + "\n")
            else:
                print(f"unknown preset {name}\n")
        elif cmd.name == "learn":
            n = corpus.add_text(cmd.text, source="console")
            corpus.write_meta()
            pending = corpus.n_documents - store.consumed_docs()
            print(f"added {n:,} characters; {pending} document(s) waiting. "
                  f"run /grow to learn them.\n")
            say(f"learned. {pending} documents waiting.")
        elif cmd.name == "grow":
            pending = corpus.n_documents - store.consumed_docs()
            if pending <= 0:
                print("nothing new to learn\n")
                continue
            n = cmd.args.get("experts", 1)
            print(f"growing by {n} expert(s) per layer on {pending} document(s)...")
            v = create_patch(args.run, args.corpus,
                             PatchConfig(mode="grow", grow_experts=n, steps=args.steps),
                             note="console", device=args.device)
            if v is None:
                print("nothing to do\n")
            else:
                print(f"v{v.parent} -> v{v.version}: "
                      f"{human(v.params_before)} -> {human(v.params_after)} params, "
                      f"loss {v.loss_before:.3f} -> {v.loss_after:.3f}\n")
                say(f"grown to version {v.version}, "
                    f"{human(v.params_after)} parameters")
                load()
        elif cmd.name == "make":
            do_make(cmd.text, cmd.args.get("path"))
        elif cmd.name == "see":
            do_see(cmd.args["path"], cmd.args.get("prompt", ""))
        elif cmd.name == "write":
            do_write(cmd.args["path"], cmd.args.get("content", ""))
        elif cmd.name == "delete":
            do_delete(cmd.args["path"])
        elif cmd.name == "find":
            do_find(cmd.args["pattern"])
        elif cmd.name == "sh":
            do_sh(cmd.args["command"])
        elif cmd.name == "run":
            do_run(cmd.args["path"])
        elif cmd.name == "ls":
            where = Path(cmd.text.strip() or ".").expanduser()
            if not where.is_dir():
                print(f"no such directory: {where}\n")
            else:
                for entry in sorted(where.iterdir())[:60]:
                    mark = "/" if entry.is_dir() else " "
                    print(f"  {entry.name}{mark}")
                print()
        elif cmd.name == "cat":
            target = Path(cmd.args["path"]).expanduser()
            if not target.is_file():
                print(f"no such file: {target}\n")
            else:
                try:
                    body = target.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    print(f"could not read it: {exc}\n")
                else:
                    print(body[:8000] + ("\n... (truncated)" if len(body) > 8000 else ""))
                    print()
        elif cmd.name in ("train", "export"):
            print(f"run that from the command line: mb {cmd.name}\n")
        else:
            ids = torch.tensor([tok.encode(cmd.text, bos=True)], device=device)
            produced: list[str] = []
            for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                        temperature=args.temperature,
                                        top_k=args.top_k, top_p=args.top_p,
                                        repetition_penalty=args.repetition_penalty,
                                        eos_id=EOS_ID):
                piece = tok.decode([token])
                produced.append(piece)
                print(piece, end="", flush=True)
            print("\n")
            say("".join(produced))


# --------------------------------------------------------------------------
# export / import


def export_model(run_dir: str, out: "Path | str", fp16: bool = True,
                 device: str = "cpu", corpus_dir: str | None = None) -> int:
    """Write a compact, self-contained, inference-only model file.

    A training checkpoint carries optimizer state and loads through pickle. An
    export carries fp16 weights with the config and tokenizer as plain JSON
    strings, so it is roughly a sixth the size and loads with
    `weights_only=True` - no code execution on load, which matters for a file
    meant to be shared. Returns the bytes written.
    """
    import json as _json

    import torch

    from motherbrain.patches import PatchStore

    model, tok, _device, version = load_current(run_dir, device)
    store = PatchStore(run_dir, create=False)

    steps = 0
    try:
        ckpt = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu",
                          weights_only=False)
        steps = ckpt.get("step", 0)
    except Exception:
        pass

    tok_path = Path(run_dir) / "tokenizer.json"
    if not tok_path.exists() and corpus_dir:
        tok_path = Path(corpus_dir) / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(DEFAULT_CORPUS) / "tokenizer.json"

    payload = {
        "format": "motherbrain-model-v1",
        "config_json": _json.dumps(model.cfg.to_dict()),
        "tokenizer_json": tok_path.read_text(),
        "weights": {k: v.detach().to(torch.float16 if fp16 else torch.float32)
                    for k, v in model.state_dict().items()},
        "version": int(version),
        "steps": int(steps),
        "base_fingerprint": store.base_fingerprint,
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out.stat().st_size


def cmd_export(args) -> int:
    """Write the current version out as a shareable model file."""
    from motherbrain.patches import PatchStore

    model, _tok, _dev, version = load_current(args.run, "cpu")
    params = model.n_params()
    del model

    size = export_model(args.run, args.out, fp16=args.fp16,
                        corpus_dir=args.corpus)
    steps = PatchStore(args.run, create=False).manifest().get("base_docs", 0)
    print(f"exported v{version} ({human(params)} params)")
    print(f"  {args.out}  {size / 1e6:,.1f} MB  "
          f"{'fp16' if args.fp16 else 'fp32'}, inference only")
    print(f"  load it with:  mb chat --model {args.out}")
    return 0


def load_exported(path: str | Path, device: str = "auto"):
    """Load a model exported by `mb export`, without executing pickled code."""
    import json as _json

    import torch

    from motherbrain.model import MotherBrain
    from motherbrain.tokenizer import Tokenizer
    from motherbrain.train import pick_device

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "motherbrain-model-v1":
        raise ValueError(f"{path} is not a MotherBrain model export")

    cfg = ModelConfig.from_dict(_json.loads(payload["config_json"]))
    dev = pick_device(device)
    model = MotherBrain(cfg)
    model.load_state_dict({k: v.float() for k, v in payload["weights"].items()})
    model.to(dev).eval()

    tok = Tokenizer.__new__(Tokenizer)
    data = _json.loads(payload["tokenizer_json"])
    Tokenizer.__init__(tok, merges=[tuple(p) for p in data["merges"]],
                       vocab_size=data["vocab_size"])
    return model, tok, dev, payload.get("version", 0), payload.get("steps", 0)


# --------------------------------------------------------------------------
# patches and versions


def grown_config(cfg: ModelConfig, n_new: int) -> ModelConfig:
    """The config a growth patch of `n_new` experts per layer would produce.

    This mirrors motherbrain.growth.grow: the first growth of a dense model
    converts its feed-forward layers to MoE, so the dense FFN becomes an
    always-on shared expert and each layer gains a router. Anything that needs
    to know the size after growing calls this, so the arithmetic exists once.
    """
    grown = ModelConfig.from_dict(cfg.to_dict())
    grown.n_experts = (cfg.n_experts or 0) + n_new
    grown.moe_every = 1
    if cfg.n_experts == 0:
        grown.n_shared_experts = max(cfg.n_shared_experts, 1)
    grown.n_experts_per_token = min(max(cfg.n_experts_per_token, 1),
                                    grown.n_experts)
    return grown


def experts_for_target(cfg: ModelConfig, target: float) -> int:
    """How many experts per layer are needed to pass `target` parameters.

    Not a straight division: converting dense layers to MoE changes what a
    layer costs, so an estimate undershoots. This starts from the estimate and
    then checks the real count.
    """
    if cfg.n_params >= target:
        return 1

    probe = grown_config(cfg, 1)
    per_expert = probe.expert_params * max(probe.n_moe_layers, 1)
    if per_expert <= 0:
        raise ValueError("this shape cannot grow by experts")

    n = max(1, math.ceil((target - cfg.n_params) / per_expert))
    while grown_config(cfg, n).n_params < target:
        n += 1
    while n > 1 and grown_config(cfg, n - 1).n_params >= target:
        n -= 1
    return n


def cmd_patch(args) -> int:
    """Train the not-yet-learned corpus documents into the next version."""
    from motherbrain.data import TEXT_SUFFIXES, Corpus
    from motherbrain.patches import PatchConfig, PatchStore, create_patch

    store = PatchStore(args.run)
    corpus = Corpus(args.corpus)
    pending = corpus.n_documents - store.consumed_docs()
    if pending <= 0:
        print(f"nothing new to learn: all {corpus.n_documents} documents are "
              f"already in v{store.current}")
        return 0

    grow_experts = args.grow
    if args.to:
        target = parse_count(args.to)
        model, _tok, _dev, _v = load_current(args.run, "cpu")
        if model.cfg.n_params >= target:
            print(f"already {human(model.cfg.n_params)} parameters, "
                  f"past {human(target)}")
            return 0
        grow_experts = experts_for_target(model.cfg, target)
        grown = grown_config(model.cfg, grow_experts)
        need_gb = grown.memory_bytes(optimizer=True) / 1e9
        print(f"to pass {human(target)}: +{grow_experts} expert(s) per layer, "
              f"{human(model.cfg.n_params)} -> {human(grown.n_params)}")
        print(f"  training that needs about {need_gb:,.1f} GB of memory")
        if need_gb > args.max_gb and not args.force:
            print(f"\nthat exceeds --max-gb {args.max_gb:g}. Growth is cheap to "
                  f"describe and expensive to hold:\n  a patch trains whole new "
                  f"experts, and they have to fit in memory.\n  Raise --max-gb, "
                  f"lower --to, or pass --force to try anyway.", file=sys.stderr)
            return 1
        del model

    print(f"patching v{store.current} with {pending} new document(s) ...")
    cfg = PatchConfig(mode=args.mode, grow_experts=grow_experts,
                      rank=args.rank, steps=args.steps, batch_size=args.batch_size,
                      lr=args.lr, replay_ratio=args.replay, seq_len=args.seq_len)
    version = create_patch(args.run, args.corpus, cfg, note=args.note,
                           device=args.device)
    if version is None:
        print("nothing to do")
        return 0
    print(f"\nv{version.parent} -> v{version.version}   patch {version.patch_id}")
    print(f"  learned      {version.n_documents} docs, {version.n_tokens:,} tokens")
    if version.mode == "grow":
        added = version.params_after - version.params_before
        print(f"  grew         +{version.grow_experts} expert(s) per layer, "
              f"+{added:,} parameters")
        print(f"  size         {human(version.params_before)} -> "
              f"{human(version.params_after)}")
    else:
        print(f"  patch size   {version.trainable_params:,} trainable params "
              f"(rank {version.rank})")
    print(f"  trained      {version.trainable_params:,} parameters")
    print(f"  loss         {version.loss_before:.4f} -> {version.loss_after:.4f}")
    return 0


def cmd_record(args) -> int:
    """Where MotherBrain stands, separating what exists from what is specified.

    A parameter count in a config file is a claim about JSON. A trained model
    is a claim about weights somebody computed. Conflating the two is the
    easiest way to overstate this project, so they are reported apart.
    """
    from motherbrain.patches import PatchStore

    store = PatchStore(args.run, create=False)
    print("trained — weights that exist")
    try:
        model, _tok, _dev, version = load_current(args.run, "cpu")
        print(f"  v{version}  {human(model.n_params())} parameters "
              f"({model.n_params():,})")
        if store.largest:
            print(f"  largest version in the lineage: {human(store.largest)}")
        del model
    except FileNotFoundError:
        print("  none — run `mb bootstrap` or clone the shipped model")

    print()
    print("specified — configurations, not weights")
    for name in ("titan", "leviathan", "mother"):
        cfg = PRESETS[name]
        print(f"  {name:<10} {human(cfg.n_params):>8} total, "
              f"{human(cfg.n_active_params):>8} active/token")

    print()
    reference = args.reference or os.environ.get("MB_RECORD_REFERENCE")
    if reference:
        target = parse_count(reference)
        mother = PRESETS["mother"].n_params
        print(f"reference {human(target)}")
        print(f"  the mother configuration is {mother / target:,.1f}x that")
        print(f"  it is a config file; nobody has trained it")
    else:
        print("no reference set. Pass --reference (e.g. --reference 2T) or set")
        print("MB_RECORD_REFERENCE to compare against a figure you trust.")
        print("Sizes of the largest models are mostly undisclosed or")
        print("unverifiable, so this tool does not ship a number of its own.")

    print()
    print(f"to grow the trained model:  mb patch --to 1B")
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
        size = (f"  {human(v.params_before)}->{human(v.params_after)}"
                if v.mode == "grow" and v.params_after else "")
        print(f"v{v.version}  {v.patch_id}  {when}  "
              f"{v.n_documents} docs / {v.n_tokens:,} tokens  "
              f"loss {v.loss_before:.3f}->{v.loss_after:.3f}{size}{mark}")
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
# tls


def cmd_cert(args) -> int:
    """Generate a self-signed certificate for serving over HTTPS.

    Shells out to openssl, which is already present nearly everywhere, so TLS
    costs no extra Python dependency. This is enough to encrypt the link on a
    network you control; a client will still warn that the certificate is not
    from a public authority, which is why the SHA-256 is printed for pinning.
    For a public hostname, use a real certificate instead.
    """
    import socket
    import subprocess

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = out / "server.crt", out / "server.key"
    if cert_path.exists() and not args.force:
        print(f"{cert_path} already exists; pass --force to replace it")
        return 1

    names, seen = [], set()

    def add(value: str) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        is_ip = all(part.isdigit() for part in value.split(".")) and value.count(".") == 3
        names.append(f"{'IP' if is_ip or ':' in value else 'DNS'}:{value}")

    for h in (args.host or []):
        add(h)
    add("localhost")
    add(socket.gethostname())
    try:
        add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    add("127.0.0.1")

    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", str(args.days), "-subj", "/CN=motherbrain",
        "-addext", f"subjectAltName={','.join(names)}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("error: openssl not found.", file=sys.stderr)
        if sys.platform == "win32":
            print("  Windows does not ship it. Options:\n"
                  "    winget install ShiningLight.OpenSSL\n"
                  "    or use Git Bash, which includes openssl\n"
                  "    or supply your own certificate to "
                  "`mb serve --tls-cert/--tls-key`", file=sys.stderr)
        else:
            print("  install it, or supply your own certificate to "
                  "`mb serve --tls-cert/--tls-key`", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        print(f"error: openssl failed:\n{proc.stderr.strip()}", file=sys.stderr)
        return 1

    # A private key readable by others is not private. chmod is a no-op on
    # Windows, where NTFS permissions are inherited instead, so say so rather
    # than implying the file is locked down when it is not.
    key_path.chmod(0o600)
    fp = subprocess.run(
        ["openssl", "x509", "-in", str(cert_path), "-noout", "-fingerprint", "-sha256"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"certificate  {cert_path}")
    if sys.platform == "win32":
        print(f"private key  {key_path}  (never commit this; on Windows its "
              f"permissions are whatever the folder grants)")
    else:
        print(f"private key  {key_path}  (mode 600, never commit this)")
    print(f"valid for    {args.days} days")
    print(f"names        {', '.join(names)}")
    print(f"{fp.lower()}")
    print(f"\nserve with:\n  mb serve --tls-cert {cert_path} --tls-key {key_path}")
    return 0


# --------------------------------------------------------------------------
# serve


def cmd_serve(args) -> int:
    import uvicorn

    from motherbrain.security import check_exposure
    from motherbrain.server import create_app

    # Validate the exposure before building anything, so a misconfigured
    # public bind fails immediately rather than after loading a model.
    if bool(args.tls_cert) != bool(args.tls_key):
        print("error: --tls-cert and --tls-key must be given together", file=sys.stderr)
        return 2
    tls = bool(args.tls_cert and args.tls_key)
    if tls:
        for p in (args.tls_cert, args.tls_key):
            if not Path(p).exists():
                print(f"error: no such file: {p}", file=sys.stderr)
                return 2

    for warning in check_exposure(args.host, args.api_key, tls, args.insecure):
        print(f"warning: {warning}")

    app = create_app(run_dir=args.run, corpus_dir=args.corpus, device=args.device,
                     api_key=args.api_key, auto_patch=not args.no_auto_patch,
                     auto_patch_chars=args.auto_patch_chars,
                     auto_patch_delay=args.auto_patch_delay,
                     allow_paths=args.allow_path, allow_origins=args.allow_origin,
                     rate_limit=args.rate_limit)

    scheme = "https" if tls else "http"
    print(f"MotherBrain serving on {scheme}://{args.host}:{args.port}")
    print(f"  OpenAI-compatible  {scheme}://{args.host}:{args.port}/v1")
    print(f"  Ollama-compatible  {scheme}://{args.host}:{args.port}")
    print(f"  auth               {'API key required' if args.api_key else 'OPEN (no key)'}")
    print(f"  path ingestion     "
          f"{', '.join(args.allow_path) if args.allow_path else 'disabled'}")
    if not tls:
        print("  plaintext: anything fed or generated crosses the network in the "
              "clear.\n                     run `mb cert`, then pass --tls-cert/--tls-key.")
    if not args.no_auto_patch:
        print("  auto-patch on: fed information becomes the next version by itself.")
    if args.host in ("0.0.0.0", "::"):
        print("reachable from any machine that can route to this host.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                ssl_certfile=args.tls_cert if tls else None,
                ssl_keyfile=args.tls_key if tls else None)
    return 0


# --------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that resolves --workspace as part of parsing.

    Doing it here rather than in main() means every caller gets it - the tests
    build a parser and invoke the command function directly, and a path that
    only main() fills in would be None for all of them.
    """

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        resolve_paths(parsed)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="mb", description="MotherBrain: build, feed, train and serve a language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--workspace", default=os.environ.get("MB_WORKSPACE"),
                        help="a MotherBrain directory to work in: its "
                             "data/corpus, runs/default and models/ are used "
                             "together. Point it at a drive to run from there.")
        sp.add_argument("--corpus", default=None, help="corpus directory")
        sp.add_argument("--run", default=None, help="run/checkpoint directory")
        return sp

    s = sub.add_parser("scale", help="price out a configuration")
    s.add_argument("--preset", default="mother", help=f"one of: {', '.join(PRESETS)}")
    s.add_argument("--params", help="instead: smallest config with at least N params (e.g. 2T)")
    s.add_argument("--base", default="titan", help="preset to scale up from with --params")
    s.add_argument("--experts", type=int, help="override the expert count")
    s.add_argument("--fit-gpus", type=int,
                   help="instead: the largest model that fits on this many GPUs")
    s.add_argument("--gpu-gb", type=float, default=80.0,
                   help="memory per GPU when using --fit-gpus (default 80)")
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
    s.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="divide the logits of tokens already seen; small models "
                        "loop without this (1.0 disables)")
    s.add_argument("--model", help="run an exported model file instead of a run dir")
    s.add_argument("--image", help="an image to look at (needs a vision tower)")
    s.set_defaults(func=cmd_chat)

    s = common(sub.add_parser("status", help="what is on disk, and what to run next"))
    s.set_defaults(func=cmd_status)

    s = common(sub.add_parser(
        "bootstrap", help="fresh clone -> a loaded model, in one command"))
    s.add_argument("--feed", action="append",
                   help="what to train on (default: this repo's own source)")
    s.add_argument("--preset", default="micro")
    s.add_argument("--vocab-size", type=int, default=4096)
    s.add_argument("--steps", type=int, default=400)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--lr", type=float, default=6e-4)
    s.add_argument("--device", default="auto")
    s.add_argument("--force", action="store_true", help="retrain even if weights exist")
    s.set_defaults(func=cmd_bootstrap)

    s = common(sub.add_parser(
        "console", help="tell MotherBrain what to do, interactively"))
    s.add_argument("--max-tokens", type=int, default=120)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--repetition-penalty", type=float, default=1.1)
    s.add_argument("--steps", type=int, default=100,
                   help="training steps used by /grow and by updating")
    s.add_argument("--grow", type=int, default=1,
                   help="experts added per layer when updating")
    s.add_argument("--export",
                   help="where applying a patch writes the model "
                        "(default: models/motherbrain.pt)")
    s.add_argument("--mode",
                   choices=["ask", "make", "do", "learn", "apply",
                            "text", "voice", "feed", "update"],
                   default="ask",
                   help="ask at startup (default), or go straight to one")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_console)

    s = common(sub.add_parser(
        "export", help="write a compact, shareable, inference-only model file"))
    s.add_argument("--out", default="models/motherbrain.pt")
    s.add_argument("--fp32", dest="fp16", action="store_false",
                   help="keep full precision (doubles the file size)")
    s.set_defaults(func=cmd_export, fp16=True)

    s = common(sub.add_parser("patch", help="learn new information as the next version"))
    s.add_argument("--mode", choices=["grow", "lora"], default="grow",
                   help="grow: add experts, so the model gets larger with every "
                        "version; lora: a low-rank delta that keeps its size")
    s.add_argument("--grow", type=int, default=1,
                   help="experts added per layer when growing")
    s.add_argument("--to", help="grow until the model passes this many "
                                "parameters, e.g. 1B, 500M, 2T")
    s.add_argument("--max-gb", type=float, default=12.0,
                   help="refuse a growth that needs more memory than this")
    s.add_argument("--force", action="store_true",
                   help="grow past --max-gb anyway")
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

    s = common(sub.add_parser(
        "record", help="what is trained, what is only configured, and how big"))
    s.add_argument("--reference", help="compare against a parameter count you "
                                       "trust, e.g. 2T")
    s.set_defaults(func=cmd_record)

    s = common(sub.add_parser("versions", help="show the model's lineage"))
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_versions)

    s = common(sub.add_parser("checkout", help="serve an earlier version"))
    s.add_argument("version", help="version number, e.g. 3 or v3")
    s.set_defaults(func=cmd_checkout)

    s = sub.add_parser("cert", help="generate a self-signed TLS certificate")
    s.add_argument("--dir", default="certs", help="where to write the pair")
    s.add_argument("--host", action="append",
                   help="extra hostname or IP to include (repeatable)")
    s.add_argument("--days", type=int, default=825)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_cert)

    s = common(sub.add_parser(
        "sight", help="give the current version sight, as the next version"))
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--batch-size", type=int, default=24)
    s.add_argument("--lr", type=float, default=6e-4)
    s.add_argument("--layers", type=int, default=4, help="vision tower depth")
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--image-size", type=int, default=64)
    s.add_argument("--patch-size", type=int, default=16)
    s.add_argument("--n-train", type=int, default=4096,
                   help="rendered image-caption pairs to train on")
    s.add_argument("--n-eval", type=int, default=192,
                   help="held-out pairs used to measure whether it can see")
    s.add_argument("--tower", help="a trained vision tower to load instead of "
                                   "training one here")
    s.add_argument("--export", help="where to write the merged model")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_sight)

    s = common(sub.add_parser(
        "workspace",
        help="copy a complete, runnable MotherBrain onto another disk"))
    s.add_argument("dest", help="where to put it, e.g. /media/usb/MotherBrain")
    s.add_argument("--with-corpus", action="store_true",
                   help="also copy the corpus (large; only needed to learn)")
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_workspace)

    s = common(sub.add_parser(
        "gui", help="open MotherBrain in a window (the same four options)"))
    s.add_argument("--max-tokens", type=int, default=120)
    s.add_argument("--steps", type=int, default=100,
                   help="training steps used when applying a patch")
    s.add_argument("--grow", type=int, default=1,
                   help="experts added per layer when applying a patch")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_gui)

    s = common(sub.add_parser("serve", help="expose the model over HTTP or HTTPS"))
    s.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept connections from other machines")
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
    s.add_argument("--tls-cert", default=os.environ.get("MB_TLS_CERT"),
                   help="PEM certificate; serves over HTTPS when given with --tls-key")
    s.add_argument("--tls-key", default=os.environ.get("MB_TLS_KEY"),
                   help="PEM private key")
    s.add_argument("--allow-path", action="append",
                   help="directory /feed may read from (repeatable; default none)")
    s.add_argument("--allow-origin", action="append",
                   help="restrict CORS to these origins (default: any)")
    s.add_argument("--rate-limit", type=int, default=120,
                   help="requests per minute per client address (0 disables)")
    s.add_argument("--insecure", action="store_true",
                   help="allow a public bind with no API key")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    # A Windows console defaults to a legacy code page, and the rules and
    # dashes this CLI prints are not in it - printing them raises
    # UnicodeEncodeError partway through a reply. Ask for UTF-8 and fall back
    # to replacement characters rather than dying mid-sentence.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

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
