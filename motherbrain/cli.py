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
import re
import time
import sys
from pathlib import Path

from motherbrain.config import PRESETS, ModelConfig, human, scale_to
from motherbrain.security import check_exposure

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
        raise FileNotFoundError(f"no checkpoint at {ckpt}; run `mb train` first")
    dev = pick_device(device)
    model, meta = load_checkpoint(ckpt, device=dev)
    model.eval()
    tok_path = run / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(DEFAULT_CORPUS) / "tokenizer.json"
    tok = Tokenizer.load(str(tok_path))
    return model, tok, dev, meta


def shipped_model(run_dir: str) -> Path | None:
    """The exported model committed alongside the code, if there is one.

    A clone carries `models/motherbrain.pt` but not a training checkpoint -
    checkpoints are far too large for a repository. Without this, a fresh
    clone has a model sitting right there and every command insists there is
    none, which is the least helpful thing it could say.
    """
    root = Path(run_dir).resolve()
    for base in (project_root(), root.parent.parent, Path.cwd()):
        for name in ("motherbrain.pt", "motherbrain-15m.pt"):
            candidate = Path(base) / "models" / name
            if candidate.is_file():
                return candidate
    return None


def load_current(run_dir: str, device: str = "auto"):
    """The model as of the current version: base checkpoint + applied patches.

    Falls back to the exported model shipped with the repository when there is
    no trained checkpoint, so a fresh clone runs without training anything.
    """
    from motherbrain.patches import build_version
    from motherbrain.train import pick_device

    if not (Path(run_dir) / "checkpoint.pt").exists():
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
        print(rule)
        t0 = _time.time()
        n = 0
        for token in model.generate(ids, max_new_tokens=args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p, eos_id=EOS_ID,
                                    repetition_penalty=args.repetition_penalty):
            print(tok.decode([token]), end="", flush=True)
            n += 1
        elapsed = _time.time() - t0
        print(f"\n{rule}")
        print(f"{n} tokens in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.1f} tok/s)\n")
    return 0


# --------------------------------------------------------------------------
# status and bootstrap


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
        print("        this file holds the weights and is too large to commit,")
        print("        so a fresh clone never has one.")

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

    pending = corpus.n_documents - store.consumed_docs()
    if pending > 0:
        print(f"  {pending} document(s) fed but not yet learned "
              f"(run `mb patch`)")

    print()
    if has_ckpt:
        loadable = True
        try:
            from motherbrain.patches import build_version

            _, _, version = build_version(args.run)
            print(f"READY — loaded v{version}.")
        except ValueError as exc:
            loadable = False
            print(f"NOT LOADABLE — {exc}")
        if loadable:
            print("  load it with:  mb chat        (interactive)")
            print("                 mb serve       (HTTP, for IDEs)")
        return 0

    print("NOT LOADABLE — there are no weights yet.")
    if has_docs:
        print("  you have a corpus, so train a base model:")
        print("    mb prepare && mb train --preset micro --steps 400")
    else:
        print("  fastest path from here:")
        print("    mb bootstrap")
    print("  or download the base checkpoint from the CI run's")
    print("  'motherbrain-base-checkpoint' artifact into runs/default/.")
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
    from motherbrain.data import Corpus
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
        action, cap = choose_start()
        mode = "voice" if action == "program-voice" else "text"
    else:
        action = "console"
        action = "feed" if args.mode == "feed" else "console"
        mode, cap = ("text" if args.mode == "feed" else args.mode), Capability()
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

    def write_program() -> None:
        """Describe a program; MotherBrain writes code from the description.

        The description becomes a module docstring, because that is the shape
        the model saw during training: a docstring followed by definitions. It
        writes plausible Python, not correct Python, and says so rather than
        letting the output imply more than it is.
        """
        import torch

        from motherbrain.tokenizer import EOS_ID

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

        # Shape the prompt like the training data - a docstring, then a
        # definition - and put the description's own words in the function
        # name. The model cannot follow an instruction, but it does continue
        # from context, so words that appear in the signature pull the body
        # towards the same subject. That is the most a base model gives you.
        words = [w for w in re.findall(r"[A-Za-z]+", want.lower())
                 if w not in {"a", "an", "the", "that", "to", "and", "of",
                              "for", "in", "it", "with", "program"}]
        slug = "_".join(words[:4]) or "main"
        head = f'"""{want}"""\n\n\ndef {slug}('
        opener = f"def {slug}("

        print(f"\nwriting ({args.max_tokens} tokens)...\n")
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
        code = "".join(produced)
        print(f"\n{rule}")
        print(f"{len(produced) - 1} tokens from a {human(model.n_params())} model. "
              f"This is plausible Python, not working Python: a model this size "
              f"reproduces the shape of code, and cannot be told what to write. "
              f"Read it before running it.\n")
        say("program written")

        try:
            where = input("save to a file? [path, or blank to skip] ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print()
            return
        if where:
            path = Path(where).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'"""{want}"""\n\n{code}\n', encoding="utf-8")
            print(f"written to {path}\n")

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

    def read_line() -> str:
        """One line of input: spoken when that is possible, typed otherwise."""
        if mode == "voice" and cap.listen:
            from motherbrain.voice import listen

            print("listening...", end="", flush=True)
            heard = listen(cap)
            print()
            if heard:
                print(f"> {heard}")
                return heard
        return input("> ")

    if action in ("program", "program-voice"):
        write_program()
    elif action == "feed" or mode == "feed":
        feed_at_startup()
        mode = "text"

    print("/help for commands, empty line or ctrl-c to leave.\n")

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


def cmd_export(args) -> int:
    """Write a compact, self-contained, inference-only model file.

    A training checkpoint carries optimizer state and loads through pickle.
    An exported model carries fp16 weights, the config and the tokenizer as
    plain JSON strings, so it is roughly a sixth the size and loads with
    `weights_only=True` - no code execution on load, which matters for a file
    meant to be shared.
    """
    import json as _json

    import torch

    from motherbrain.patches import PatchStore

    model, tok, device, version = load_current(args.run, "cpu")
    store = PatchStore(args.run, create=False)

    steps = 0
    try:
        ckpt = torch.load(Path(args.run) / "checkpoint.pt", map_location="cpu",
                          weights_only=False)
        steps = ckpt.get("step", 0)
    except Exception:
        pass

    weights = {k: v.detach().to(torch.float16 if args.fp16 else torch.float32)
               for k, v in model.state_dict().items()}
    payload = {
        "format": "motherbrain-model-v1",
        "config_json": _json.dumps(model.cfg.to_dict()),
        "tokenizer_json": Path(args.run if (Path(args.run) / "tokenizer.json").exists()
                               else args.corpus).joinpath("tokenizer.json").read_text(),
        "weights": weights,
        "version": int(version),
        "steps": int(steps),
        "base_fingerprint": store.base_fingerprint,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    size = out.stat().st_size / 1e6
    print(f"exported v{version} ({human(model.n_params())} params, {steps:,} steps)")
    print(f"  {out}  {size:,.1f} MB  "
          f"{'fp16' if args.fp16 else 'fp32'}, inference only")
    print(f"  load it with:  mb chat --model {out}")
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
    cfg = PatchConfig(mode=args.mode, grow_experts=args.grow,
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
        print("error: openssl not found; install it or supply your own "
              "certificate to `mb serve --tls-cert/--tls-key`", file=sys.stderr)
        return 1
    if proc.returncode != 0:
        print(f"error: openssl failed:\n{proc.stderr.strip()}", file=sys.stderr)
        return 1

    key_path.chmod(0o600)  # a private key readable by others is not private
    fp = subprocess.run(
        ["openssl", "x509", "-in", str(cert_path), "-noout", "-fingerprint", "-sha256"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"certificate  {cert_path}")
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
                   help="training steps used by /grow")
    s.add_argument("--mode", choices=["ask", "text", "voice", "feed"],
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

    s = sub.add_parser("cert", help="generate a self-signed TLS certificate")
    s.add_argument("--dir", default="certs", help="where to write the pair")
    s.add_argument("--host", action="append",
                   help="extra hostname or IP to include (repeatable)")
    s.add_argument("--days", type=int, default=825)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_cert)

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
