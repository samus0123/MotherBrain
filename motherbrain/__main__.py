"""Command line interface: ``python -m motherbrain <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import SEED
from .checkpoint import default_checkpoint
from .lineage import DEFAULT_LINEAGE_PATH, Lineage, LineageError, Release
from .version import Version


def _human(count: int) -> str:
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= limit:
            return f"{count / limit:.2f}{suffix}"
    return str(count)


def _load(path: Path) -> Lineage:
    if not path.exists():
        return Lineage([SEED])
    return Lineage.load(path)


def _describe(release: Release) -> str:
    arch = release.architecture
    lines = [
        f"MotherBrain {release.version}"
        + (f"  ({release.note})" if release.note else ""),
        f"  parameters     {_human(release.parameter_count)} "
        f"({release.parameter_count:,})",
        f"  d_model        {arch.d_model}",
        f"  n_layers       {arch.n_layers}",
        f"  n_heads        {arch.n_heads} (head_dim {arch.head_dim})",
        f"  d_ff           {arch.d_ff}",
        f"  context        {arch.context_length}",
        f"  vocab          {arch.vocab_size}",
    ]
    return "\n".join(lines)


def _cmd_show(args: argparse.Namespace) -> int:
    lineage = _load(args.lineage)
    release = lineage.get(args.version) if args.version else lineage.current
    print(_describe(release))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    lineage = _load(args.lineage)
    print(f"{'version':<10}{'parameters':>14}{'growth':>10}  architecture")
    previous = None
    for release in lineage:
        arch = release.architecture
        growth = (
            f"{release.parameter_count / previous.parameter_count:.2f}x"
            if previous
            else "-"
        )
        print(
            f"{str(release.version):<10}"
            f"{_human(release.parameter_count):>14}"
            f"{growth:>10}  "
            f"d{arch.d_model} x{arch.n_layers}L "
            f"ff{arch.d_ff} ctx{arch.context_length}"
        )
        previous = release
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    lineage = _load(args.lineage)
    before = lineage.current
    release = lineage.evolve(args.kind, note=args.note or "")
    if args.write:
        lineage.save(args.lineage)
    print(
        f"{before.version} -> {release.version}: "
        f"{_human(before.parameter_count)} -> {_human(release.parameter_count)} "
        f"parameters ({release.parameter_count / before.parameter_count:.2f}x)"
    )
    print(_describe(release))
    if not args.write:
        print("\n(dry run; pass --write to record this release)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        lineage = _load(args.lineage)
    except LineageError as error:
        print(f"invalid lineage: {error}", file=sys.stderr)
        return 1
    print(f"ok: {len(lineage)} releases, parameters increase at every version")
    return 0


def _resolve_release(lineage: Lineage, version: str | None) -> Release:
    return lineage.get(version) if version else lineage.current


def _checkpoint_path(args: argparse.Namespace, version: Version) -> Path:
    return args.checkpoint or default_checkpoint(version)


def _cmd_init(args: argparse.Namespace) -> int:
    from . import checkpoint as ckpt
    from .training import new_model

    lineage = _load(args.lineage)
    release = _resolve_release(lineage, args.version)
    path = _checkpoint_path(args, release.version)

    meta_path, weights_path = ckpt.checkpoint_paths(path)
    if meta_path.exists() and not args.force:
        print(
            f"{meta_path} already exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 1

    model = new_model(release.architecture, seed=args.seed)
    meta_path, weights_path = ckpt.save(
        model, release.version, release.architecture, path
    )
    print(_describe(release))
    print(f"\ninitialised weights (random, untrained) -> {weights_path}")
    print(f"metadata -> {meta_path}")
    print("\nThese weights know nothing yet. Train them:")
    print(f"  python -m motherbrain train --data <your-text> --version {release.version}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from . import checkpoint as ckpt
    from .training import new_model, train

    lineage = _load(args.lineage)
    release = _resolve_release(lineage, args.version)
    path = _checkpoint_path(args, release.version)
    meta_path, _ = ckpt.checkpoint_paths(path)

    steps_before = 0
    tokens_before = 0
    if meta_path.exists() and not args.restart:
        loaded = ckpt.load(path)
        if loaded.architecture != release.architecture:
            print(
                f"checkpoint at {meta_path} has a different architecture than "
                f"release {release.version}; pass --restart to start fresh",
                file=sys.stderr,
            )
            return 1
        model = loaded.model
        steps_before, tokens_before = loaded.steps_trained, loaded.tokens_seen
        print(f"continuing from {meta_path} ({steps_before} steps so far)")
    else:
        model = new_model(release.architecture, seed=args.seed)
        print(f"starting from fresh weights for {release.version}")

    print(_describe(release))
    print()

    result = train(
        model,
        args.data,
        seq_len=args.seq_len or min(256, release.architecture.context_length),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    meta_path, weights_path = ckpt.save(
        model,
        release.version,
        release.architecture,
        path,
        steps_trained=steps_before + result.steps,
        tokens_seen=tokens_before + result.tokens_seen,
    )
    print(f"\nloss {result.final_loss:.4f}", end="")
    if result.final_val_loss is not None:
        print(f"  val_loss {result.final_val_loss:.4f}", end="")
    print(f"  ({result.tokens_seen:,} tokens this run)")
    print(f"saved -> {weights_path}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from . import checkpoint as ckpt
    from .generation import generate

    lineage = _load(args.lineage)
    release = _resolve_release(lineage, args.version)
    path = _checkpoint_path(args, release.version)
    meta_path, _ = ckpt.checkpoint_paths(path)
    if not meta_path.exists():
        print(
            f"no checkpoint at {meta_path}; train one first:\n"
            f"  python -m motherbrain train --data <your-text> "
            f"--version {release.version}",
            file=sys.stderr,
        )
        return 1

    loaded = ckpt.load(path)
    if loaded.steps_trained == 0:
        print(
            "warning: these weights are untrained, so the output is noise",
            file=sys.stderr,
        )

    text = generate(
        loaded.model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(args.prompt + text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motherbrain",
        description="Inspect and grow the MotherBrain lineage.",
    )
    parser.add_argument(
        "--lineage",
        type=Path,
        default=DEFAULT_LINEAGE_PATH,
        help="path to the lineage file (default: the packaged lineage.json)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="describe a release (default: the latest)")
    show.add_argument("version", nargs="?", help="e.g. 0.2.0")
    show.set_defaults(func=_cmd_show)

    history = sub.add_parser("history", help="list every release and its size")
    history.set_defaults(func=_cmd_history)

    evolve = sub.add_parser("evolve", help="grow the model by one version bump")
    evolve.add_argument("kind", choices=("major", "minor", "patch"))
    evolve.add_argument("--note", help="what changed in this release")
    evolve.add_argument(
        "--write", action="store_true", help="record the new release in the lineage"
    )
    evolve.set_defaults(func=_cmd_evolve)

    validate = sub.add_parser("validate", help="check the growth invariant")
    validate.set_defaults(func=_cmd_validate)

    for name, help_text in (
        ("init", "create randomly initialised weights for a release"),
        ("train", "train a release on your own text"),
        ("generate", "sample text from a trained release"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--version", help="release to use (default: the latest)")
        cmd.add_argument(
            "--checkpoint",
            type=Path,
            help="checkpoint stem (default: checkpoints/motherbrain-<version>)",
        )
        cmd.add_argument("--seed", type=int, default=0, help="random seed")
        if name == "init":
            cmd.add_argument(
                "--force", action="store_true", help="overwrite existing weights"
            )
            cmd.set_defaults(func=_cmd_init)
        elif name == "train":
            cmd.add_argument(
                "--data", required=True, type=Path, help="text file or directory"
            )
            cmd.add_argument("--epochs", type=int, default=1)
            cmd.add_argument("--batch-size", type=int, default=8)
            cmd.add_argument(
                "--seq-len",
                type=int,
                help="training window (default: min(256, context length))",
            )
            cmd.add_argument("--learning-rate", type=float, default=3e-4)
            cmd.add_argument("--validation-split", type=float, default=0.1)
            cmd.add_argument(
                "--restart",
                action="store_true",
                help="ignore existing weights and train from scratch",
            )
            cmd.set_defaults(func=_cmd_train)
        else:
            cmd.add_argument("prompt", help="text to continue")
            cmd.add_argument("--max-new-tokens", type=int, default=200)
            cmd.add_argument("--temperature", type=float, default=0.8)
            cmd.add_argument("--top-k", type=int, default=40)
            cmd.add_argument("--top-p", type=float, default=0.95)
            cmd.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
