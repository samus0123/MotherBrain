"""Saving and loading MotherBrain checkpoints.

A checkpoint is two files that travel together: ``<name>.weights.h5`` holds
the weights, and ``<name>.json`` records the version and architecture they
belong to. Loading rebuilds the model from that record, so weights can never
be read back into a differently shaped network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .architecture import Architecture
from .tokenizer import ByteTokenizer
from .version import Version

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def checkpoint_paths(path: Path | str) -> tuple[Path, Path]:
    """Return the (metadata, weights) paths for a checkpoint stem."""
    stem = Path(path)
    name = stem.name
    # Not `with_suffix`: version stems like "motherbrain-0.1.0" end in what
    # looks like a file extension, and would lose their patch number.
    for suffix in (".weights.h5", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return stem.with_name(name + ".json"), stem.with_name(name + ".weights.h5")


def default_checkpoint(version: Version, directory: Path | str = DEFAULT_CHECKPOINT_DIR):
    return Path(directory) / f"motherbrain-{version}"


@dataclass(frozen=True)
class Checkpoint:
    """A model plus the metadata describing what it is."""

    version: Version
    architecture: Architecture
    model: object  # keras.Model; typed loosely to keep imports lazy
    steps_trained: int = 0
    tokens_seen: int = 0

    @property
    def tokenizer(self) -> ByteTokenizer:
        return ByteTokenizer()


def save(
    model,
    version: Version,
    architecture: Architecture,
    path: Path | str,
    *,
    steps_trained: int = 0,
    tokens_seen: int = 0,
) -> tuple[Path, Path]:
    meta_path, weights_path = checkpoint_paths(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    meta = {
        "version": str(version),
        "architecture": architecture.to_dict(),
        "parameters": architecture.parameter_count,
        "tokenizer": "byte",
        "steps_trained": steps_trained,
        "tokens_seen": tokens_seen,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path, weights_path


def load(path: Path | str) -> Checkpoint:
    from .model import build_model

    meta_path, weights_path = checkpoint_paths(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"no checkpoint metadata at {meta_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"no checkpoint weights at {weights_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    architecture = Architecture.from_dict(meta["architecture"])
    model = build_model(architecture)
    model.load_weights(weights_path)
    return Checkpoint(
        version=Version.parse(meta["version"]),
        architecture=architecture,
        model=model,
        steps_trained=meta.get("steps_trained", 0),
        tokens_seen=meta.get("tokens_seen", 0),
    )
