"""Typed, composable configuration.

Configs are plain dataclasses. They can be loaded from YAML (with single-level
``inherit:`` support so experiment files stay small) and overridden from the
command line with dotted ``key=value`` pairs, e.g. ``train.max_steps=5000``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ModelConfig",
    "DataConfig",
    "OptimConfig",
    "TrainConfig",
    "Config",
    "load_config",
]


@dataclass
class ModelConfig:
    """Decoder-only transformer hyper-parameters."""

    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    # Number of key/value heads. None means multi-head attention (n_kv_head == n_head);
    # a smaller value enables grouped-query attention.
    n_kv_head: int | None = None
    d_model: int = 768
    # Inner width of the SwiGLU MLP. None derives 8/3 * d_model rounded up to
    # ``ffn_multiple_of`` -- the usual Llama-style sizing.
    d_ff: int | None = None
    ffn_multiple_of: int = 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    # Fields that are filled in from other fields when left unset, mapped to the
    # flag tracking whether the current value is ours to recompute.
    _DERIVED_FIELDS = {"n_kv_head": "_auto_n_kv_head", "d_ff": "_auto_d_ff"}

    def __setattr__(self, name: str, value: Any) -> None:
        # Assigning a derived field pins it; assigning None hands it back to _derive().
        if name in ModelConfig._DERIVED_FIELDS:
            object.__setattr__(self, ModelConfig._DERIVED_FIELDS[name], value is None)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self._derive()
        self.validate()

    def _derive(self) -> None:
        """Fill in unset fields. Re-run on validate so edits to n_head/d_model take."""
        # object.__setattr__ so filling a value in does not count as pinning it.
        if self._auto_n_kv_head:
            object.__setattr__(self, "n_kv_head", self.n_head)
        if self._auto_d_ff:
            hidden = int(8 * self.d_model / 3)
            mult = self.ffn_multiple_of
            object.__setattr__(self, "d_ff", mult * ((hidden + mult - 1) // mult))

    def validate(self) -> None:
        self._derive()
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        assert self.n_kv_head is not None
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        for name in ("vocab_size", "n_layer", "n_head", "d_model", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class DataConfig:
    """Where tokenized shards live and how batches are drawn from them."""

    data_dir: str = "data/tokenized"
    train_split: str = "train"
    val_split: str = "val"
    # Draw windows in a shuffled order (train) or straight through (eval).
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class OptimConfig:
    """AdamW plus the learning-rate schedule."""

    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2000
    # None means "decay across the whole run" (train.max_steps).
    decay_steps: int | None = None
    schedule: str = "cosine"  # cosine | linear | constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    fused: bool = True

    def validate(self) -> None:
        if self.schedule not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


@dataclass
class TrainConfig:
    """Loop control, precision, checkpointing and logging."""

    out_dir: str = "runs/default"
    # Per-device micro-batch; the effective batch is
    # batch_size * grad_accum_steps * world_size.
    batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 10000
    seed: int = 1337

    device: str = "auto"  # auto | cuda | cpu | mps
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    compile: bool = False

    eval_interval: int = 500
    eval_steps: int = 50
    log_interval: int = 10
    ckpt_interval: int = 1000
    keep_last_n: int = 3
    # Save an extra checkpoint whenever validation loss improves.
    save_best: bool = True
    # Skip the (expensive) first-step eval when resuming or debugging.
    eval_at_start: bool = False

    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def validate(self) -> None:
        for name in ("batch_size", "grad_accum_steps", "max_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.device not in ("auto", "cuda", "cpu", "mps"):
            raise ValueError(f"unknown device {self.device!r}")
        if self.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"unknown dtype {self.dtype!r}")


@dataclass
class Config:
    """Root config object -- everything a run needs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.optim.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        cfg = _from_dict(cls, data)
        cfg.validate()
        return cfg

    def flat_dict(self) -> dict[str, Any]:
        """Flatten to ``{"model.n_layer": 12, ...}`` -- handy for experiment loggers."""
        out: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                out[prefix] = value

        walk("", self.to_dict())
        return out


# --------------------------------------------------------------------------------------
# dict/YAML/CLI plumbing
# --------------------------------------------------------------------------------------


def _unwrap_optional(hint: Any) -> Any:
    """``int | None`` -> ``int``; anything else is returned unchanged."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _resolve_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    hints = _resolve_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = _unwrap_optional(hints[name])
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _from_dict(hint, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce(raw: str, hint: Any) -> Any:
    """Turn a command-line string into the type the dataclass field declares."""
    hint = _unwrap_optional(hint)
    if raw.lower() in ("none", "null"):
        return None
    if hint is bool:
        low = raw.lower()
        if low in ("true", "1", "yes", "y", "on"):
            return True
        if low in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"cannot parse {raw!r} as bool")
    if hint is int:
        # Tolerate "1e4" / "5_000" style ints from the shell.
        return int(float(raw)) if ("e" in raw.lower() or "." in raw) else int(raw)
    if hint is float:
        return float(raw)
    return raw


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` strings in place and return the config."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        target: Any = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section {part!r} in {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not dataclasses.is_dataclass(target) or not hasattr(target, leaf):
            raise ValueError(f"unknown config key {dotted!r}")
        hint = _resolve_hints(type(target))[leaf]
        setattr(target, leaf, _coerce(raw, hint))
    cfg.validate()
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Build a :class:`Config` from an optional YAML file plus CLI overrides.

    A config may declare ``inherit: other.yaml`` (resolved relative to itself) to
    layer on top of a shared base. Inheritance chains are followed to any depth
    but cycles are rejected.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        chain: list[Path] = []
        seen: set[Path] = set()
        current: Path | None = path
        while current is not None:
            resolved = current.resolve()
            if resolved in seen:
                raise ValueError(f"circular inherit chain at {current}")
            seen.add(resolved)
            chain.append(current)
            raw = _read_yaml(current)
            parent = raw.get("inherit")
            current = (current.parent / parent) if parent else None
        # Apply oldest ancestor first so the requested file wins.
        for cfg_path in reversed(chain):
            layer = _read_yaml(cfg_path)
            layer.pop("inherit", None)
            data = _deep_merge(data, layer)

    cfg = Config.from_dict(data)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg
