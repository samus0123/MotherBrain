"""Configuration objects for MotherBrain models and training runs.

Everything the model needs to be rebuilt lives in `ModelConfig`, which is
serialised into every checkpoint. Training knobs live in `TrainConfig` and are
deliberately kept separate so a checkpoint can be reloaded under a different
training regime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _filter_known(cls, raw: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


@dataclass
class ModelConfig:
    """Architecture definition for a MotherBrain decoder.

    The defaults describe a small model that trains on a laptop. The presets in
    `PRESETS` scale the same architecture up to mixture-of-experts models whose
    total parameter counts run into the trillions while keeping the *active*
    parameters per token affordable.
    """

    # --- core dimensions -------------------------------------------------
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # None -> multi-head attention (n_kv_heads == n_heads)
    head_dim: int | None = None  # None -> dim // n_heads
    max_seq_len: int = 1024

    # --- feed-forward ----------------------------------------------------
    ffn_hidden: int | None = None  # None -> derived from ffn_mult, rounded to a multiple
    ffn_mult: float = 8 / 3  # SwiGLU keeps parameter count at ~ 4*dim^2 with this factor
    ffn_multiple_of: int = 128

    # --- normalisation / position ---------------------------------------
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0  # >1 stretches RoPE for longer contexts than trained on

    # --- regularisation --------------------------------------------------
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # --- embeddings ------------------------------------------------------
    tie_embeddings: bool = True
    logit_softcap: float = 0.0  # 0 disables; else logits = cap * tanh(logits / cap)

    # --- mixture of experts ----------------------------------------------
    # n_experts <= 1 means every layer is a dense SwiGLU MLP.
    n_experts: int = 1
    n_experts_per_tok: int = 2
    n_shared_experts: int = 0  # always-on experts, added to the routed output
    moe_layer_freq: int = 1  # every Nth layer is MoE; 1 = all layers
    moe_first_dense_layers: int = 0  # keep the first N layers dense (stabilises routing)
    moe_ffn_hidden: int | None = None  # per-expert hidden size; None -> ffn_hidden
    aux_loss_coef: float = 0.01  # load-balancing loss weight
    router_z_loss_coef: float = 1e-3  # keeps router logits from drifting
    router_jitter: float = 0.0  # multiplicative noise on router inputs while training

    # --- init ------------------------------------------------------------
    init_std: float = 0.02
    scale_residual_init: bool = True  # 1/sqrt(2*n_layers) on residual projections

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            if self.dim % self.n_heads != 0:
                raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
            self.head_dim = self.dim // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} must be a multiple of n_kv_heads {self.n_kv_heads}"
            )
        if self.ffn_hidden is None:
            hidden = int(self.dim * self.ffn_mult)
            mult = self.ffn_multiple_of
            self.ffn_hidden = ((hidden + mult - 1) // mult) * mult
        if self.moe_ffn_hidden is None:
            self.moe_ffn_hidden = self.ffn_hidden
        if self.n_experts > 1 and self.n_experts_per_tok > self.n_experts:
            raise ValueError("n_experts_per_tok cannot exceed n_experts")
        if self.moe_layer_freq < 1:
            raise ValueError("moe_layer_freq must be >= 1")

    # --- helpers ---------------------------------------------------------
    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head."""
        return self.n_heads // self.n_kv_heads

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.n_experts <= 1:
            return False
        if layer_idx < self.moe_first_dense_layers:
            return False
        return (layer_idx - self.moe_first_dense_layers) % self.moe_layer_freq == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        return cls(**_filter_known(cls, raw))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class TrainConfig:
    """Optimisation and run-management settings."""

    # data
    data_dir: str = "data/tokens"
    train_split: str = "train"
    val_split: str = "val"

    # batching
    batch_size: int = 8
    seq_len: int = 1024
    grad_accum_steps: int = 1

    # schedule
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"  # cosine | linear | constant | wsd

    # optimiser
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # precision / performance
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    compile: bool = False
    grad_checkpoint: bool = False
    device: str = "auto"

    # bookkeeping
    out_dir: str = "runs/motherbrain"
    log_every: int = 10
    eval_every: int = 250
    eval_steps: int = 20
    save_every: int = 500
    keep_last: int = 3
    seed: int = 1337
    resume: str = ""  # path to a checkpoint, or "auto" to pick up out_dir/latest.pt

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        return cls(**_filter_known(cls, raw))


@dataclass
class RunConfig:
    """A model + training pair, which is what a config file on disk holds."""

    name: str = "motherbrain"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model.to_dict(), "train": self.train.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        return cls(
            name=raw.get("name", "motherbrain"),
            model=ModelConfig.from_dict(raw.get("model", {})),
            train=TrainConfig.from_dict(raw.get("train", {})),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
