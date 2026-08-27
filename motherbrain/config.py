"""Model configuration and exact parameter accounting for MotherBrain.

MotherBrain is a sparse Mixture-of-Experts transformer. That choice is what
makes the parameter count open-ended: total parameters grow linearly with the
number of experts, while the compute spent on any single token grows only with
`n_experts_per_token`. A configuration with a trillion total parameters can
route each token through a couple of billion of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields


def human(n: float) -> str:
    """Format a parameter count the way people actually say it out loud."""
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= limit:
            return f"{n / limit:.4g}{suffix}"
    return str(int(n))


@dataclass
class ModelConfig:
    """Every architectural knob MotherBrain has.

    Defaults describe the `micro` preset: small enough to train on a laptop
    CPU, identical in structure to the largest presets.
    """

    # Vocabulary and context
    vocab_size: int = 4096
    max_seq_len: int = 512

    # Core transformer shape
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4  # grouped-query attention; == n_heads means plain MHA
    head_dim: int | None = None  # defaults to d_model // n_heads

    # Feed-forward
    d_ff: int = 704  # hidden width of a dense FFN / of one expert

    # Mixture of experts. n_experts == 0 makes every layer a dense FFN.
    n_experts: int = 0
    n_experts_per_token: int = 2
    n_shared_experts: int = 0  # always-on experts, added to the routed result
    moe_every: int = 1  # put MoE on every Nth layer; others stay dense
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 1e-3

    # Regularization and numerics
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    # Bookkeeping
    name: str = "micro"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}; "
                    "set head_dim explicitly if you want a non-square attention shape"
                )
            self.head_dim = self.d_model // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be a multiple of n_kv_heads={self.n_kv_heads}"
            )
        if self.n_experts and self.n_experts_per_token > self.n_experts:
            raise ValueError(
                f"n_experts_per_token={self.n_experts_per_token} exceeds "
                f"n_experts={self.n_experts}"
            )
        if self.moe_every < 1:
            raise ValueError("moe_every must be >= 1")

    # ---- layer composition -------------------------------------------------

    def is_moe_layer(self, layer_idx: int) -> bool:
        """MoE layers are placed every `moe_every` layers, counting from the top."""
        return self.n_experts > 0 and ((layer_idx + 1) % self.moe_every == 0)

    @property
    def n_moe_layers(self) -> int:
        return sum(self.is_moe_layer(i) for i in range(self.n_layers))

    @property
    def n_dense_layers(self) -> int:
        return self.n_layers - self.n_moe_layers

    # ---- parameter accounting ---------------------------------------------
    #
    # These are computed analytically rather than by building the model, so a
    # configuration far too large to instantiate can still be priced out
    # exactly. tests/test_params.py checks the arithmetic against real modules.

    @property
    def attn_params_per_layer(self) -> int:
        q = self.d_model * self.n_heads * self.head_dim
        k = self.d_model * self.n_kv_heads * self.head_dim
        v = k
        o = self.n_heads * self.head_dim * self.d_model
        return q + k + v + o

    @property
    def expert_params(self) -> int:
        """One SwiGLU expert: gate, up, down."""
        return 3 * self.d_model * self.d_ff

    @property
    def dense_ffn_params_per_layer(self) -> int:
        return self.expert_params

    @property
    def moe_ffn_params_per_layer(self) -> int:
        router = self.d_model * self.n_experts
        experts = self.n_experts * self.expert_params
        shared = self.n_shared_experts * self.expert_params
        return router + experts + shared

    @property
    def norm_params_per_layer(self) -> int:
        return 2 * self.d_model  # pre-attention and pre-FFN RMSNorm weights

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def n_params(self) -> int:
        """Total parameters, counting each tied tensor once."""
        total = self.embedding_params
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        total += self.d_model  # final norm
        total += self.n_layers * (self.attn_params_per_layer + self.norm_params_per_layer)
        total += self.n_dense_layers * self.dense_ffn_params_per_layer
        total += self.n_moe_layers * self.moe_ffn_params_per_layer
        return total

    @property
    def n_active_params(self) -> int:
        """Parameters actually touched by a single token.

        For a dense model this equals `n_params`. For MoE it is far smaller,
        and it is what determines the cost of a forward pass.
        """
        total = self.embedding_params
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        total += self.d_model
        total += self.n_layers * (self.attn_params_per_layer + self.norm_params_per_layer)
        total += self.n_dense_layers * self.dense_ffn_params_per_layer
        if self.n_moe_layers:
            router = self.d_model * self.n_experts
            live = (self.n_experts_per_token + self.n_shared_experts) * self.expert_params
            total += self.n_moe_layers * (router + live)
        return total

    @property
    def n_non_embedding_params(self) -> int:
        emb = self.embedding_params * (1 if self.tie_embeddings else 2)
        return self.n_params - emb

    def memory_bytes(self, bytes_per_param: int = 2, optimizer: bool = False) -> int:
        """Bytes needed to hold the weights (bf16 by default).

        With `optimizer=True`, adds fp32 master weights plus Adam's two moments,
        which is what a training run actually has to fit.
        """
        weights = self.n_params * bytes_per_param
        if optimizer:
            weights += self.n_params * 12  # fp32 master + m + v
        return weights

    def summary(self) -> str:
        active_pct = 100.0 * self.n_active_params / max(self.n_params, 1)
        lines = [
            f"MotherBrain preset: {self.name}",
            f"  total parameters     {human(self.n_params)}  ({self.n_params:,})",
            f"  active per token     {human(self.n_active_params)}  ({active_pct:.3g}% of total)",
            f"  non-embedding        {human(self.n_non_embedding_params)}",
            f"  layers               {self.n_layers} "
            f"({self.n_moe_layers} MoE, {self.n_dense_layers} dense)",
            f"  d_model / d_ff       {self.d_model} / {self.d_ff}",
            f"  heads (q/kv)         {self.n_heads} / {self.n_kv_heads} x {self.head_dim}",
            f"  experts              {self.n_experts}"
            + (
                f", top-{self.n_experts_per_token}"
                f"{f' + {self.n_shared_experts} shared' if self.n_shared_experts else ''}"
                if self.n_experts
                else " (dense FFN)"
            ),
            f"  vocab / context      {self.vocab_size} / {self.max_seq_len}",
            f"  weights (bf16)       {self.memory_bytes() / 1e9:,.1f} GB",
            f"  training footprint   {self.memory_bytes(optimizer=True) / 1e9:,.1f} GB",
        ]
        return "\n".join(lines)

    # ---- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def _preset(name: str, **kw) -> ModelConfig:
    return ModelConfig(name=name, **kw)


# Presets climb from "runs on this laptop in a minute" to "needs a datacenter".
# Everything up to `medium` is trainable on ordinary hardware. Past that the
# architecture is unchanged and only the resources differ.
PRESETS: dict[str, ModelConfig] = {
    "micro": _preset(
        "micro", vocab_size=4096, max_seq_len=512, d_model=256, n_layers=6,
        n_heads=8, n_kv_heads=4, d_ff=704,
    ),
    "small": _preset(
        "small", vocab_size=16384, max_seq_len=1024, d_model=512, n_layers=8,
        n_heads=8, n_kv_heads=4, d_ff=1408,
    ),
    "small-moe": _preset(
        "small-moe", vocab_size=16384, max_seq_len=1024, d_model=512, n_layers=8,
        n_heads=8, n_kv_heads=4, d_ff=1408,
        n_experts=8, n_experts_per_token=2, n_shared_experts=1,
    ),
    "medium": _preset(
        "medium", vocab_size=32768, max_seq_len=2048, d_model=1024, n_layers=16,
        n_heads=16, n_kv_heads=4, d_ff=2816,
        n_experts=16, n_experts_per_token=2, n_shared_experts=1,
    ),
    "large": _preset(
        "large", vocab_size=65536, max_seq_len=4096, d_model=4096, n_layers=32,
        n_heads=32, n_kv_heads=8, d_ff=11008,
        n_experts=64, n_experts_per_token=4, n_shared_experts=2, moe_every=2,
    ),
    "titan": _preset(
        "titan", vocab_size=131072, max_seq_len=8192, d_model=8192, n_layers=80,
        n_heads=64, n_kv_heads=8, d_ff=22016,
        n_experts=256, n_experts_per_token=6, n_shared_experts=2, moe_every=2,
        tie_embeddings=False,
    ),
    # The two below exist to be priced, not to be casually launched. `mb scale`
    # will tell you exactly what hardware they imply.
    "leviathan": _preset(
        "leviathan", vocab_size=131072, max_seq_len=16384, d_model=16384, n_layers=120,
        n_heads=128, n_kv_heads=8, d_ff=45056,
        n_experts=512, n_experts_per_token=8, n_shared_experts=2,
        tie_embeddings=False,
    ),
    "mother": _preset(
        "mother", vocab_size=262144, max_seq_len=32768, d_model=20480, n_layers=160,
        n_heads=160, n_kv_heads=16, d_ff=57344,
        n_experts=2048, n_experts_per_token=8, n_shared_experts=4,
        tie_embeddings=False,
    ),
}


def scale_to(target_params: float, base: str = "titan") -> ModelConfig:
    """Return a config with at least `target_params` parameters.

    Widths are kept fixed and experts are added until the count is reached,
    which is the cheapest axis to grow: total parameters rise linearly while
    per-token compute stays flat.
    """
    cfg = ModelConfig.from_dict(PRESETS[base].to_dict())
    cfg.name = f"{base}-scaled"
    if cfg.n_experts == 0:
        cfg.n_experts = 1
        cfg.n_experts_per_token = 1
    per_expert = cfg.expert_params * max(cfg.n_moe_layers, 1)
    while cfg.n_params < target_params:
        deficit = target_params - cfg.n_params
        cfg.n_experts += max(1, int(deficit // per_expert))
    return cfg
