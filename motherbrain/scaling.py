"""Analytic parameter counting and the model-size ladder.

Counts are computed from a `ModelConfig` alone, so a configuration can be sized
up long before there is hardware able to instantiate it. `verify_counts` checks
the arithmetic against a real `nn.Module` for small configs, which is what keeps
the formulas honest.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig

BYTES_PER_DTYPE = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


@dataclass
class ParamBreakdown:
    embedding: int
    attention: int
    dense_ffn: int
    moe_experts: int
    moe_shared: int
    router: int
    norms: int
    lm_head: int

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.attention
            + self.dense_ffn
            + self.moe_experts
            + self.moe_shared
            + self.router
            + self.norms
            + self.lm_head
        )


def count_parameters(cfg: ModelConfig) -> ParamBreakdown:
    """Exact parameter count for `cfg`, without building the model."""
    d, h, kv, hd = cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    embedding = cfg.vocab_size * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * d

    # q + k + v + o projections, all bias-free
    per_layer_attn = d * (h * hd) + 2 * d * (kv * hd) + (h * hd) * d
    attention = per_layer_attn * cfg.n_layers

    n_moe = sum(1 for i in range(cfg.n_layers) if cfg.is_moe_layer(i))
    n_dense = cfg.n_layers - n_moe

    dense_ffn = n_dense * 3 * d * cfg.ffn_hidden
    moe_experts = n_moe * cfg.n_experts * 3 * d * cfg.moe_ffn_hidden
    moe_shared = n_moe * 3 * d * (cfg.moe_ffn_hidden * cfg.n_shared_experts)
    router = n_moe * d * cfg.n_experts

    # Two RMSNorms per block, plus the final norm.
    norms = cfg.n_layers * 2 * d + d

    return ParamBreakdown(
        embedding=embedding,
        attention=attention,
        dense_ffn=dense_ffn,
        moe_experts=moe_experts,
        moe_shared=moe_shared,
        router=router,
        norms=norms,
        lm_head=lm_head,
    )


def count_active_parameters(cfg: ModelConfig) -> int:
    """Parameters that participate in the forward pass of a single token."""
    b = count_parameters(cfg)
    n_moe = sum(1 for i in range(cfg.n_layers) if cfg.is_moe_layer(i))
    active_experts = (
        n_moe * cfg.n_experts_per_tok * 3 * cfg.dim * cfg.moe_ffn_hidden if n_moe else 0
    )
    return (
        b.embedding
        + b.attention
        + b.dense_ffn
        + active_experts
        + b.moe_shared
        + b.router
        + b.norms
        + b.lm_head
    )


def flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> int:
    """Forward FLOPs per token: 2 * active params, plus attention's quadratic term."""
    seq_len = seq_len or cfg.max_seq_len
    dense = 2 * count_active_parameters(cfg)
    attn = 2 * 2 * cfg.n_layers * cfg.n_heads * cfg.head_dim * seq_len
    return dense + attn


def memory_estimate(cfg: ModelConfig, dtype: str = "bf16", optimizer: str = "adamw") -> dict[str, float]:
    """Bytes needed to hold weights, gradients and optimizer state."""
    total = count_parameters(cfg).total
    w = BYTES_PER_DTYPE.get(dtype, 2)
    weights = total * w
    grads = total * w
    # AdamW keeps fp32 exp_avg and exp_avg_sq, plus an fp32 master copy under AMP.
    opt = total * 12 if optimizer == "adamw" else 0
    return {
        "weights_bytes": weights,
        "grad_bytes": grads,
        "optimizer_bytes": opt,
        "train_total_bytes": weights + grads + opt,
        "inference_bytes": weights,
    }


def kv_cache_bytes(cfg: ModelConfig, batch_size: int, seq_len: int, dtype: str = "bf16") -> float:
    w = BYTES_PER_DTYPE.get(dtype, 2)
    return 2 * cfg.n_layers * batch_size * seq_len * cfg.n_kv_heads * cfg.head_dim * w


def chinchilla_tokens(active_params: int, ratio: int = 20) -> int:
    """Compute-optimal training tokens (~20 tokens per active parameter)."""
    return active_params * ratio


def human(n: float) -> str:
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return f"{n:.0f}"


def human_bytes(n: float) -> str:
    for unit, size in (("PiB", 2**50), ("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f} B"


def describe(cfg: ModelConfig, name: str = "model") -> str:
    """A human-readable size report for a configuration."""
    b = count_parameters(cfg)
    total = b.total
    active = count_active_parameters(cfg)
    mem = memory_estimate(cfg)
    lines = [
        f"{name}",
        f"  dim {cfg.dim} x {cfg.n_layers} layers | {cfg.n_heads} heads "
        f"(kv {cfg.n_kv_heads}) | ctx {cfg.max_seq_len} | vocab {cfg.vocab_size}",
    ]
    if cfg.n_experts > 1:
        n_moe = sum(1 for i in range(cfg.n_layers) if cfg.is_moe_layer(i))
        lines.append(
            f"  MoE: {cfg.n_experts} experts x {n_moe} layers, top-{cfg.n_experts_per_tok}"
            + (f" + {cfg.n_shared_experts} shared" if cfg.n_shared_experts else "")
        )
    lines += [
        f"  total params:  {human(total)}  ({total:,})",
        f"  active params: {human(active)}  ({active / total * 100:.2f}% of total)",
        f"  breakdown: embed {human(b.embedding)} | attn {human(b.attention)} | "
        f"dense-ffn {human(b.dense_ffn)} | experts {human(b.moe_experts)}",
        f"  weights ({'bf16'}): {human_bytes(mem['weights_bytes'])} | "
        f"AdamW training state: {human_bytes(mem['train_total_bytes'])}",
        f"  fwd FLOPs/token: {human(flops_per_token(cfg))} | "
        f"Chinchilla tokens: {human(chinchilla_tokens(active))}",
    ]
    return "\n".join(lines)


def verify_counts(cfg: ModelConfig) -> tuple[int, int]:
    """Build the model and compare the analytic counts against the real thing."""
    from .model import MotherBrain  # imported lazily so this module stays torch-free

    model = MotherBrain(cfg)
    real_total = model.num_parameters()
    real_active = model.num_active_parameters()
    predicted_total = count_parameters(cfg).total
    predicted_active = count_active_parameters(cfg)
    if real_total != predicted_total:
        raise AssertionError(f"total mismatch: predicted {predicted_total}, got {real_total}")
    if real_active != predicted_active:
        raise AssertionError(f"active mismatch: predicted {predicted_active}, got {real_active}")
    return real_total, real_active


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------
# Each preset is the same architecture at a different scale. Everything up to
# `xl` is a dense model you can actually train; the `moe-*` tiers use sparsity
# to grow total parameters without growing the per-token compute.
PRESETS: dict[str, ModelConfig] = {
    # -- dense, trainable on ordinary hardware ---------------------------
    "nano": ModelConfig(vocab_size=8192, dim=128, n_layers=6, n_heads=4, n_kv_heads=2, max_seq_len=512),
    "micro": ModelConfig(vocab_size=16384, dim=256, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    "mini": ModelConfig(vocab_size=32000, dim=512, n_layers=12, n_heads=8, n_kv_heads=4, max_seq_len=2048),
    "small": ModelConfig(vocab_size=32000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048),
    "medium": ModelConfig(vocab_size=32000, dim=1024, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=4096),
    "large": ModelConfig(vocab_size=32000, dim=1536, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=4096),
    "xl": ModelConfig(vocab_size=32000, dim=2048, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=4096),
    # -- sparse: total parameters grow, compute per token does not -------
    "moe-small": ModelConfig(
        vocab_size=32000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048,
        n_experts=8, n_experts_per_tok=2, n_shared_experts=1, moe_first_dense_layers=1,
    ),
    "moe-large": ModelConfig(
        vocab_size=32000, dim=2048, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=4096,
        n_experts=64, n_experts_per_tok=4, n_shared_experts=1, moe_first_dense_layers=1,
    ),
    "moe-xl": ModelConfig(
        vocab_size=64000, dim=4096, n_layers=48, n_heads=32, n_kv_heads=8, max_seq_len=8192,
        n_experts=128, n_experts_per_tok=6, n_shared_experts=2, moe_first_dense_layers=2,
        moe_ffn_hidden=2048,
    ),
    # -- the top of the ladder: configurations no one can train today ----
    "titan": ModelConfig(
        vocab_size=128000, dim=8192, n_layers=96, n_heads=64, n_kv_heads=8, max_seq_len=16384,
        n_experts=512, n_experts_per_tok=8, n_shared_experts=2, moe_first_dense_layers=3,
        moe_ffn_hidden=4096,
    ),
    "motherbrain": ModelConfig(
        vocab_size=256000, dim=16384, n_layers=128, n_heads=128, n_kv_heads=16, max_seq_len=32768,
        n_experts=4096, n_experts_per_tok=8, n_shared_experts=2, moe_first_dense_layers=4,
        moe_ffn_hidden=8192,
    ),
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'. Available: {', '.join(PRESETS)}")
    # Return a copy so callers can mutate freely.
    return ModelConfig.from_dict(PRESETS[name].to_dict())


def ladder_table() -> str:
    rows = [f"{'preset':<14}{'total':>12}{'active':>12}{'active %':>10}{'bf16 weights':>16}"]
    rows.append("-" * len(rows[0]))
    for name, cfg in PRESETS.items():
        total = count_parameters(cfg).total
        active = count_active_parameters(cfg)
        mem = memory_estimate(cfg)["weights_bytes"]
        rows.append(
            f"{name:<14}{human(total):>12}{human(active):>12}"
            f"{active / total * 100:>9.1f}%{human_bytes(mem):>16}"
        )
    return "\n".join(rows)
