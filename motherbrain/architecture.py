"""Model architecture and its parameter count.

The parameter count is derived from the architecture rather than stored, so
it can never drift out of sync with the dimensions it describes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Architecture:
    """Shape of a decoder-only transformer.

    Defaults follow a modern stack: rotary positions (no learned position
    table), RMSNorm (no norm biases), no linear biases, and a gated MLP.
    """

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    context_length: int
    gated_mlp: bool = True
    tie_embeddings: bool = True
    learned_positions: bool = False

    def __post_init__(self) -> None:
        positive = (
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "vocab_size",
            "context_length",
        )
        for field in positive:
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters implied by these dimensions."""
        d = self.d_model
        total = self.vocab_size * d  # token embeddings
        if self.learned_positions:
            total += self.context_length * d
        if not self.tie_embeddings:
            total += self.vocab_size * d  # separate output head

        per_layer = 4 * d * d  # q, k, v, o projections
        per_layer += (3 if self.gated_mlp else 2) * d * self.d_ff
        per_layer += 2 * d  # two RMSNorms
        total += self.n_layers * per_layer

        total += d  # final RMSNorm
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Architecture":
        fields = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown architecture keys: {sorted(unknown)}")
        return cls(**data)
