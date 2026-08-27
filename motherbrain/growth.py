"""How MotherBrain grows when its version increases.

Every bump scales the architecture up, so the parameter count strictly
increases at every step. Dimensions are only ever grown, never shrunk, and
the result is checked before it is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .architecture import Architecture
from .version import BumpKind

#: Dimensions are kept on this grid so they stay hardware-friendly.
ALIGNMENT = 64


def _grow_to_grid(value: int, factor: float, alignment: int = ALIGNMENT) -> int:
    """Scale ``value`` by ``factor`` and round up onto the alignment grid.

    Rounding up guarantees the result exceeds ``value`` whenever
    ``factor > 1``, which is what keeps growth strictly monotonic.
    """
    if factor < 1.0:
        raise ValueError("growth factors must be >= 1.0")
    return max(alignment, int(math.ceil(value * factor / alignment)) * alignment)


def _heads_for(d_model: int, head_dim: int, min_heads: int) -> int:
    """Pick a head count that divides ``d_model`` and keeps ``head_dim`` close.

    Attention heads never shrink, and ``d_model`` must stay divisible by the
    head count, so we take the divisor nearest the ideal that satisfies both.
    """
    divisors = [h for h in range(1, d_model + 1) if d_model % h == 0]
    ideal = max(1, d_model // head_dim)
    allowed = [h for h in divisors if h >= min_heads] or divisors
    return min(allowed, key=lambda h: (abs(h - ideal), h))


@dataclass(frozen=True)
class GrowthStep:
    """Scaling applied by one bump of a given kind."""

    width: float = 1.0  # multiplier on d_model (and d_ff, to hold the ratio)
    depth: float = 1.0  # multiplier on n_layers
    depth_add: int = 0  # layers added after the depth multiplier
    ffn: float = 1.0  # extra multiplier on d_ff alone
    context: float = 1.0  # multiplier on context_length


@dataclass(frozen=True)
class GrowthPolicy:
    """The growth applied for each kind of version bump.

    A larger bump means a larger jump in capacity: a patch widens the MLP a
    little, a minor release widens the residual stream and adds a layer, and
    a major release is a genuine scale-up in width, depth and context.
    """

    major: GrowthStep = GrowthStep(width=1.5, depth=2.0, context=2.0)
    minor: GrowthStep = GrowthStep(width=1.10, depth_add=1)
    patch: GrowthStep = GrowthStep(ffn=1.06)

    def step_for(self, kind: BumpKind) -> GrowthStep:
        try:
            return getattr(self, kind)
        except AttributeError:
            raise ValueError(f"unknown bump kind: {kind!r}") from None

    def grow(self, arch: Architecture, kind: BumpKind) -> Architecture:
        """Return the architecture one bump of ``kind`` above ``arch``."""
        step = self.step_for(kind)

        d_model = _grow_to_grid(arch.d_model, step.width)
        # Hold the FFN ratio across width changes, then apply any extra
        # FFN-only growth on top of it.
        ffn_factor = (d_model / arch.d_model) * step.ffn
        d_ff = _grow_to_grid(arch.d_ff, ffn_factor)
        n_layers = int(math.ceil(arch.n_layers * step.depth)) + step.depth_add
        context_length = _grow_to_grid(arch.context_length, step.context)
        # Keep head_dim roughly fixed so heads scale with the width.
        n_heads = _heads_for(d_model, arch.head_dim, arch.n_heads)

        grown = Architecture(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=arch.vocab_size,
            context_length=context_length,
            gated_mlp=arch.gated_mlp,
            tie_embeddings=arch.tie_embeddings,
            learned_positions=arch.learned_positions,
        )

        if grown.parameter_count <= arch.parameter_count:
            raise ValueError(
                f"{kind} bump did not grow the model: "
                f"{arch.parameter_count} -> {grown.parameter_count}"
            )
        return grown
