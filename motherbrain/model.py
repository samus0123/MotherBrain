"""The MotherBrain network: a decoder-only transformer in Keras/TensorFlow.

The layer set is deliberately the same one the parameter formula in
:mod:`motherbrain.architecture` assumes -- RMSNorm, rotary positions, a
gated (SwiGLU) MLP, no biases anywhere, tied embeddings -- so the built
model's weight count matches ``Architecture.parameter_count`` exactly.
"""

from __future__ import annotations

import keras
from keras import layers, ops

from .architecture import Architecture

ROPE_BASE = 10_000.0


class RMSNorm(layers.Layer):
    """Root-mean-square normalisation: one scale vector, no bias."""

    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape) -> None:
        self.scale = self.add_weight(
            name="scale", shape=(input_shape[-1],), initializer="ones"
        )
        super().build(input_shape)

    def call(self, x):
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        return x * ops.rsqrt(variance + self.epsilon) * self.scale


def _rope_tables(seq_len: int, head_dim: int, dtype: str):
    """Cosine/sine tables for rotary position embeddings (no parameters)."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        ROPE_BASE ** (ops.cast(ops.arange(0, half), dtype) * 2.0 / head_dim)
    )
    positions = ops.cast(ops.arange(0, seq_len), dtype)
    angles = positions[:, None] * inv_freq[None, :]  # (seq, half)
    return ops.cos(angles), ops.sin(angles)


def _apply_rope(x, cos, sin):
    """Rotate query/key pairs. ``x`` is (batch, heads, seq, head_dim)."""
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    stacked = ops.stack([rotated_even, rotated_odd], axis=-1)
    return ops.reshape(stacked, ops.shape(x))


class CausalSelfAttention(layers.Layer):
    """Multi-head causal attention with rotary positions."""

    def __init__(self, d_model: int, n_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = layers.Dense(d_model, use_bias=False, name="q")
        self.k = layers.Dense(d_model, use_bias=False, name="k")
        self.v = layers.Dense(d_model, use_bias=False, name="v")
        self.o = layers.Dense(d_model, use_bias=False, name="o")

    def _split_heads(self, x, batch, seq):
        x = ops.reshape(x, (batch, seq, self.n_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, x):
        shape = ops.shape(x)
        batch, seq = shape[0], shape[1]

        q = self._split_heads(self.q(x), batch, seq)
        k = self._split_heads(self.k(x), batch, seq)
        v = self._split_heads(self.v(x), batch, seq)

        cos, sin = _rope_tables(seq, self.head_dim, x.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, x.dtype))

        positions = ops.arange(0, seq)
        causal = positions[None, :] <= positions[:, None]  # (seq, seq)
        scores = ops.where(causal[None, None, :, :], scores, float("-inf"))

        weights = ops.softmax(scores, axis=-1)
        context = ops.matmul(weights, v)
        context = ops.transpose(context, (0, 2, 1, 3))
        context = ops.reshape(context, (batch, seq, self.d_model))
        return self.o(context)


class SwiGLU(layers.Layer):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ff: int, gated: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gated = gated
        self.up = layers.Dense(d_ff, use_bias=False, name="up")
        self.gate = (
            layers.Dense(d_ff, use_bias=False, name="gate") if gated else None
        )
        self.down = layers.Dense(d_model, use_bias=False, name="down")

    def call(self, x):
        hidden = self.up(x)
        hidden = (
            ops.silu(self.gate(x)) * hidden if self.gated else ops.silu(hidden)
        )
        return self.down(hidden)


class TransformerBlock(layers.Layer):
    """Pre-norm attention and MLP, each with a residual connection."""

    def __init__(self, arch: Architecture, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention_norm = RMSNorm(name="attention_norm")
        self.attention = CausalSelfAttention(arch.d_model, arch.n_heads)
        self.mlp_norm = RMSNorm(name="mlp_norm")
        self.mlp = SwiGLU(arch.d_model, arch.d_ff, gated=arch.gated_mlp)

    def call(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class MotherBrain(keras.Model):
    """A decoder-only language model built from an :class:`Architecture`."""

    def __init__(self, architecture: Architecture, **kwargs) -> None:
        super().__init__(**kwargs)
        self.architecture = architecture
        self.embedding = layers.Embedding(
            architecture.vocab_size, architecture.d_model, name="embedding"
        )
        self.blocks = [
            TransformerBlock(architecture, name=f"block_{i}")
            for i in range(architecture.n_layers)
        ]
        self.final_norm = RMSNorm(name="final_norm")
        self.head = (
            None
            if architecture.tie_embeddings
            else layers.Dense(
                architecture.vocab_size, use_bias=False, name="head"
            )
        )

    def call(self, token_ids):
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if self.head is not None:
            return self.head(x)
        # Tied embeddings: reuse the embedding matrix as the output head.
        return ops.matmul(x, ops.transpose(self.embedding.embeddings))


def build_model(architecture: Architecture) -> MotherBrain:
    """Instantiate and build a model so its weights exist and can be counted."""
    model = MotherBrain(architecture)
    # A forward pass materialises every lazily-built sublayer.
    model(ops.zeros((1, 8), dtype="int32"))
    return model
