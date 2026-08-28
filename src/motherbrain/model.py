"""A Llama-style decoder-only transformer.

Pre-norm blocks with RMSNorm, rotary position embeddings, grouped-query
attention on top of PyTorch's fused SDPA kernels, and a SwiGLU MLP. No biases
anywhere; embeddings are optionally tied to the output head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["RMSNorm", "Attention", "MLP", "Block", "Transformer", "KVCache"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability, then cast back to the activation dtype.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotary cos/sin tables, each of shape ``[seq_len, head_dim]``."""
    if head_dim % 2 != 0:
        raise ValueError(f"rotary head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[B, n_head, T, head_dim]`` by the given tables."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, n_kv_head, T, hd]`` to ``[B, n_kv_head * n_rep, T, hd]`` for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


@dataclass
class KVCache:
    """A pre-allocated key/value cache for one attention layer."""

    k: torch.Tensor  # [B, n_kv_head, max_seq_len, head_dim]
    v: torch.Tensor
    length: int = 0

    @classmethod
    def empty(
        cls,
        batch_size: int,
        n_kv_head: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        shape = (batch_size, n_kv_head, max_seq_len, head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
        )

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` at ``start_pos`` and return the cache up to the new end."""
        end = start_pos + k.shape[2]
        if end > self.k.shape[2]:
            raise ValueError(
                f"KV cache overflow: writing up to position {end} but capacity is {self.k.shape[2]}"
            )
        self.k[:, :, start_pos:end] = k
        self.v[:, :, start_pos:end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self) -> None:
        self.length = 0


class Attention(nn.Module):
    """Causal self-attention with rotary embeddings and grouped-query heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_kv_head is not None
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v, start_pos)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # With no cache, queries and keys line up and the fused causal kernel applies.
        # With a cache the query block sits at offset `start_pos`, so build the mask.
        attn_mask = None
        is_causal = cache is None
        if cache is not None and t > 1:
            k_len = k.shape[2]
            q_idx = torch.arange(start_pos, start_pos + t, device=x.device)[:, None]
            k_idx = torch.arange(k_len, device=x.device)[None, :]
            attn_mask = k_idx <= q_idx

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(out))


class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_ff is not None
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache, start_pos=start_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    """The full decoder-only language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale the residual-path output projections so activation variance stays
        # roughly constant with depth (GPT-2 style).
        residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count; by default excludes the token embedding table."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            # An untied head is a separate table and also does not count as "body".
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)``; ``loss`` is ``None`` when no targets are given."""
        _, t = idx.shape
        end = start_pos + t
        if end > self.cfg.max_seq_len:
            raise ValueError(f"sequence position {end} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos[start_pos:end]
        sin = self.rope_sin[start_pos:end]

        x = self.emb_dropout(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, cos, sin, cache=cache, start_pos=start_pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    def init_caches(self, batch_size: int, max_seq_len: int | None = None) -> list[KVCache]:
        """Allocate one :class:`KVCache` per layer, on the model's device/dtype."""
        assert self.cfg.n_kv_head is not None
        max_seq_len = max_seq_len or self.cfg.max_seq_len
        param = next(self.parameters())
        return [
            KVCache.empty(
                batch_size,
                self.cfg.n_kv_head,
                max_seq_len,
                self.cfg.head_dim,
                param.device,
                param.dtype,
            )
            for _ in range(self.cfg.n_layer)
        ]

    def flops_per_token(self) -> float:
        """Approximate forward+backward FLOPs per token (PaLM appendix B)."""
        cfg = self.cfg
        n = self.num_params(non_embedding=True)
        attn_flops = 12 * cfg.n_layer * cfg.n_head * cfg.head_dim * cfg.max_seq_len
        return 6 * n + attn_flops

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` ``[B, T]`` using a KV cache."""
        was_training = self.training
        self.eval()
        try:
            b, t = idx.shape
            budget = min(self.cfg.max_seq_len, t + max_new_tokens)
            if t >= budget:
                raise ValueError("prompt is already at or beyond max_seq_len")
            caches = self.init_caches(b, budget)

            # Prefill on the prompt, then decode one token at a time.
            logits, _ = self.forward(idx, caches=caches, start_pos=0)
            pos = t
            done = torch.zeros(b, dtype=torch.bool, device=idx.device)
            out = idx

            for _ in range(budget - t):
                next_token = self._sample_next(
                    logits[:, -1, :], temperature, top_k, top_p, generator
                )
                if eos_id is not None:
                    # Once a row has emitted EOS, keep padding it with EOS.
                    next_token = torch.where(
                        done[:, None], torch.full_like(next_token, eos_id), next_token
                    )
                    done |= next_token.squeeze(-1) == eos_id
                out = torch.cat((out, next_token), dim=1)
                if eos_id is not None and bool(done.all()):
                    break
                logits, _ = self.forward(next_token, caches=caches, start_pos=pos)
                pos += 1
            return out
        finally:
            self.train(was_training)

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        logits = logits.float()
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Keep the smallest prefix whose mass exceeds top_p (always >= 1 token).
            remove = probs - F.softmax(sorted_logits, dim=-1) >= top_p
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)
