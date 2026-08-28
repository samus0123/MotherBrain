"""The MotherBrain decoder-only transformer.

Architecture summary
--------------------
* Pre-norm residual blocks with RMSNorm.
* Rotary position embeddings (RoPE), with optional linear scaling for
  extrapolating past the trained context length.
* Grouped-query attention (GQA): `n_kv_heads` <= `n_heads` shrinks the KV cache,
  which is what actually bounds long-context inference.
* SwiGLU feed-forward networks.
* Optional mixture-of-experts FFNs with top-k routing, shared always-on experts,
  a load-balancing auxiliary loss and a router z-loss.
* A KV cache for incremental decoding.

The MoE path is the reason this repo can describe models with very large total
parameter counts: only `n_experts_per_tok` of `n_experts` run for any given
token, so total parameters grow with `n_experts` while the FLOPs per token do
not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# --------------------------------------------------------------------------
# Rotary position embeddings
# --------------------------------------------------------------------------
def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float() / scaling
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `x` of shape (B, n_heads, T, head_dim).

    `cos`/`sin` are (T, head_dim // 2) slices already aligned to the positions
    of `x`.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------
@dataclass
class KVCache:
    """Pre-allocated key/value cache for incremental decoding."""

    k: torch.Tensor  # (B, n_kv_heads, max_len, head_dim)
    v: torch.Tensor
    length: int = 0

    @classmethod
    def empty(
        cls,
        batch_size: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
        )

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        if self.length + t > self.k.shape[2]:
            raise ValueError(
                f"KV cache overflow: {self.length} + {t} > {self.k.shape[2]}. "
                "Increase max_seq_len or truncate the prompt."
            )
        self.k[:, :, self.length : self.length + t] = k
        self.v[:, :, self.length : self.length + t] = v
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self) -> None:
        self.length = 0


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.attn_dropout = cfg.attn_dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # With a cache holding T_past keys and T new queries, causal masking is
        # only correct when the queries are the *last* T positions. That is the
        # case both for a full prefill (T_past == 0) and for decoding steps
        # (T == 1), which is all we generate.
        is_causal = T > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.wo(y))


# --------------------------------------------------------------------------
# Feed-forward: dense SwiGLU and mixture-of-experts
# --------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ExpertBank(nn.Module):
    """`n_experts` SwiGLU MLPs stored as batched weight tensors."""

    def __init__(self, n_experts: int, dim: int, hidden: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.w1 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w3 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, hidden, dim))
        # These are raw Parameters rather than submodules, so nothing else will
        # initialise them. A bank built outside a MotherBrain must still be
        # usable on its own.
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float = 0.02) -> None:
        for p in (self.w1, self.w2, self.w3):
            nn.init.normal_(p, mean=0.0, std=init_std)

    def forward_subset(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        h = F.silu(x @ self.w1[expert_idx]) * (x @ self.w3[expert_idx])
        return h @ self.w2[expert_idx]


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of experts with optional shared experts.

    The routed experts are evaluated with a gather/scatter loop over experts:
    each expert sees only the tokens assigned to it, so the cost is
    proportional to `n_experts_per_tok`, not `n_experts`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        self.router_jitter = cfg.router_jitter

        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = ExpertBank(cfg.n_experts, cfg.dim, cfg.moe_ffn_hidden, cfg.init_std)
        self.shared = (
            SwiGLU(cfg.dim, cfg.moe_ffn_hidden * cfg.n_shared_experts, cfg.dropout)
            if cfg.n_shared_experts > 0
            else None
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Losses from the most recent forward pass, collected by the model.
        self.aux_loss: torch.Tensor | None = None
        self.router_z_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.view(-1, C)

        routed_in = flat
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(routed_in).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            routed_in = routed_in * noise

        logits = self.gate(routed_in).float()
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        self.aux_loss = self._load_balancing_loss(probs, topk_idx)
        self.router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        out = torch.zeros_like(flat)
        # (N * top_k,) flattened assignment lists
        flat_expert = topk_idx.reshape(-1)
        flat_weight = topk_probs.reshape(-1).to(flat.dtype)
        flat_token = torch.arange(flat.shape[0], device=x.device).repeat_interleave(self.top_k)

        for e in range(self.n_experts):
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tokens = flat_token[sel]
            y = self.experts.forward_subset(flat[tokens], e)
            out.index_add_(0, tokens, y * flat_weight[sel].unsqueeze(-1))

        out = out.view(B, T, C)
        if self.shared is not None:
            out = out + self.shared(x)
        return self.dropout(out)

    def _load_balancing_loss(self, probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load balancing: N * sum_i(frac_tokens_i * mean_prob_i)."""
        n_tokens = probs.shape[0]
        one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1).float()
        frac_tokens = one_hot.sum(dim=0) / (n_tokens * self.top_k)
        mean_prob = probs.mean(dim=0)
        return self.n_experts * torch.sum(frac_tokens * mean_prob)


# --------------------------------------------------------------------------
# Block
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.ffn = MoEFeedForward(cfg) if self.is_moe else SwiGLU(cfg.dim, cfg.ffn_hidden, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    router_z_loss: torch.Tensor | None = None


class MotherBrain(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.scale_residual_init:
            scale = 1.0 / math.sqrt(2 * cfg.n_layers)
            for name, p in self.named_parameters():
                if name.endswith(("attn.wo.weight", "ffn.w2.weight", "shared.w2.weight")):
                    with torch.no_grad():
                        p.mul_(scale)
                if name.endswith("experts.w2"):
                    with torch.no_grad():
                        p.mul_(scale)

        self.grad_checkpoint = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ExpertBank):
            for p in (module.w1, module.w2, module.w3):
                nn.init.normal_(p, mean=0.0, std=std)

    # -- introspection ----------------------------------------------------
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)

    def num_active_parameters(self) -> int:
        """Parameters touched by a single token (MoE-aware)."""
        total = 0
        for name, p in self.named_parameters():
            if ".experts.w" in name:
                per_expert = p.numel() // self.cfg.n_experts
                total += per_expert * self.cfg.n_experts_per_tok
            else:
                total += p.numel()
        if self.cfg.tie_embeddings:
            # lm_head shares storage with tok_emb; named_parameters() already
            # de-duplicates, so nothing to subtract here.
            pass
        return total

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.grad_checkpoint = enabled

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
        ignore_index: int = -100,
    ) -> ModelOutput:
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence position {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )

        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.emb_dropout(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache, use_reentrant=False
                )
            else:
                x = block(x, cos, sin, cache)
        x = self.norm(x)

        if targets is None:
            # Only the final position is needed to sample the next token.
            logits = self.lm_head(x[:, -1:, :])
        else:
            logits = self.lm_head(x)

        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        if targets is None:
            return ModelOutput(logits=logits)

        lm_loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        aux, zloss = self.collect_router_losses()
        loss = lm_loss
        if aux is not None:
            loss = loss + self.cfg.aux_loss_coef * aux + self.cfg.router_z_loss_coef * zloss
        return ModelOutput(
            logits=logits, loss=loss, lm_loss=lm_loss, aux_loss=aux, router_z_loss=zloss
        )

    def collect_router_losses(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        aux, zloss, n = None, None, 0
        for module in self.modules():
            if isinstance(module, MoEFeedForward) and module.aux_loss is not None:
                aux = module.aux_loss if aux is None else aux + module.aux_loss
                zloss = module.router_z_loss if zloss is None else zloss + module.router_z_loss
                n += 1
        if n == 0:
            return None, None
        return aux / n, zloss / n

    # -- caches -----------------------------------------------------------
    def make_caches(self, batch_size: int, max_len: int | None = None) -> list[KVCache]:
        max_len = max_len or self.cfg.max_seq_len
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return [
            KVCache.empty(
                batch_size, self.cfg.n_kv_heads, max_len, self.cfg.head_dim, device, dtype
            )
            for _ in range(self.cfg.n_layers)
        ]
