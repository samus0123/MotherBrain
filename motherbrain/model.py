"""The MotherBrain transformer.

A decoder-only transformer with the pieces that current frontier models use:
RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU feed-
forwards, and sparse mixture-of-experts routing. The MoE layers are what let
the parameter count grow without the per-token compute growing with it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from motherbrain.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute the cos/sin tables for rotary embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the query/key pairs. x is (B, n_heads, T, head_dim)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped key/value heads to match the number of query heads."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class Attention(nn.Module):
    """Causal self-attention with grouped-query heads and a KV cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: dict | None = None):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # With a cache the query is short and the keys are long, so the plain
        # causal mask no longer lines up; only mask when we have a full block.
        is_causal = t > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """A single feed-forward block, and also the unit an 'expert' is made of."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Sparse mixture of experts with top-k routing.

    Every token is scored against `n_experts` routers, sent to its best `k`,
    and the results are recombined weighted by the router's confidence. Only
    those `k` experts do any work, which is why total parameters and per-token
    cost come apart here.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_token
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_shared_experts)]
        )
        # Added to the router logits. Zero for a model trained from scratch; an
        # expert appended later starts strongly negative here so it cannot be
        # selected, which is what lets the model grow without its output
        # changing. The patch trainer releases the new experts when it starts.
        self.expert_bias = nn.Parameter(torch.zeros(cfg.n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.view(-1, d)

        logits = self.router(flat).float() + self.expert_bias.float()
        probs = F.softmax(logits, dim=-1)
        weights, idx = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(x.dtype)

        out = torch.zeros_like(flat)
        # One pass per expert, gathering only the tokens routed to it.
        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = (idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            contribution = expert(flat[token_idx])
            out.index_add_(
                0, token_idx, contribution * weights[token_idx, slot_idx].unsqueeze(-1)
            )

        for expert in self.shared:
            out = out + expert(flat)

        self.aux_loss = self._balance_loss(logits, probs, idx)
        return out.view(b, t, d)

    def _balance_loss(self, logits, probs, idx) -> torch.Tensor:
        """Keep the router from collapsing onto a favourite expert.

        `frac` is the share of tokens each expert actually received and `mean_p`
        the share of probability mass it was assigned; their dot product is
        minimised when both are uniform. The z-loss keeps router logits from
        drifting to large magnitudes.
        """
        cfg = self.cfg
        n_tok = logits.shape[0]
        counts = torch.zeros(self.n_experts, device=logits.device, dtype=logits.dtype)
        counts.index_add_(
            0, idx.reshape(-1), torch.ones(idx.numel(), device=logits.device, dtype=logits.dtype)
        )
        frac = counts / (n_tok * self.top_k)
        mean_p = probs.mean(dim=0)
        balance = self.n_experts * torch.sum(frac * mean_p)
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return cfg.router_aux_loss_coef * balance + cfg.router_z_loss_coef * z


class Block(nn.Module):
    """Pre-norm transformer block: attention, then FFN (dense or MoE)."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


class MotherBrain(nn.Module):
    """The model itself."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Sight is optional and additive: with vision_layers == 0 there is no
        # tower, no extra parameters, and the forward pass is what it was.
        self.vision = None
        if cfg.vision_layers > 0:
            from motherbrain.vision import VisionTower

            self.vision = VisionTower(cfg)

        self.apply(self._init_weights)
        # Scale down the residual-path projections so activations don't grow
        # with depth (the GPT-2 initialisation trick).
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

        self._rope_cache: tuple | None = None

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _rope(self, seq_len: int, offset: int, device, dtype):
        need = offset + seq_len
        if self._rope_cache is None or self._rope_cache[0] < need or \
                self._rope_cache[1].device != device or self._rope_cache[1].dtype != dtype:
            length = max(need, self.cfg.max_seq_len)
            cos, sin = build_rope_cache(length, self.cfg.head_dim, self.cfg.rope_theta,
                                        device, dtype)
            self._rope_cache = (length, cos, sin)
        _, cos, sin = self._rope_cache
        return cos[offset:offset + seq_len], sin[offset:offset + seq_len]

    # ---- parameter accounting ---------------------------------------------

    def n_params(self, non_embedding: bool = False) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:  # tied weights are one tensor, counted once
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    # ---- forward -----------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                caches: list | None = None, offset: int = 0,
                images: torch.Tensor | None = None):
        """Run the model. `images` prepends visual tokens to the sequence.

        Once an image has been through the vision tower it is a run of vectors
        in the model's own dimensions, indistinguishable from word embeddings,
        and the transformer treats it as text that happens to come first.
        """
        b, t = idx.shape
        x = self.embed(idx)

        n_visual = 0
        if images is not None:
            if self.vision is None:
                raise ValueError(
                    "this model has no vision tower; build it with "
                    "vision_layers > 0")
            visual = self.vision(images.to(x.dtype))
            if visual.shape[0] != b:
                visual = visual.expand(b, -1, -1)
            n_visual = visual.shape[1]
            x = torch.cat([visual, x], dim=1)

        cos, sin = self._rope(x.shape[1], offset, x.device, torch.float32)

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        x = self.norm(x)

        # The visual positions were context, not something to predict.
        if n_visual:
            x = x[:, n_visual:, :]

        if targets is None:
            # Only the last position matters when sampling.
            logits = self.lm_head(x[:, -1:, :])
            return logits, None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        aux = self.aux_loss()
        return logits, loss + aux

    def aux_loss(self) -> torch.Tensor:
        """Sum the router losses recorded during the last forward pass."""
        total = None
        for block in self.blocks:
            if block.is_moe:
                al = block.ffn.aux_loss
                total = al if total is None else total + al
        if total is None:
            return torch.zeros((), device=self.embed.weight.device)
        return total

    @torch.no_grad()
    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Final-layer representations, (B, T, d_model).

        Used for embeddings: an IDE asking for a vector per file or symbol gets
        the model's own view of that text rather than a bag of words.
        """
        self.eval()
        x = self.embed(idx)
        cos, sin = self._rope(idx.shape[1], 0, x.device, torch.float32)
        for block in self.blocks:
            x = block(x, cos, sin, None)
        return self.norm(x)

    @torch.no_grad()
    def embed_text(self, idx: torch.Tensor) -> torch.Tensor:
        """One L2-normalised vector per sequence, mean-pooled over positions."""
        h = self.hidden_states(idx).float().mean(dim=1)
        return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int | None = 40,
                 top_p: float | None = 0.95, repetition_penalty: float = 1.0,
                 eos_id: int | None = None, use_cache: bool = True,
                 images: torch.Tensor | None = None):
        """Autoregressive sampling. Yields one token id at a time.

        An image is encoded once, on the first pass, and lives in the KV cache
        from then on - re-encoding it for every token would be wasted work.
        """
        self.eval()
        # Caching and images interact, and the first pass must carry both.
        caches = [{} for _ in self.blocks] if use_cache else None
        offset = 0
        cur = idx

        for _ in range(max_new_tokens):
            window = cur if cur.shape[1] <= self.cfg.max_seq_len \
                else cur[:, -self.cfg.max_seq_len:]
            if caches is not None and offset > 0:
                window = cur[:, -1:]
            first = offset == 0
            logits, _ = self(window, caches=caches, offset=offset,
                             images=images if first else None)
            offset += window.shape[1] + (
                self.vision.n_tokens if (first and images is not None) else 0)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for tok in set(cur[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p and top_p < 1.0:
                    ordered, order = torch.sort(logits, descending=True, dim=-1)
                    cum = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
                    drop = cum - F.softmax(ordered, dim=-1) > top_p
                    ordered = ordered.masked_fill(drop, float("-inf"))
                    logits = torch.empty_like(logits).scatter_(-1, order, ordered)
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            cur = torch.cat([cur, nxt], dim=1)
            token = int(nxt.item())
            yield token
            if eos_id is not None and token == eos_id:
                break


def build_model(cfg: ModelConfig) -> MotherBrain:
    return MotherBrain(cfg)
