"""Text generation with a KV cache."""

from __future__ import annotations

from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from .model import MotherBrain


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style penalty: divide positive logits of seen tokens, multiply negative ones."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for b in range(logits.shape[0]):
        seen = torch.unique(generated[b])
        scores = logits[b, seen]
        logits[b, seen] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def filter_top_k_top_p(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Mask out tokens outside the top-k / nucleus, in-place on a copy."""
    logits = logits.clone()
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep every token up to and including the one that crosses top_p.
        remove = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: MotherBrain,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    on_token: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Autoregressively extend `prompt_ids` (B, T). Returns (B, T + new)."""
    model.eval()
    device = next(model.parameters()).device
    idx = prompt_ids.to(device)
    if idx.dim() == 1:
        idx = idx[None, :]
    B, T = idx.shape

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    max_len = min(model.cfg.max_seq_len, T + max_new_tokens)
    caches = model.make_caches(B, max_len) if use_cache else None

    if use_cache:
        out = model(idx, caches=caches, start_pos=0)
        next_logits = out.logits[:, -1, :]
        pos = T
    else:
        pos = T

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        if not use_cache:
            window = idx[:, -model.cfg.max_seq_len :]
            next_logits = model(window).logits[:, -1, :]
        elif step > 0:
            out = model(idx[:, -1:], caches=caches, start_pos=pos - 1)
            next_logits = out.logits[:, -1, :]

        logits = next_logits.float()
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)

        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = filter_top_k_top_p(logits / temperature, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)

        if eos_id is not None:
            # Once a sequence emits EOS, keep padding it with EOS.
            next_id = torch.where(finished[:, None], torch.full_like(next_id, eos_id), next_id)
            finished = finished | (next_id.squeeze(-1) == eos_id)

        idx = torch.cat([idx, next_id], dim=1)
        pos += 1
        if on_token is not None:
            on_token(next_id)
        if eos_id is not None and bool(finished.all()):
            break
        if pos >= model.cfg.max_seq_len:
            break

    return idx


@torch.no_grad()
def stream(model: MotherBrain, prompt_ids: torch.Tensor, tokenizer, **kwargs) -> Iterator[str]:
    """Yield decoded text incrementally. Single sequence only."""
    pieces: list[int] = []
    buffered: list[int] = []

    def collect(tok: torch.Tensor) -> None:
        pieces.append(int(tok[0, 0]))

    generate(model, prompt_ids, on_token=collect, **kwargs)
    # Decode incrementally, holding back bytes that do not yet form a character.
    emitted = 0
    for i in range(len(pieces)):
        text = tokenizer.decode(pieces[: i + 1])
        if not text.endswith("�"):
            yield text[emitted:]
            emitted = len(text)
