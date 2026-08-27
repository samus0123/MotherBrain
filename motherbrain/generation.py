"""Sampling text from a trained MotherBrain."""

from __future__ import annotations

import numpy as np

from .tokenizer import ByteTokenizer


def _filter_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= logits.size:
        return logits
    cutoff = np.partition(logits, -top_k)[-top_k]
    return np.where(logits < cutoff, -np.inf, logits)


def _filter_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    if not 0.0 < top_p < 1.0:
        return logits
    order = np.argsort(logits)[::-1]
    probs = _softmax(logits[order])
    keep = np.cumsum(probs) - probs < top_p  # always keeps the top token
    filtered = np.full_like(logits, -np.inf)
    filtered[order[keep]] = logits[order[keep]]
    return filtered


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def generate(
    model,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int | None = None,
    stop_at_eos: bool = True,
) -> str:
    """Continue ``prompt`` one token at a time.

    Context is trimmed to the model's trained window, so long prompts and
    long generations both stay inside what the architecture supports.
    """
    tokenizer = ByteTokenizer()
    rng = np.random.default_rng(seed)
    context_length = model.architecture.context_length

    ids = tokenizer.encode(prompt, bos=True)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        window = np.asarray([ids[-context_length:]], dtype=np.int32)
        logits = np.asarray(model(window, training=False))[0, -1].astype(np.float64)

        if temperature <= 0:
            next_id = int(np.argmax(logits))
        else:
            logits = logits / temperature
            logits = _filter_top_k(logits, top_k)
            logits = _filter_top_p(logits, top_p)
            next_id = int(rng.choice(len(logits), p=_softmax(logits)))

        if stop_at_eos and next_id == tokenizer.eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)
