"""MotherBrain as an inference model: many prompts, one forward pass.

Training a model and serving one are different jobs and want different
code. `model.generate` is the training-time shape of it - one sequence,
one token at a time, yielded as it comes - and it is exactly right for a
person watching text appear. It is exactly wrong for anything else. Eight
callers on the board asking at once cost eight full generations, and a
file of a thousand prompts costs a thousand, when the arithmetic for all
of them would fit in the same matrices.

So this is the serving shape:

* **Batched decoding.** Prompts are left-padded to a common width and a
  mask keeps each row from attending to the padding beside it. RoPE needs
  no adjustment for that padding, because it encodes relative position:
  shifting a whole row by the same amount leaves every difference between
  its positions unchanged. Rows finish at different times and are held at
  their end token until the slowest is done.

* **Continuous batching.** `Engine` collects whatever requests arrive
  inside a short window and runs them together. On a board where callers
  arrive independently, that is the difference between the eighth caller
  waiting for seven generations and waiting for one.

* **Measurement.** `Throughput` reports prompt tokens, generated tokens
  and seconds, because "faster" without a number is a claim rather than a
  result.

The batched path is checked against the single path token for token under
a fixed seed. An inference engine that is quick and disagrees with the
model it is serving is not serving that model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Throughput:
    """What a run actually cost. Printed rather than asserted."""

    prompts: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    seconds: float = 0.0
    batches: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    def render(self) -> str:
        return (f"{self.prompts} prompt(s) in {self.batches} batch(es): "
                f"{self.prompt_tokens:,} prompt tokens, "
                f"{self.generated_tokens:,} generated in "
                f"{self.seconds:.2f}s "
                f"({self.tokens_per_second:.1f} tokens/s)")


@dataclass
class Request:
    """One prompt on its way through the engine."""

    prompt: str
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    text: str = ""
    tokens: list[int] = field(default_factory=list)


def _pack(tok, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a batch of prompts and say which positions are real.

    Left rather than right, because after the prefill every row's next token
    is at the same index - which is what lets one decode step serve the whole
    batch.
    """
    encoded = [tok.encode(p, bos=True) for p in prompts]
    width = max(len(e) for e in encoded)
    pad_id = 0
    idx = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    real = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, ids in enumerate(encoded):
        idx[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        real[row, width - len(ids):] = True
    return idx.to(device), real.to(device)


def _prefill_mask(real: torch.Tensor) -> torch.Tensor:
    """Causal, and blind to the padding. Shape (b, 1, t, t) for SDPA."""
    b, t = real.shape
    causal = torch.ones(t, t, dtype=torch.bool, device=real.device).tril()
    allowed = causal[None, :, :] & real[:, None, :]
    # A padding row attends to nothing, which SDPA turns into NaN. Let each
    # position see itself so the softmax has something to divide by; the
    # output there is discarded either way.
    eye = torch.eye(t, dtype=torch.bool, device=real.device)
    allowed = allowed | eye[None, :, :]
    return allowed[:, None, :, :]


def _decode_mask(real: torch.Tensor) -> torch.Tensor:
    """One query row against every key so far. Shape (b, 1, 1, kv)."""
    return real[:, None, None, :]


def _sample(logits: torch.Tensor, request_rows: list[Request],
            history: torch.Tensor) -> torch.Tensor:
    """Sample one token per row, honouring each row's own settings."""
    out = torch.empty((logits.shape[0], 1), dtype=torch.long,
                      device=logits.device)
    for row, req in enumerate(request_rows):
        line = logits[row].float()
        if req.repetition_penalty != 1.0:
            for seen in set(history[row].tolist()):
                line[seen] /= req.repetition_penalty
        if req.temperature <= 0:
            out[row, 0] = line.argmax()
            continue
        line = line / req.temperature
        if req.top_k:
            kth = torch.topk(line, min(req.top_k, line.size(-1)))[0][-1]
            line = line.masked_fill(line < kth, float("-inf"))
        if req.top_p and req.top_p < 1.0:
            ordered, order = torch.sort(line, descending=True)
            probs = F.softmax(ordered, dim=-1)
            drop = torch.cumsum(probs, dim=-1) - probs > req.top_p
            ordered = ordered.masked_fill(drop, float("-inf"))
            line = torch.empty_like(line).scatter_(-1, order, ordered)
        out[row, 0] = torch.multinomial(F.softmax(line, dim=-1), 1)
    return out


@torch.no_grad()
def generate_batch(model, tok, requests: list[Request], device,
                   eos_id: int | None = None,
                   images: torch.Tensor | None = None) -> Throughput:
    """Run a whole batch to completion, filling in each request's `text`.

    Every row shares the decode steps. A row that hits its end token or its
    own token budget stops contributing and is simply not written to again;
    the batch ends when every row has.
    """
    if not requests:
        return Throughput()

    model.eval()
    started = time.perf_counter()
    idx, real = _pack(tok, [r.prompt for r in requests], device)
    prompt_tokens = int(real.sum().item())

    # An image becomes a run of vectors in front of the text, so the mask has
    # to cover them too. They are context for every row, never padding.
    visual = _visual(model, images)
    if visual:
        real = torch.cat([torch.ones((len(requests), visual), dtype=torch.bool,
                                     device=device), real], dim=1)

    caches = [{} for _ in model.blocks]
    logits, _ = model(idx, caches=caches, offset=0, images=images,
                      mask=_prefill_mask(real))
    keys = real.clone()

    live = [True] * len(requests)
    history = idx
    budget = max(r.max_new_tokens for r in requests)
    generated = 0

    for step in range(budget):
        nxt = _sample(logits[:, -1, :], requests, history)
        for row, req in enumerate(requests):
            if not live[row]:
                continue
            token = int(nxt[row, 0])
            if eos_id is not None and token == eos_id:
                live[row] = False
                continue
            if len(req.tokens) >= req.max_new_tokens:
                live[row] = False
                continue
            req.tokens.append(token)
            generated += 1
        if not any(live):
            break

        history = torch.cat([history, nxt], dim=1)
        keys = torch.cat(
            [keys, torch.ones((len(requests), 1), dtype=torch.bool,
                              device=device)], dim=1)
        logits, _ = model(nxt, caches=caches,
                          offset=idx.shape[1] + visual + step,
                          mask=_decode_mask(keys))

    for req in requests:
        req.text = tok.decode(req.tokens)
    return Throughput(prompts=len(requests), prompt_tokens=prompt_tokens,
                      generated_tokens=generated,
                      seconds=time.perf_counter() - started, batches=1)


def _visual(model, images) -> int:
    if images is None or getattr(model, "vision", None) is None:
        return 0
    return model.vision.n_tokens


def run(model, tok, prompts: list[str], device, batch_size: int = 8,
        max_new_tokens: int = 120, temperature: float = 0.8,
        top_k: int | None = 40, top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None) -> tuple[list[str], Throughput]:
    """Every prompt, in batches, with one throughput figure for the lot."""
    out: list[str] = []
    total = Throughput()
    for start in range(0, len(prompts), batch_size):
        chunk = [Request(p, max_new_tokens, temperature, top_k, top_p,
                         repetition_penalty)
                 for p in prompts[start:start + batch_size]]
        measured = generate_batch(model, tok, chunk, device, eos_id=eos_id)
        out.extend(r.text for r in chunk)
        total.prompts += measured.prompts
        total.prompt_tokens += measured.prompt_tokens
        total.generated_tokens += measured.generated_tokens
        total.seconds += measured.seconds
        total.batches += 1
    return out, total


# ---- serving many callers ---------------------------------------------------

class Engine:
    """Continuous batching for an asyncio server.

    Requests that arrive close together are run together. The window is
    short - a caller should not wait to be batched with someone who has not
    typed anything yet - but on a board where people arrive independently
    it is the difference between the eighth caller waiting for seven
    generations and waiting for one.
    """

    def __init__(self, model, tok, device, eos_id: int | None = None,
                 max_batch: int = 8, window: float = 0.05,
                 adaptive: bool = True) -> None:
        import asyncio

        self.model, self.tok, self.device = model, tok, device
        self.eos_id = eos_id
        self.max_batch, self.window = max_batch, window
        # How long to hold the door open for the next caller. A fixed 50ms is
        # right when a generation takes 50ms and useless when it takes
        # thirteen seconds - callers arriving a quarter-second apart each get
        # their own batch. So it tracks the last batch: waiting 2% of a
        # generation to halve the number of generations is a trade worth
        # making, and on a fast device it stays near zero by itself.
        self.adaptive = adaptive
        self.max_window = 0.5
        self.last_seconds = 0.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.served = Throughput()
        self._worker = None

    def start(self) -> None:
        import asyncio

        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    @property
    def waiting(self) -> int:
        return self.queue.qsize()

    async def submit(self, request: Request) -> str:
        """Queue a prompt and wait for its own completion, not the batch's."""
        import asyncio

        self.start()
        done = asyncio.get_running_loop().create_future()
        await self.queue.put((request, done))
        return await done

    async def _loop(self) -> None:
        import asyncio

        while True:
            first = await self.queue.get()
            batch = [first]
            wait = self.window
            if self.adaptive:
                wait = max(wait, min(self.max_window,
                                     0.02 * self.last_seconds))
            deadline = asyncio.get_running_loop().time() + wait
            while len(batch) < self.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(),
                                                        remaining))
                except asyncio.TimeoutError:
                    break

            requests = [r for r, _ in batch]
            try:
                measured = await asyncio.to_thread(
                    generate_batch, self.model, self.tok, requests,
                    self.device, self.eos_id)
                self.served.prompts += measured.prompts
                self.served.prompt_tokens += measured.prompt_tokens
                self.served.generated_tokens += measured.generated_tokens
                self.served.seconds += measured.seconds
                self.served.batches += 1
                self.last_seconds = measured.seconds
                for request, future in batch:
                    if not future.done():
                        future.set_result(request.text)
            except asyncio.CancelledError:
                for _, future in batch:
                    if not future.done():
                        future.cancel()
                raise
            except Exception as exc:                      # noqa: BLE001
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
