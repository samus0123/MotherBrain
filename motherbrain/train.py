"""Training loop: turn a corpus of tokens into learned weights."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from motherbrain.config import ModelConfig, human
from motherbrain.data import Corpus, TokenStream
from motherbrain.model import MotherBrain


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 8
    grad_accum: int = 1
    seq_len: int | None = None       # defaults to the model's context length
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 10
    log_every: int = 10
    save_every: int = 100
    seed: int = 1337
    device: str = "auto"
    compile: bool = False
    amp: bool = True


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup into a cosine decay."""
    if step < tc.warmup:
        return tc.lr * (step + 1) / max(tc.warmup, 1)
    progress = (step - tc.warmup) / max(tc.steps - tc.warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    floor = tc.lr * tc.min_lr_ratio
    return floor + 0.5 * (tc.lr - floor) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: MotherBrain, tc: TrainConfig) -> torch.optim.Optimizer:
    """Decay matrices, leave norms and biases alone."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), fused=fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))


@torch.no_grad()
def evaluate(model, stream, tc, device, split="val") -> float:
    model.eval()
    losses = []
    for _ in range(tc.eval_batches):
        x, y = stream.batch(tc.batch_size, split=split)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(path: Path, model: MotherBrain, opt, step: int,
                    cfg: ModelConfig, tc: TrainConfig, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "step": step,
        "config": cfg.to_dict(),
        "train_config": asdict(tc),
        "history": history,
    }, path)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[MotherBrain, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = MotherBrain(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def train(corpus_dir: str, out_dir: str, cfg: ModelConfig, tc: TrainConfig,
          resume: bool = False, progress_cb=None) -> dict:
    """Run a training job. Returns a summary dict."""
    torch.manual_seed(tc.seed)
    device = pick_device(tc.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(corpus_dir)
    tok = corpus.load_tokenizer()
    if tok.vocab_size != cfg.vocab_size:
        # The corpus is the authority on vocabulary size.
        cfg.vocab_size = tok.vocab_size

    seq_len = tc.seq_len or cfg.max_seq_len
    seq_len = min(seq_len, cfg.max_seq_len)
    stream = TokenStream(corpus.tokens_path, seq_len=seq_len, seed=tc.seed)

    ckpt_path = out / "checkpoint.pt"
    history: list = []
    start_step = 0

    if resume and ckpt_path.exists():
        model, ckpt = load_checkpoint(ckpt_path, device=device)
        cfg = model.cfg
        start_step = ckpt["step"]
        history = ckpt.get("history", [])
        opt = build_optimizer(model, tc)
        if ckpt.get("optimizer"):
            opt.load_state_dict(ckpt["optimizer"])
        print(f"resumed from {ckpt_path} at step {start_step}")
    else:
        model = MotherBrain(cfg).to(device)
        opt = build_optimizer(model, tc)

    model.train()
    cfg.save(str(out / "config.json"))

    print(f"device               {device}")
    print(f"total parameters     {human(model.n_params())} ({model.n_params():,})")
    print(f"active per token     {human(cfg.n_active_params)}")
    print(f"corpus               {len(stream):,} tokens "
          f"({stream.split_at:,} train / {stream.n_val:,} val)")
    tokens_per_step = tc.batch_size * tc.grad_accum * seq_len
    print(f"tokens/step          {tokens_per_step:,}")
    print(f"steps                {start_step} -> {tc.steps}")
    print("-" * 58, flush=True)

    if tc.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:  # pragma: no cover - backend dependent
            print(f"torch.compile unavailable ({exc}); continuing eager")

    use_amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else torch.amp.autocast("cpu", enabled=False))

    from motherbrain.patches import PatchStore, weights_fingerprint

    store = PatchStore(out)
    stamped = False

    def stamp_base(base_model) -> None:
        """Adopt these weights as the base as soon as they are written.

        Doing this only at the end of training would leave every intermediate
        checkpoint unloadable: the manifest would still describe the previous
        base, and the lineage guard would refuse the new weights. An
        interrupted run would produce a checkpoint nobody could open.
        """
        nonlocal stamped
        dropped = store.set_base(weights_fingerprint(base_model), corpus.n_documents)
        if dropped and not stamped:
            print(f"note: dropped {len(dropped)} patch(es) trained against the "
                  f"previous base checkpoint: {', '.join(dropped)}", flush=True)
        stamped = True

    t0 = time.time()
    last_loss = float("nan")
    best_val = min((h["val_loss"] for h in history), default=float("inf"))
    best_path = out / "best.pt"

    for step in range(start_step, tc.steps):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(tc.grad_accum):
            x, y = stream.batch(tc.batch_size, split="train")
            x, y = x.to(device), y.to(device)
            with autocast:
                _, loss = model(x, y)
            loss = loss / tc.grad_accum
            scaler.scale(loss).backward() if use_amp else loss.backward()
            accum_loss += loss.item()

        if tc.grad_clip:
            if use_amp:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)

        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        last_loss = accum_loss
        done = step + 1

        if done % tc.log_every == 0 or done == tc.steps:
            elapsed = time.time() - t0
            tps = tokens_per_step * (done - start_step) / max(elapsed, 1e-9)
            print(f"step {done:>6}/{tc.steps}  loss {accum_loss:.4f}  "
                  f"lr {lr:.2e}  {tps:,.0f} tok/s", flush=True)

        if done % tc.eval_every == 0 or done == tc.steps:
            val = evaluate(model, stream, tc, device)
            ppl = math.exp(min(val, 20))
            history.append({"step": done, "train_loss": accum_loss,
                            "val_loss": val, "perplexity": ppl})
            marker = ""
            if val < best_val:
                # Validation loss turns back up once a run starts overfitting,
                # and the rolling checkpoint would overwrite the best weights
                # with worse ones. Keep the best separately.
                best_val = val
                base = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(best_path, base, opt, done, cfg, tc, history)
                marker = "  <- best"
            print(f"           eval  val_loss {val:.4f}  perplexity {ppl:.2f}"
                  f"{marker}", flush=True)

        if progress_cb is not None:
            progress_cb({"step": done, "total": tc.steps, "loss": accum_loss, "lr": lr})

        if done % tc.save_every == 0 or done == tc.steps:
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(ckpt_path, base, opt, done, cfg, tc, history)
            tok.save(str(out / "tokenizer.json"))
            stamp_base(base)  # keep every saved checkpoint loadable

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(ckpt_path, base, opt, tc.steps, cfg, tc, history)
    tok.save(str(out / "tokenizer.json"))
    # Everything in the corpus right now lives in these weights, so later
    # patches start from here rather than relearning the whole corpus.
    stamp_base(base)

    if best_path.exists() and best_val < float("inf"):
        print(f"best validation loss {best_val:.4f} kept at {best_path}")

    summary = {
        "steps": tc.steps,
        "final_loss": last_loss,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "params": base.n_params(),
        "checkpoint": str(ckpt_path),
        "history": history,
        "elapsed_sec": time.time() - t0,
    }
    with open(out / "history.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("-" * 58)
    print(f"done in {summary['elapsed_sec']:.1f}s -> {ckpt_path}")
    return summary
