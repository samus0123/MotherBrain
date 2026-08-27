"""Incremental learning: information becomes a patch, a patch becomes a version.

Feeding MotherBrain something new does not retrain it from scratch. The new
information is trained into a small low-rank delta — a *patch* — while the
existing weights stay frozen. Applying that patch produces the next sequential
version of the model:

    v0  the base checkpoint from `mb train`
    v1  = v0 + patch-0001   (the release notes you fed it)
    v2  = v1 + patch-0002   (the API docs you fed it next)

Patches are small (megabytes, not gigabytes), they stack in order, and any
earlier version can be rebuilt exactly by replaying the base plus the first N
patches. Nothing is overwritten, so `mb checkout v1` is always available.

Two details make this work rather than merely run:

* Replay. Training only on new text makes a model forget the old text. Each
  patch trains on a mixture of the new information and a sample of everything
  learned before it, which is what keeps v7 still fluent in what v1 taught.
* A frozen vocabulary. The tokenizer is byte-level, so new information can use
  words the base corpus never contained without any vocabulary surgery.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from motherbrain.data import TOKEN_DTYPE, Corpus

# Weights a patch is allowed to touch. Expert matrices are excluded by default:
# in a large MoE model they dominate the parameter count, and a patch that
# covered them would stop being small.
ATTN_TARGETS = ("wq", "wk", "wv", "wo")
FFN_TARGETS = ("gate", "up", "down")


@dataclass
class PatchConfig:
    rank: int = 8
    alpha: float = 16.0
    steps: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    replay_ratio: float = 0.25   # share of each batch drawn from older material
    seq_len: int | None = None
    include_ffn: bool = True
    include_experts: bool = False
    seed: int = 1337


def weights_fingerprint(model: nn.Module) -> str:
    """A stable identity for a set of base weights.

    A patch is a delta against particular weights; applied to different ones it
    produces confident nonsense rather than an error. Hashing the parameter
    names, shapes and a bounded sample of their values gives every base
    checkpoint an identity, cheaply enough that it stays practical for a large
    model.
    """
    h = hashlib.sha256()
    budget = 1 << 20  # a megabyte of actual values is plenty to disambiguate
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(str(p.dtype).encode())
        if budget > 0:
            flat = p.detach().reshape(-1).float()
            take = min(flat.numel(), budget // 4)
            if take:
                h.update(flat[:take].cpu().numpy().tobytes())
                budget -= take * 4
    return h.hexdigest()[:32]


@dataclass
class Version:
    """One entry in the model's lineage."""

    version: int
    patch_id: str
    parent: int
    created_at: float
    doc_start: int          # slice of the corpus this patch learned
    doc_end: int
    n_documents: int
    n_chars: int
    n_tokens: int
    steps: int
    rank: int
    trainable_params: int
    loss_before: float
    loss_after: float
    sources: list[str] = field(default_factory=list)
    note: str = ""
    base_fingerprint: str = ""

    @property
    def filename(self) -> str:
        return f"{self.version:04d}-{self.patch_id}.pt"


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update.

    y = W x + (alpha / r) * B (A x), with W frozen and only A, B learned.
    `merge_` folds the update into W so the result is an ordinary Linear again
    with no inference cost.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        # A is random and B is zero, so the patch starts as an exact no-op and
        # the model's behaviour at step 0 is identical to the version before it.
        nn.init.normal_(self.A, std=1.0 / max(rank, 1))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B) * self.scale

    @torch.no_grad()
    def merge_(self) -> nn.Linear:
        self.base.weight.add_((self.B @ self.A) * self.scale)
        return self.base


def _target_names(cfg: PatchConfig) -> tuple[str, ...]:
    return ATTN_TARGETS + (FFN_TARGETS if cfg.include_ffn else ())


def inject_lora(model: nn.Module, cfg: PatchConfig) -> list[LoRALinear]:
    """Wrap every targeted Linear in a LoRALinear. Returns the wrappers."""
    targets = _target_names(cfg)
    wrapped: list[LoRALinear] = []

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in targets:
                if not cfg.include_experts and ".experts." in f".{path}.":
                    continue
                lora = LoRALinear(child, cfg.rank, cfg.alpha)
                setattr(module, name, lora)
                wrapped.append(lora)
            else:
                walk(child, path)

    walk(model, "")
    return wrapped


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the A/B tensors — this is the patch payload."""
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if name.endswith((".A", ".B"))
    }


def merge_all(model: nn.Module) -> int:
    """Fold every LoRA wrapper back into its base Linear. Returns how many."""
    merged = 0

    def walk(module: nn.Module) -> None:
        nonlocal merged
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge_())
                merged += 1
            else:
                walk(child)

    walk(model)
    return merged


def apply_patch(model: nn.Module, payload: dict, cfg: PatchConfig) -> int:
    """Load a saved patch into `model` and merge it in permanently."""
    inject_lora(model, cfg)
    missing = model.load_state_dict(payload, strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected:
        raise ValueError(f"patch does not fit this model: {unexpected[:3]}")
    return merge_all(model)


class PatchStore:
    """The lineage on disk: base checkpoint, patch files, version manifest."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run = Path(run_dir)
        self.dir = self.run / "patches"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.run / "versions.json"

    def manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as fh:
                return json.load(fh)
        return {"current": 0, "head": 0, "base_docs": 0, "versions": []}

    def write(self, manifest: dict) -> None:
        with open(self.manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    @property
    def current(self) -> int:
        return self.manifest()["current"]

    @property
    def head(self) -> int:
        """The highest version ever produced (checkout can sit below it)."""
        m = self.manifest()
        return m.get("head", m["current"])

    def versions(self) -> list[Version]:
        return [Version(**v) for v in self.manifest()["versions"]]

    def version(self, n: int) -> Version | None:
        return next((v for v in self.versions() if v.version == n), None)

    def consumed_docs(self) -> int:
        """How many corpus documents the current lineage has already learned.

        `base_docs` is the watermark written when the base model was trained:
        everything in the corpus at that moment is already in v0, so a patch
        must only ever learn what arrived afterwards.
        """
        m = self.manifest()
        from_patches = max((v["doc_end"] for v in m["versions"]), default=0)
        return max(from_patches, m.get("base_docs", 0))

    def set_base_docs(self, n: int) -> None:
        """Record that the first `n` documents are baked into the base weights."""
        m = self.manifest()
        m["base_docs"] = n
        self.write(m)

    @property
    def base_fingerprint(self) -> str:
        return self.manifest().get("base_fingerprint", "")

    def set_base(self, fingerprint: str, n_docs: int) -> list[str]:
        """Adopt a base checkpoint, discarding a lineage built on a different one.

        Retraining the base produces different weights, which makes every
        existing patch a delta against something that no longer exists. Keeping
        them would silently corrupt the model, so they are dropped and named.
        """
        m = self.manifest()
        previous = m.get("base_fingerprint", "")
        dropped: list[str] = []
        if previous and previous != fingerprint and m["versions"]:
            for v in m["versions"]:
                dropped.append(f"v{v['version']} ({v['patch_id']})")
                stale = self.dir / f"{v['version']:04d}-{v['patch_id']}.pt"
                stale.unlink(missing_ok=True)
            m["versions"] = []
            m["current"] = 0
            m["head"] = 0
        m["base_fingerprint"] = fingerprint
        m["base_docs"] = n_docs
        self.write(m)
        return dropped

    def record(self, v: Version, payload: dict) -> None:
        torch.save(payload, self.dir / v.filename)
        m = self.manifest()
        m["versions"] = [x for x in m["versions"] if x["version"] != v.version]
        m["versions"].append(asdict(v))
        m["versions"].sort(key=lambda x: x["version"])
        m["current"] = v.version
        m["head"] = max(v.version, m.get("head", 0))
        self.write(m)

    def set_current(self, n: int) -> None:
        m = self.manifest()
        if n != 0 and not any(x["version"] == n for x in m["versions"]):
            raise ValueError(f"no such version: v{n}")
        m["current"] = n
        self.write(m)

    def load_payload(self, v: Version) -> dict:
        return torch.load(self.dir / v.filename, map_location="cpu", weights_only=True)


def build_version(run_dir: str | Path, target: int | None = None, device="cpu"):
    """Materialise a specific version: base checkpoint + the first N patches.

    Returns (model, tokenizer, version_number).
    """
    from motherbrain.cli import load_runtime

    store = PatchStore(run_dir)
    target = store.current if target is None else target
    model, tok, dev, meta = load_runtime(str(run_dir), device if device != "cpu" else "auto")

    expected = store.base_fingerprint
    if expected:
        actual = weights_fingerprint(model)
        if actual != expected:
            raise ValueError(
                f"these patches were trained against a different base checkpoint "
                f"(manifest {expected}, loaded {actual}). Applying them would "
                f"corrupt the model. Retrain the base, or restore the checkpoint "
                f"the lineage was built on."
            )

    for v in store.versions():
        if v.version > target:
            break
        cfg = PatchConfig(rank=v.rank, include_ffn=True)
        apply_patch(model, store.load_payload(v), cfg)
    model.eval()
    return model, tok, target


# ---- training a patch -----------------------------------------------------


def _tokenize_docs(corpus: Corpus, tok, docs: list[dict]) -> np.ndarray:
    ids: list[int] = []
    for doc in docs:
        ids.extend(tok.encode(doc["text"], bos=True, eos=True))
    return np.asarray(ids, dtype=TOKEN_DTYPE)


class _PatchBatcher:
    """Mixes new information with replay from everything learned before."""

    def __init__(self, new_tokens: np.ndarray, old_path: Path, seq_len: int,
                 replay_ratio: float, seed: int) -> None:
        self.new = new_tokens
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.replay_ratio = replay_ratio if old_path.exists() else 0.0
        self.old = np.memmap(old_path, dtype=TOKEN_DTYPE, mode="r") \
            if self.replay_ratio > 0 else None
        if self.old is not None and len(self.old) < seq_len + 2:
            self.old, self.replay_ratio = None, 0.0

    def _draw(self, source: np.ndarray, n: int) -> list[np.ndarray]:
        hi = len(source) - self.seq_len - 1
        if hi <= 0:
            # Too short to slice: pad by tiling so short patches still train.
            reps = int(np.ceil((self.seq_len + 2) / max(len(source), 1)))
            source = np.tile(source, reps)
            hi = len(source) - self.seq_len - 1
        starts = self.rng.integers(0, max(hi, 1), size=n)
        return [source[s:s + self.seq_len + 1] for s in starts]

    def batch(self, batch_size: int):
        n_replay = int(round(batch_size * self.replay_ratio)) if self.old is not None else 0
        n_new = max(1, batch_size - n_replay)
        chunks = self._draw(self.new, n_new)
        if n_replay:
            chunks += self._draw(self.old, n_replay)
        arr = np.stack(chunks).astype(np.int64)
        x = torch.from_numpy(arr[:, :-1])
        y = torch.from_numpy(arr[:, 1:])
        return x, y


@torch.no_grad()
def _mean_loss(model, batcher, batch_size: int, device, n: int = 5) -> float:
    model.eval()
    total = 0.0
    for _ in range(n):
        x, y = batcher.batch(batch_size)
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    return total / n


def create_patch(run_dir: str | Path, corpus_dir: str | Path,
                 cfg: PatchConfig | None = None, note: str = "",
                 device: str = "auto", progress_cb=None,
                 doc_start: int | None = None) -> Version | None:
    """Train a patch on the corpus documents not yet folded into a version.

    Returns the new Version, or None when there is nothing new to learn.
    """
    from motherbrain.train import pick_device

    cfg = cfg or PatchConfig()
    torch.manual_seed(cfg.seed)
    store = PatchStore(run_dir)
    corpus = Corpus(corpus_dir)

    docs = list(corpus.documents())
    start = store.consumed_docs() if doc_start is None else doc_start
    new_docs = docs[start:]
    if not new_docs:
        return None

    dev = pick_device(device)
    model, tok, base_version = build_version(run_dir, device=device)
    model.to(dev)

    seq_len = min(cfg.seq_len or model.cfg.max_seq_len, model.cfg.max_seq_len)
    new_tokens = _tokenize_docs(corpus, tok, new_docs)
    if len(new_tokens) < 2:
        return None

    batcher = _PatchBatcher(new_tokens, Path(corpus.tokens_path), seq_len,
                            cfg.replay_ratio, cfg.seed)
    # Training mixes in replay, but "did this patch work?" is a question about
    # the new material, so the reported loss is measured on that alone.
    probe = _PatchBatcher(new_tokens, Path(corpus.tokens_path), seq_len, 0.0, cfg.seed)

    loss_before = _mean_loss(model, probe, cfg.batch_size, dev)

    # Freeze everything, then add the trainable low-rank delta.
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = inject_lora(model, cfg)
    model.to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if not trainable:
        raise RuntimeError("patch has no trainable parameters; check target names")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()

    for step in range(cfg.steps):
        x, y = batcher.batch(cfg.batch_size)
        _, loss = model(x.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if progress_cb:
            progress_cb({"step": step + 1, "total": cfg.steps, "loss": loss.item()})

    loss_after = _mean_loss(model, probe, cfg.batch_size, dev)

    payload = lora_state(model)
    version = Version(
        version=store.head + 1,
        patch_id=uuid.uuid4().hex[:8],
        parent=base_version,
        created_at=time.time(),
        doc_start=start,
        doc_end=len(docs),
        n_documents=len(new_docs),
        n_chars=sum(d["chars"] for d in new_docs),
        n_tokens=int(len(new_tokens)),
        steps=cfg.steps,
        rank=cfg.rank,
        trainable_params=n_trainable,
        loss_before=round(loss_before, 4),
        loss_after=round(loss_after, 4),
        sources=sorted({d.get("source", "?") for d in new_docs})[:20],
        note=note,
        base_fingerprint=store.base_fingerprint,
    )
    store.record(version, payload)
    return version
