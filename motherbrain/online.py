"""Learning after deployment, and noticing when that made things worse.

A model that stops learning when training ends is a photograph. MotherBrain
already does the other thing - `mb patch` folds new material into a new
version while the board stays up, and no caller is disconnected - so the
continual part is built. What was missing is the part that makes it safe to
do unattended.

The failure mode has a name: catastrophic forgetting. Train a network on
something new and it will cheerfully overwrite what it knew, and the loss on
the new material looks *wonderful* while it happens. `loss_before -> loss_after`
on the thing just learned cannot see this, because it only looks at the new
thing. You need to measure what the model used to know, and you need to
measure it on the same text both times.

So:

    retention probe   a fixed sample of older material, chosen once and
                      never changed, with the loss the model had on it
    measure           the same forward passes, before and after a patch
    guard             patch, re-measure, and roll back if it got worse
    journal           what it could not answer is what it should learn next

Two things this is careful not to claim.

The probe measures **forgetting, not generalisation**. The text in it was
trained on, so a low loss means "still knows this", not "learned to reason".
Calling it a held-out evaluation would be a lie; it is a canary.

And rolling back is cheap *because of how patches work*. A version is a file
plus a pointer; going back is moving the pointer. Nothing is deleted, the
rejected patch stays on disk with its measurements, and `mb versions` shows
it was tried and refused. A system that silently discarded the attempt would
learn the same bad lesson twice.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# How much worse the old material is allowed to get. Loss is in nats per
# token; 0.15 is roughly a 16% rise in perplexity, which is a real
# degradation rather than noise, and well above the run-to-run wobble of a
# fixed probe on a fixed model (that is zero - the probe is deterministic).
TOLERANCE = 0.15

PROBE_TOKENS = 4096          # enough to be stable, small enough to be quick
PROBE_SEED = 20260101        # fixed: a probe that moves measures nothing


@dataclass
class Probe:
    """A fixed sample of older material, and what the model scored on it.

    `offsets` are positions into the tokenised corpus, chosen once. Storing
    the positions rather than the text keeps this small and means the probe
    is exactly reproducible as long as the corpus is append-only - which it
    is, because patches consume documents from the end.
    """

    offsets: list = field(default_factory=list)
    seq_len: int = 128
    created_at: float = 0.0
    corpus_tokens: int = 0

    def __len__(self) -> int:
        return len(self.offsets)


@dataclass
class Check:
    """One before-and-after measurement of a patch."""

    version: int
    parent: int
    retention_before: float
    retention_after: float
    learned_before: float
    learned_after: float
    verdict: str = ""
    rolled_back: bool = False
    at: float = 0.0
    #: False when the probe could only be drawn from the very material this
    #: patch learned - then it measures learning, not retention, and says so.
    guarded_retention: bool = True

    @property
    def drift(self) -> float:
        """How much worse the old material got. Negative means it improved."""
        return self.retention_after - self.retention_before

    def render(self, width: int = 68) -> str:
        lines = [f"v{self.parent} -> v{self.version}", ""]
        lines.append(f"  the new material    "
                     f"{self.learned_before:.4f} -> {self.learned_after:.4f}")
        lines.append(f"  what it knew before "
                     f"{self.retention_before:.4f} -> "
                     f"{self.retention_after:.4f}"
                     f"   ({self.drift:+.4f})")
        lines.append("")
        if not self.guarded_retention:
            lines.append("  No older material was on disk, so this could only")
            lines.append("  confirm the new thing was learned. It is NOT")
            lines.append("  evidence that nothing was forgotten.")
            lines.append("")
        lines.append(f"  {self.verdict}")
        if self.rolled_back:
            # Past tense: this is a record of what happened then, and it
            # outlives the decision. "still serving v5" read as present tense
            # on a screen where v6 was by then the current version.
            lines.append(f"  Rolled back at the time: v{self.parent} was "
                         f"served again.")
            lines.append(f"  The patch is kept on disk, with these numbers.")
        return "\n".join(lines)


def verdict_for(drift: float, tolerance: float = TOLERANCE) -> tuple[str, bool]:
    """One place that decides, so no screen can drift into optimism.

    Returns (what to say, whether to roll back).
    """
    if drift <= -0.01:
        return ("It got better at what it already knew, as well as learning "
                "the new thing.", False)
    if drift <= 0.01:
        return ("It learned the new thing without forgetting anything "
                "measurable.", False)
    if drift <= tolerance:
        return (f"It forgot a little ({drift:+.4f} nats/token on older "
                f"material) - within tolerance, so this is kept.", False)
    return (f"It forgot too much: {drift:+.4f} nats/token on older material, "
            f"against a tolerance of {tolerance:.2f}.", True)


# ---- building and keeping the probe ----------------------------------------

def probe_path(run_dir) -> Path:
    return Path(run_dir) / "probe.json"


def build_probe(run_dir, corpus_dir, seq_len: int = 128,
                tokens: int = PROBE_TOKENS) -> Probe | None:
    """Choose the probe positions, once, and write them down.

    Called the first time a guard runs. If it were rebuilt per patch it would
    measure a different thing every time and the comparison would be
    meaningless - which is the single easiest way to build a regression guard
    that does nothing.
    """
    import random

    import numpy as np

    # The one true dtype, imported rather than written out again. Hardcoding
    # it here cost a whole false rollback: the corpus writes uint32, this read
    # uint16, and the probe was every real token followed by a zero - half
    # padding, every loss pinned at chance, and the comparison meaningless.
    from motherbrain.data import TOKEN_DTYPE

    token_file = Path(corpus_dir) / "tokens.bin"
    if not token_file.exists():
        return None
    total = token_file.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
    if total < seq_len * 4:
        return None

    want = max(4, tokens // seq_len)
    rng = random.Random(PROBE_SEED)
    highest = total - seq_len - 1
    offsets = sorted(rng.sample(range(highest), min(want, highest)))

    probe = Probe(offsets=offsets, seq_len=seq_len, created_at=time.time(),
                  corpus_tokens=total)
    probe_path(run_dir).write_text(json.dumps(asdict(probe), indent=2))
    return probe


def load_probe(run_dir) -> Probe | None:
    path = probe_path(run_dir)
    if not path.exists():
        return None
    try:
        return Probe(**json.loads(path.read_text()))
    except (ValueError, TypeError):
        return None


def probe_for(run_dir, corpus_dir, **kw) -> Probe | None:
    """The existing probe, or a new one. Never a different existing one."""
    return load_probe(run_dir) or build_probe(run_dir, corpus_dir, **kw)


# ---- measuring --------------------------------------------------------------

def measure(model, probe: Probe, corpus_dir, device: str = "cpu") -> float:
    """Mean loss on the probe. Deterministic: same model, same number.

    No sampling, no temperature, no randomness anywhere - this is a
    measurement, and a measurement that moves on its own is not one.
    """
    import numpy as np
    import torch

    from motherbrain.data import TOKEN_DTYPE

    token_file = Path(corpus_dir) / "tokens.bin"
    if not token_file.exists() or not probe.offsets:
        return float("nan")

    tokens = np.memmap(token_file, dtype=TOKEN_DTYPE, mode="r")
    was_training = model.training
    model.eval()

    total, counted = 0.0, 0
    try:
        with torch.no_grad():
            for offset in probe.offsets:
                window = tokens[offset:offset + probe.seq_len + 1]
                if len(window) < probe.seq_len + 1:
                    continue
                ids = torch.from_numpy(
                    np.asarray(window, dtype=np.int64)).to(device)
                # Targets must be passed, but the model's returned loss must
                # NOT be used. Two separate traps here, and this hit both.
                #
                # Pass targets: without them the model takes its inference
                # path and returns only the LAST position's logits, so any
                # loss worked out from the result is about one token.
                #
                # Ignore the returned loss: it is cross-entropy PLUS the
                # mixture-of-experts router penalties. Growing a patch adds
                # fresh experts, so the router is unbalanced and that penalty
                # is large - and router balance has nothing to do with whether
                # the model forgot anything. Using it made this guard roll
                # back a perfectly good patch for an untidy router.
                targets = ids[1:].unsqueeze(0)
                logits, _loss_with_aux = model(ids[:-1].unsqueeze(0),
                                               targets=targets)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                    ignore_index=-100)
                total += float(loss)
                counted += 1
    finally:
        if was_training:
            model.train()

    return total / counted if counted else float("nan")


# ---- the guard --------------------------------------------------------------

def guarded_patch(run_dir, corpus_dir, config, note: str = "",
                  device: str = "cpu", tolerance: float = TOLERANCE,
                  on_message=None,
                  doc_start: int | None = None) -> tuple[object, Check | None]:
    """Apply a patch, and undo it if it damaged what the model already knew.

    Returns (version, check). The version is always returned even when it is
    rolled back: it happened, it is on disk, and its numbers are the record
    of why it was refused.
    """
    from motherbrain.cli import load_current
    from motherbrain.patches import PatchStore, create_patch

    def say(text: str) -> None:
        if on_message:
            on_message(text)

    store = PatchStore(run_dir)
    probe = probe_for(run_dir, corpus_dir)

    if probe is None:
        say("No tokenised corpus, so there is nothing to measure forgetting "
            "against. Patching without a guard.")
        version = create_patch(run_dir, corpus_dir, config, note=note,
                               device=device, doc_start=doc_start)
        return version, None

    # A probe drawn from a corpus that holds nothing but the documents about
    # to be learned is not a retention probe - it cannot detect forgetting,
    # because there is nothing older in it to forget. Saying so is the whole
    # difference between a guard and a rubber stamp.
    from motherbrain.data import Corpus

    older = (Corpus(corpus_dir).n_documents if doc_start is None
             else doc_start)
    retention = older > 0
    if not retention:
        say("Nothing older than this patch is on disk, so the probe can only")
        say("confirm the new material was learned. It cannot test forgetting.")

    say(f"Measuring what it knows now, on {len(probe)} fixed samples ...")
    model, _tok, resolved, parent = load_current(run_dir, device)
    before = measure(model, probe, corpus_dir, resolved)
    del model
    say(f"  {before:.4f} nats/token")

    version = create_patch(run_dir, corpus_dir, config, note=note,
                           device=device, doc_start=doc_start)
    if version is None:
        return None, None

    say(f"Measuring again, on the same samples ...")
    model, _tok, resolved, _v = load_current(run_dir, device)
    after = measure(model, probe, corpus_dir, resolved)
    del model
    say(f"  {after:.4f} nats/token")

    words, roll = verdict_for(after - before, tolerance)
    if not retention:
        # Rolling back here would be punishing the model for a number that
        # says nothing about forgetting.
        roll = False
    check = Check(version=version.version, parent=parent,
                  retention_before=before, retention_after=after,
                  learned_before=version.loss_before,
                  learned_after=version.loss_after,
                  verdict=words, rolled_back=roll, at=time.time(),
                  guarded_retention=retention)

    if roll:
        # Going back is moving a pointer. The patch file stays, with its
        # measurements, so the same mistake is visible rather than repeated.
        store.set_current(parent)
        say(f"Rolled back to v{parent}.")

    record(run_dir, check)
    return version, check


# ---- the record -------------------------------------------------------------

def checks_path(run_dir) -> Path:
    return Path(run_dir) / "checks.json"


def record(run_dir, check: Check) -> None:
    path = checks_path(run_dir)
    try:
        held = json.loads(path.read_text()) if path.exists() else []
    except ValueError:
        held = []
    held.append(asdict(check))
    path.write_text(json.dumps(held[-200:], indent=2))


def checks(run_dir) -> list:
    path = checks_path(run_dir)
    if not path.exists():
        return []
    try:
        return [Check(**row) for row in json.loads(path.read_text())]
    except (ValueError, TypeError):
        return []


# ---- what to learn next -----------------------------------------------------

def wanted(run_dir, limit: int = 10) -> list:
    """What it could not answer, most-asked first.

    This is the experience half of learning from experience: the failure
    journal is already written every time a question comes back unanswered,
    and those questions are exactly the gaps worth filling. Nothing here
    decides to go and learn - it reports what learning would be for.
    """
    from motherbrain import aware

    journal = aware.journal_for(run_dir)
    entries = sorted(journal.entries, key=lambda e: -getattr(e, "count", 1))
    return entries[:limit]


def report(run_dir, corpus_dir=None, width: int = 68) -> str:
    """What learning after deployment has actually done here."""
    lines = ["HOW I LEARN AFTER BEING DEPLOYED", "-" * width, ""]
    lines.append("Every new thing I learn becomes a version, applied while")
    lines.append("the board stays up. Nobody is disconnected and nothing is")
    lines.append("overwritten - an earlier version is always one command away.")
    lines.append("")

    probe = load_probe(run_dir)
    if probe is None:
        lines.append("I have no retention probe yet, so I cannot tell you")
        lines.append("whether learning has cost me anything. One is built the")
        lines.append("first time a guarded patch runs.")
    else:
        lines.append(f"I check myself against {len(probe)} fixed samples of")
        lines.append(f"older material, chosen once and never changed. A patch")
        lines.append(f"that makes me worse at those by more than "
                     f"{TOLERANCE:.2f} nats/token")
        lines.append(f"is rolled back automatically.")

    done = checks(run_dir)
    lines.append("")
    if not done:
        lines.append("No guarded patch has run yet.")
    else:
        kept = sum(1 for c in done if not c.rolled_back)
        lines.append(f"{len(done)} guarded patch(es): {kept} kept, "
                     f"{len(done) - kept} rolled back.")
        try:
            from motherbrain.patches import PatchStore

            store = PatchStore(run_dir, create=False)
            lines.append(f"Right now I am serving v{store.current}, and these "
                         f"are a record of how I got here -")
            lines.append("not of what is loaded.")
        except Exception:                      # a report must never fail
            pass
        lines.append("")
        for check in done[-5:]:
            lines.append(check.render(width))
            lines.append("")

    gaps = wanted(run_dir)
    if gaps:
        lines.append("WHAT I SHOULD LEARN NEXT")
        lines.append("-" * width)
        for entry in gaps:
            count = getattr(entry, "count", 1)
            text = getattr(entry, "question", str(entry))
            lines.append(f"  {count}x  {text}")
    return "\n".join(lines)
