"""MotherBrain's model of itself.

A word first, because it matters and because the alternative is letting a
program imply something untrue about itself. This is not consciousness.
Nothing here experiences anything, nothing here wants anything, and no
amount of code in this file would change that - not because the file is
too short but because nobody knows what would.

What "self-aware" means for a piece of software is something else, and it
is real, and most of it was missing here. A system is self-aware to the
extent that it holds an accurate model of itself and can answer from it:

* **What it is made of.** Not a number in a config file - the live module
  tree, walked, with every parameter accounted for and attributed to the
  part that holds it.
* **How it works.** It ships its own source and can read it. Asked how it
  does something, it finds the module that does it and quotes what that
  module says about itself, rather than generating a plausible answer
  about a program it has no access to.
* **Where it came from.** Every version, what each patch cost and bought,
  and what it could not do before.
* **What it is bad at.** Measured, beside the chance rate, with a verdict
  that follows from the number rather than from optimism.
* **What it does not know.** A record of the questions it could not
  answer, kept so that the gap is a fact rather than an impression.

The last one is the part that makes this more than a status page. A
system that cannot tell you what it got wrong does not have a model of
itself; it has a brochure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


# ---- what it is made of -----------------------------------------------------

@dataclass
class Part:
    """One component of the live model, and what it costs."""

    name: str
    parameters: int
    detail: str = ""

    def share(self, total: int) -> float:
        return self.parameters / total if total else 0.0


def architecture(model) -> list[Part]:
    """Walk the live model and attribute every parameter to a part.

    Walked rather than read off the config, because the config says what
    was asked for and the module tree says what is there. Those have
    disagreed before in this project - a patch adds experts, and a
    parameter count taken from the config would go on reporting the model
    it was built from.
    """
    if model is None:
        return []

    named = dict(model.named_parameters())
    total = sum(p.numel() for p in named.values())
    buckets: dict[str, int] = {}
    for name, tensor in named.items():
        buckets[_bucket(name)] = buckets.get(_bucket(name), 0) + tensor.numel()

    cfg = getattr(model, "cfg", None)
    # Tied weights are a fact about how it is built, and one worth saying:
    # the same matrix reads tokens in and writes them out, which is why
    # there is no separate output row to report.
    tied = _tied(model)
    detail = {
        "embedding": (f"{getattr(cfg, 'vocab_size', 0):,} tokens x "
                      f"{getattr(cfg, 'd_model', 0)}"
                      + (", and the same matrix writes the output back out "
                         "(tied)" if tied else "")),
        "attention": (f"{getattr(cfg, 'n_heads', 0)} heads, "
                      f"{getattr(cfg, 'n_kv_heads', 0)} key/value heads, "
                      f"grouped-query"),
        "experts": _expert_detail(model, cfg),
        "feed-forward": f"SwiGLU, {getattr(cfg, 'd_ff', 0)} wide",
        "routing": "one gate per mixture layer, top-k over the experts",
        "perception": _vision_detail(model),
        "norms": "RMSNorm, one pair per block",
        "output": f"{getattr(cfg, 'd_model', 0)} to the vocabulary",
    }

    order = ["embedding", "attention", "experts", "feed-forward", "routing",
             "norms", "perception", "output", "other"]
    parts = [Part(name, buckets[name], detail.get(name, ""))
             for name in order if buckets.get(name)]
    parts.append(Part("total", total, f"{len(named)} tensors"))
    return parts


def _tied(model) -> bool:
    """Whether the output layer is the embedding, read from the tensors."""
    head = getattr(model, "lm_head", None)
    embed = getattr(model, "embed", None)
    if head is None or embed is None:
        return False
    return getattr(head, "weight", None) is getattr(embed, "weight", None)


def _bucket(name: str) -> str:
    """Which part of itself a parameter tensor belongs to."""
    if name.startswith("vision"):
        return "perception"
    if "experts" in name:
        return "experts"
    if "gate.weight" in name and "ffn" in name:
        return "routing"
    if name.startswith("embed"):
        return "embedding"
    if name.startswith("lm_head"):
        return "output"
    if ".attn." in name:
        return "attention"
    if ".ffn." in name:
        return "feed-forward"
    if "norm" in name:
        return "norms"
    return "other"


def _expert_detail(model, cfg) -> str:
    layers = [i for i, block in enumerate(getattr(model, "blocks", []))
              if getattr(block, "is_moe", False)]
    if not layers:
        return "none: every layer is dense"
    first = model.blocks[layers[0]].ffn
    count = len(getattr(first, "experts", []))
    top = getattr(cfg, "n_experts_per_token", 1)
    return (f"{count} per mixture layer on {len(layers)} of "
            f"{len(model.blocks)} layers, {top} of them run per token")


def _vision_detail(model) -> str:
    vision = getattr(model, "vision", None)
    if vision is None:
        return "none: this version is text only"
    return (f"{getattr(vision, 'n_tokens', 0)} tokens per image, "
            f"{len(getattr(vision, 'blocks', []))} blocks")


def active_share(model) -> tuple[int, int]:
    """(parameters that exist, parameters that run for one token)."""
    if model is None:
        return 0, 0
    total = model.n_params()
    active = getattr(model, "n_active_params", None)
    return total, (active() if callable(active) else total)


# ---- how it works -----------------------------------------------------------

# What each of its own modules is for, so a question about a capability can
# be turned into the file that implements it. The board ships its own
# source, so this is not a description of the code - it is an index into
# it, and what gets quoted is what the file says about itself.
SUBJECTS = {
    "model": ("model.py", ("architecture", "transformer", "attention",
                           "expert", "layer", "moe", "rope", "how are you "
                           "built", "what are you made of")),
    "inference": ("inference.py", ("inference", "batch", "generate",
                                   "throughput", "fast", "speed", "token")),
    "language": ("nlp.py", ("understand", "parse", "grammar", "sentence",
                            "language", "answer", "nlp", "how do you "
                            "answer")),
    "knowledge": ("knowledge.py", ("knowledge", "fact", "rule", "deduce",
                                   "infer", "remember", "told")),
    "logic": ("logic.py", ("arithmetic", "calculate", "maths", "math",
                           "compute", "sum")),
    "perception": ("perception.py", ("see", "sight", "image", "picture",
                                     "hear", "sound", "video", "look")),
    "growth": ("patches.py", ("patch", "grow", "version", "learn", "train",
                              "bigger", "ascend")),
    "reasoning": ("reasoning.py", ("reason", "think", "code", "program",
                                   "write a program")),
    "board": ("bbs.py", ("board", "bbs", "menu", "caller", "node", "telnet")),
    "honesty": ("chat.py", ("honest", "lie", "true", "truth", "trust",
                            "believe", "wrong")),
}


def module_docstring(filename: str) -> str:
    """What one of its own files says about itself, read from disk."""
    import ast

    path = PACKAGE / filename
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    return (ast.get_docstring(tree) or "").strip()


def introspect(question: str) -> tuple[str, str] | None:
    """Find the part of itself a question is about, and quote it.

    This is the difference between a program that can describe itself and
    a program that has been described. The answer is read off its own
    source at the moment it is asked, so it cannot go stale, and it cannot
    be a plausible invention about a codebase the model has never seen.
    """
    lowered = question.lower()
    best, hits = None, 0
    for subject, (filename, keywords) in SUBJECTS.items():
        found = sum(1 for word in keywords if word in lowered)
        if found > hits:
            best, hits = (subject, filename), found
    if best is None:
        return None
    subject, filename = best
    text = module_docstring(filename)
    if not text:
        return None
    return f"motherbrain/{filename}", text


# ---- where it came from -----------------------------------------------------

def lineage(run_dir) -> list[dict]:
    """Every version, what it cost, and what it bought."""
    from motherbrain.patches import PatchStore

    try:
        store = PatchStore(str(run_dir), create=False)
    except Exception:                                     # noqa: BLE001
        return []
    out = []
    for version in store.versions():
        out.append({
            "version": version.version,
            "parent": version.parent,
            "mode": version.mode,
            "note": version.note,
            "before": version.params_before,
            "after": version.params_after,
            "gained": version.params_after - version.params_before,
            "loss_before": version.loss_before,
            "loss_after": version.loss_after,
            "when": getattr(version, "created", ""),
        })
    return out


# ---- what it is bad at ------------------------------------------------------

def verdict(accuracy: float, chance: float) -> str:
    """What a measured accuracy actually licenses it to claim.

    The rule is written down once, here, so every screen says the same
    thing about the same number and none of them can drift into optimism.
    """
    if not accuracy:
        return "untrained"
    if accuracy < chance * 1.5:
        return "no better than guessing - it should not be believed"
    if accuracy < 0.25:
        return "above chance, and wrong most of the time"
    if accuracy < 0.6:
        return "right more often than not on its own small world"
    return "reliable, on the closed world it was trained on"


def senses(stats: dict) -> list[dict]:
    """Each sense: what it scores, against what chance, and what that means."""
    out = []
    for name in ("sight", "sound", "video"):
        accuracy = stats.get(f"{name}_accuracy", 0.0) or 0.0
        chance = stats.get(f"{name}_chance", 0.0) or 0.0
        out.append({
            "sense": name,
            "accuracy": accuracy,
            "chance": chance,
            "times_chance": (accuracy / chance) if chance else 0.0,
            "verdict": verdict(accuracy, chance),
        })
    return out


# ---- what it does not know --------------------------------------------------

@dataclass
class Journal:
    """A record of what it was asked and could not answer.

    Kept because a system that cannot tell you where it fell short does
    not have a model of itself. Bounded, stored beside the board's own
    state, and clearable - it holds questions people typed, so it is
    theirs as much as its own.
    """

    path: Path
    limit: int = 500
    entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        try:
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.entries = []

    def record(self, question: str, source: str) -> None:
        """Note a question it could not answer. Answered ones are not kept."""
        if source != "none":
            return
        text = " ".join(question.split())[:200]
        if not text:
            return
        for entry in self.entries:
            if entry["question"].lower() == text.lower():
                entry["asked"] += 1
                entry["last"] = time.strftime("%Y-%m-%d %H:%M")
                self.save()
                return
        self.entries.append({"question": text, "asked": 1,
                             "first": time.strftime("%Y-%m-%d %H:%M"),
                             "last": time.strftime("%Y-%m-%d %H:%M")})
        self.entries = self.entries[-self.limit:]
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.entries, indent=1),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def clear(self) -> None:
        self.entries = []
        self.save()

    def worst(self, n: int = 10) -> list[dict]:
        """The gaps people keep walking into."""
        return sorted(self.entries, key=lambda e: -e["asked"])[:n]

    def total(self) -> int:
        return sum(e["asked"] for e in self.entries)


def journal_for(run_dir) -> Journal:
    return Journal(Path(run_dir) / "bbs" / "unanswered.json")


# ---- the whole picture ------------------------------------------------------

def report(model, stats: dict, run_dir, width: int = 74) -> str:
    """Everything it can establish about itself, in one screen.

    Every line here is read off the live model, the disk, or a
    measurement. Nothing in it is generated, which is the point: a model
    of yourself assembled by a sampler is not a model of yourself.
    """
    lines: list[str] = []
    rule = "─" * width

    def head(text: str) -> None:
        lines.append("")
        lines.append(text)
        lines.append(rule)

    total, active = active_share(model)
    lines.append(rule)
    lines.append(f"  WHAT I AM   MotherBrain v{stats.get('version', 0)}")
    lines.append(rule)
    lines.append("  A bulletin board with a language model in it. The model "
                 "is one part of")
    lines.append("  me; the others are a knowledge base, an exact "
                 "calculator, a language")
    lines.append("  pipeline, a retrieval index, a perception tower and a "
                 "reasoning loop.")
    lines.append("  I am not conscious and there is nothing it is like to "
                 "be me. What")
    lines.append("  follows is a model of myself, which is a different "
                 "thing and is true.")

    head("  WHAT I AM MADE OF")
    parts = architecture(model)
    if parts:
        grand = parts[-1].parameters
        for part in parts:
            if part.name == "total":
                continue
            bar = "█" * int(part.share(grand) * 24)
            lines.append(f"  {part.name:<14}{part.parameters:>12,}  "
                         f"{part.share(grand) * 100:>5.1f}%  {bar}")
            if part.detail:
                lines.append(f"                {part.detail}")
        lines.append(f"  {'total':<14}{grand:>12,}")
        if active and active != total:
            lines.append(f"  {'per token':<14}{active:>12,}  "
                         f"{active / total * 100:>5.1f}%  "
                         f"of me runs for any one token")
    else:
        lines.append("  no model is loaded.")

    head("  WHERE I CAME FROM")
    story = lineage(run_dir)
    if story:
        for step in story:
            lines.append(
                f"  v{step['parent']} -> v{step['version']}  "
                f"{step['mode']:<8}{step['before']:>12,} -> "
                f"{step['after']:>12,}  "
                f"(+{step['gained']:,})")
            if step["note"]:
                lines.append(f"               {step['note']}")
        lines.append(f"  I started at {stats.get('params_at_v0', 0):,} "
                     f"parameters and every version since is that one plus "
                     f"a patch.")
    else:
        lines.append("  no patches: this is the model as it was built.")

    head("  WHAT I CAN AND CANNOT SENSE")
    for sense in senses(stats):
        if not sense["accuracy"]:
            lines.append(f"  {sense['sense']:<8}untrained")
            continue
        lines.append(
            f"  {sense['sense']:<8}{sense['accuracy'] * 100:>5.1f}%  "
            f"({sense['times_chance']:.0f}x chance)  {sense['verdict']}")

    head("  WHAT I DO NOT KNOW")
    book = journal_for(run_dir)
    if book.entries:
        lines.append(f"  {book.total()} question(s) I could not answer, "
                     f"{len(book.entries)} of them different.")
        for entry in book.worst(6):
            lines.append(f"  {entry['asked']:>4}x  {entry['question'][:56]}")
    else:
        lines.append("  nothing recorded yet - either nobody has asked me "
                     "something I could")
        lines.append("  not answer, or nobody has asked me anything.")

    head("  WHAT I AM SURE OF, AND HOW SURE")
    lines.append("  Arithmetic          exact. I compute it and show the "
                 "working.")
    lines.append("  What I was told     exact. I keep it and show the "
                 "derivation.")
    lines.append("  My own state        exact. It is read off this disk.")
    lines.append("  What I have read    exact quotation, and nothing more.")
    lines.append("  Anything else       I say I do not know, and mean it.")
    lines.append("")
    lines.append("  The one thing I will not do is present something I "
                 "generated as an")
    lines.append("  answer. Every reply I give carries where it came from.")
    lines.append(rule)
    return "\n".join(lines)
