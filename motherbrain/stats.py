"""What MotherBrain currently is, gathered once and displayed everywhere.

Three consoles showing the same numbers is three chances to compute them
differently, and a stat that disagrees with itself between the window and the
terminal is worse than no stat at all. So the gathering happens here, returns
plain data, and each surface only decides how to draw it.

Nothing here loads a model. The caller passes one if it has one, because the
consoles have already paid for that and a status display should not pay again.
"""

from __future__ import annotations

from pathlib import Path


def human(n: float) -> str:
    """A parameter count at a glance: 47.2M, 1.16Q."""
    for limit, suffix in ((1e15, "Q"), (1e12, "T"), (1e9, "B"), (1e6, "M"),
                          (1e3, "K")):
        if abs(n) >= limit:
            return f"{n / limit:,.3g}{suffix}"
    return f"{n:,.0f}"


def gather(run_dir, corpus_dir, model=None, device=None, steps: int = 0) -> dict:
    """Everything worth knowing about the model on disk, as plain data."""
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchStore

    store = PatchStore(run_dir, create=False)
    corpus = Corpus(corpus_dir, create=False)
    versions = store.versions()

    grown = [v for v in versions if v.params_after]
    sighted = [v for v in versions if v.mode == "sight"]
    best_sight = max(sighted, key=lambda v: v.sight_accuracy, default=None)

    out: dict = {
        "version": store.current,
        "head": store.head,
        "patches": len(versions),
        "run_dir": str(Path(run_dir).resolve()),
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "base_fingerprint": store.base_fingerprint,
        "documents": corpus.n_documents,
        "chars": corpus.n_chars,
        "tokens": corpus.n_tokens,
        "pending": max(0, corpus.n_documents - store.consumed_docs()),
        "trained_steps": steps,
        "device": str(device) if device else "",
        "can_see": False,
        "sight_accuracy": 0.0,
        "sight_chance": 0.0,
        "history": [
            {"version": v.version, "mode": v.mode,
             "params_before": v.params_before, "params_after": v.params_after,
             "loss_before": v.loss_before, "loss_after": v.loss_after,
             "documents": v.n_documents}
            for v in versions
        ],
    }

    if grown:
        out["params_at_v0"] = grown[0].params_before
        out["params_at_v0_human"] = human(grown[0].params_before)
        out["growth"] = grown[-1].params_after - grown[0].params_before

    if best_sight is not None:
        out["sight_accuracy"] = best_sight.sight_accuracy
        out["sight_version"] = best_sight.version
        # 8 colours x 4 shapes: the forced-choice baseline the accuracy is
        # measured against, carried so a display never has to hardcode it.
        from motherbrain.sight import all_captions

        out["sight_chance"] = 1 / len(all_captions())

    if model is not None:
        cfg = model.cfg
        total = model.n_params()
        out.update({
            "total_params": total,
            "active_params": cfg.n_active_params,
            "active_share": (cfg.n_active_params / total) if total else 0.0,
            "layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "heads": cfg.n_heads,
            "kv_heads": cfg.n_kv_heads,
            "experts": cfg.n_experts,
            "experts_per_token": cfg.n_experts_per_token,
            "context": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "preset": cfg.name,
            "can_see": getattr(model, "vision", None) is not None,
            # Pre-formatted too, so a browser rendering these does not need a
            # second implementation of the same abbreviation.
            "total_params_human": human(total),
            "active_params_human": human(cfg.n_active_params),
        })
        if out["can_see"]:
            vision_params = sum(p.numel() for p in model.vision.parameters())
            out.update({
                "vision_params": vision_params,
                "vision_params_human": human(vision_params),
                "vision_layers": cfg.vision_layers,
                "image_size": cfg.image_size,
                "visual_tokens": model.vision.n_tokens,
            })
    return out


def _bar(share: float, width: int = 20) -> str:
    filled = max(0, min(width, round(share * width)))
    return "█" * filled + "·" * (width - filled)


def render(s: dict, width: int = 66) -> str:
    """The stats block the consoles print above the menu."""
    rule = "─" * width

    def row(label: str, value: str) -> str:
        return f"  {label:<15}{value}"

    lines = [rule, f"  MotherBrain v{s['version']}" +
             (f" of v{s['head']}" if s.get("head", 0) > s["version"] else ""),
             rule]

    if "total_params" in s:
        total, active = s["total_params"], s["active_params"]
        lines.append(row("parameters", f"{total:,}  ({human(total)})"))
        lines.append(row("active/token",
                         f"{human(active)}  {_bar(s['active_share'])}  "
                         f"{s['active_share']:.0%}"))
        lines.append(row("architecture",
                         f"{s['layers']} layers x {s['d_model']}, "
                         f"{s['heads']} heads, {s['experts']} experts "
                         f"(top-{s['experts_per_token']})"))
        lines.append(row("context", f"{s['context']:,} tokens, "
                                    f"vocab {s['vocab_size']:,}"))

    if s.get("can_see"):
        chance = s.get("sight_chance", 0)
        verdict = ("above chance" if s["sight_accuracy"] > chance * 2
                   else "NOT above chance")
        lines.append(row("sight", f"{human(s['vision_params'])} params, "
                                  f"{s['visual_tokens']} visual tokens, "
                                  f"{s['image_size']}px"))
        lines.append(row("", f"names {s['sight_accuracy']:.1%} of held-out "
                             f"images ({verdict}, chance {chance:.1%})"))
    else:
        lines.append(row("sight", "none — `mb sight` adds a vision tower"))

    if s.get("growth"):
        lines.append(row("growth", f"{human(s['params_at_v0'])} -> "
                                   f"{human(s['total_params'] if 'total_params' in s else 0)}"
                                   f" over {s['patches']} patch(es), "
                                   f"+{s['growth']:,}"))

    lines.append(row("corpus", f"{s['documents']:,} documents, "
                               f"{s['chars']:,} chars, {s['tokens']:,} tokens"))
    if s["pending"]:
        lines.append(row("", f"{s['pending']:,} fed but not yet learned "
                             f"— option 4 applies them"))
    if s.get("trained_steps"):
        lines.append(row("trained", f"{s['trained_steps']:,} steps"))
    if s.get("device"):
        lines.append(row("running on", s["device"]))
    lines.append(rule)
    return "\n".join(lines)
