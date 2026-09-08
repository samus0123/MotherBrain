"""Answering honestly at 50M parameters.

A base model this size continues text. It does not answer questions, and a
chat box that presents its continuations as answers is lying about what it is
- fluently, which is the worst way to be wrong.

But MotherBrain knows a great deal that it does not have to generate. Its
version, its size, what it has learned and when, whether it can see, how well
it can see, what is in its corpus - all of that is on disk and can be reported
exactly. So this module answers what it can answer truthfully, from state, and
marks anything the model produced as the model's continuation rather than an
answer.

The rule is one sentence: never present generated text as a fact.
"""

from __future__ import annotations

import re

# Questions about itself that have real answers on disk. Matched literally,
# because a matcher that guesses would reintroduce exactly the dishonesty
# this module exists to avoid.
_PATTERNS = [
    ("version", r"\b(what|which)\s+version|\bhow many versions|\bwhat v\b"),
    ("size", r"\bhow (big|large)\b|\bhow many parameters|\bparameter count"),
    ("sight", r"\bcan you see\b|\bdo you see\b|\bcan you look\b|\bsight\b|"
              r"\bare you multimodal\b|\bcan you (hear|watch)\b"),
    ("learned", r"\bwhat (have you|did you) learn|\bwhat do you know\b|"
                r"\bwhat were you trained on\b|\byour corpus\b"),
    ("limits", r"\bwhat can'?t you do\b|\bwhat can you not do\b|"
               r"\byour limits\b|\bwhat are you bad at\b|"
               r"\bcan you (reason|think|understand)\b|\bare you (smart|conscious|alive)\b"),
    ("identity", r"\bwho are you\b|\bwhat are you\b|\bwhat is motherbrain\b"),
    ("capability", r"\bwhat can you do\b|\bhelp\b$|\bwhat are my options\b"),
]


def classify(text: str) -> str | None:
    """Which question about itself this is, if it is one at all."""
    lowered = text.strip().lower()
    for name, pattern in _PATTERNS:
        if re.search(pattern, lowered):
            return name
    return None


def answer_about_self(kind: str, stats: dict) -> str:
    """A truthful answer, built from what is actually on disk."""
    version = stats.get("version", 0)
    total = stats.get("total_params", 0)
    active = stats.get("active_params", 0)
    can_see = stats.get("can_see", False)
    accuracy = stats.get("sight_accuracy", 0.0)
    chance = stats.get("sight_chance", 0.0)

    if kind == "version":
        head = stats.get("head", version)
        line = f"I am version {version}"
        if head > version:
            line += f", checked out from a lineage whose newest is v{head}"
        return (f"{line}. {stats.get('patches', 0)} patch(es) have been applied "
                f"since the base, each adding parameters.")

    if kind == "size":
        return (f"{total:,} parameters. {active:,} of them run for any one "
                f"token, because most of the model is experts that only some "
                f"tokens are routed to. I started at "
                f"{stats.get('params_at_v0', 0):,} and grew.")

    if kind == "sight":
        if not can_see:
            return ("No. I have no perception tower in this version. "
                    "`mb sight` adds one.")

        # Read every sense from state. This answer once said sound and video
        # were untrained, which was true when it was written and false the
        # moment a patch trained them - self-knowledge that hardcodes what it
        # knows goes stale silently, which is the worst way for it to be wrong.
        senses = []
        working = 0
        for sense, what in (("sight", "images"), ("sound", "sounds"),
                            ("video", "clips")):
            accuracy = stats.get(f"{sense}_accuracy", 0.0)
            baseline = stats.get(f"{sense}_chance", 0.0)
            if not accuracy or not baseline:
                continue
            if accuracy > baseline * 2:
                working += 1
                senses.append(f"{sense}: {accuracy:.1%} of held-out {what} "
                              f"named correctly against {baseline:.1%} chance, "
                              f"{accuracy / baseline:.0f} times chance")
            else:
                senses.append(f"{sense}: {accuracy:.1%} against {baseline:.1%} "
                              f"chance, which is not meaningfully better - I "
                              f"should not be believed about {what}")
        if not senses:
            return ("I have a perception tower but nothing has measured it, "
                    "so I cannot tell you whether it works.")

        # Opening with "yes" when nothing clears its baseline is an oversell,
        # and the numbers underneath do not undo a lead that already claimed
        # the capability.
        lead = ("Yes, in narrow ways, and here is exactly how well:"
                if working else
                "No, not really. A tower is attached and measured, and none of "
                "it is meaningfully better than guessing:")
        return (lead + "\n  " + "\n  ".join(senses)
                + "\nThose were measured on generated worlds - coloured "
                  "shapes, tones with a pitch and a timbre, shapes moving in "
                  "one direction. Anything outside those worlds is outside "
                  "what I was trained on, and I would be guessing.")

    if kind == "learned":
        return (f"My base was trained on {stats.get('documents', 0):,} "
                f"documents, {stats.get('tokens', 0):,} tokens - mostly Python "
                f"source and C headers from a few dozen packages. That is why "
                f"everything I write comes out looking like library code. "
                f"Since then, {stats.get('patches', 0)} patch(es).")

    if kind == "limits":
        return ("I continue text. I do not reason, and I do not understand "
                "what I write. At this size I reproduce the shape of code and "
                "prose convincingly without the substance, so anything I "
                "generate should be read as a sample, not an answer. Facts "
                "about myself - version, size, what I learned - I report from "
                "disk rather than generate, and those you can rely on.")

    if kind == "identity":
        heard = stats.get("sound_accuracy", 0.0)
        sight = ("can see and hear, narrowly" if can_see and heard
                 else "can see, narrowly" if can_see else "text only")
        return (f"MotherBrain: a mixture-of-experts transformer, v{version}, "
                f"{total:,} parameters, {sight}. Trained from nothing on this "
                f"machine, and grown by patches rather than retrained.")

    if kind == "capability":
        return ("Four things: write a program from a description, carry out an "
                "instruction (make, run, write, find, delete, list, sh), learn "
                "something new, or apply what was learned as a patch. Ask me "
                "what I am, how big I am, what I learned, or whether I can "
                "see, and you will get facts. Ask me anything else and you "
                "will get a continuation, which is not the same thing.")
    return ""


def respond(text: str, stats: dict) -> tuple[str, str]:
    """(kind, answer). kind is "fact" when this is knowledge, not generation.

    A caller that gets "fact" should print it as MotherBrain's answer. A caller
    that gets "generate" should generate, and label what comes back as a
    continuation rather than a reply.
    """
    kind = classify(text)
    if kind:
        return "fact", answer_about_self(kind, stats)
    return "generate", ""


CONTINUATION_NOTE = (
    "^ that is a continuation of what you wrote, not an answer to it. "
    "I have no way to answer questions about the world."
)
