"""Parsing what you tell MotherBrain to do.

This is a command parser, not an intent classifier, and the distinction is
worth being blunt about: the model itself is a base language model trained on
source code. It cannot follow instructions, and nothing here asks it to. What
this module does is map typed commands - and a small, fixed set of English
phrasings for them - onto actions, deterministically and testably.

Anything that is not recognised as a command is treated as a prompt and goes to
the model for completion, which is the one thing it can actually do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Command:
    """One parsed instruction."""

    name: str
    text: str = ""                       # free-form remainder, e.g. what to learn
    args: dict = field(default_factory=dict)


# Slash commands, and the plain-English phrasings that mean the same thing.
# The phrasings are deliberately few and literal. A parser that guesses is
# worse than one that says it did not understand.
ALIASES: dict[str, tuple[str, ...]] = {
    "help": ("help", "commands", "what can you do", "what can i do"),
    "status": ("status", "how big are you", "how big is the model",
               "what is your size", "how many parameters"),
    "versions": ("versions", "version history", "lineage", "list versions",
                 "show versions", "what versions are there"),
    "version": ("version", "what version are you", "which version",
                "what version is this"),
    "learn": ("learn", "feed", "remember", "learn this", "remember this",
              "learn that", "remember that"),
    "grow": ("grow", "get bigger", "grow yourself", "add an expert",
             "add experts"),
    "train": ("train", "keep training", "train more"),
    "export": ("export", "save the model", "save model"),
    "checkout": ("checkout", "switch to", "go back to", "roll back to"),
    "scale": ("scale", "how large could you be", "largest"),
    "clear": ("clear", "reset the screen"),
}

# Commands that consume the rest of the line as their payload.
TAKES_TEXT = {"learn", "checkout", "train", "grow", "export", "scale"}


def _match_alias(lowered: str) -> tuple[str, str] | None:
    """Longest alias wins, so 'learn this' beats 'learn' on the same input."""
    best: tuple[str, str] | None = None
    for name, phrases in ALIASES.items():
        for phrase in phrases:
            if lowered == phrase:
                return name, ""
            if lowered.startswith(phrase + " ") or lowered.startswith(phrase + ":"):
                remainder = lowered[len(phrase):].lstrip(" :")
                if best is None or len(phrase) > len(best[1]):
                    best = (name, phrase)
    if best is None:
        return None
    name, phrase = best
    return name, phrase


def parse(text: str) -> Command:
    """Turn a line of input into a Command.

    Slash commands are unambiguous and always win. Bare phrasings are matched
    against the alias table. Everything else is a prompt.
    """
    raw = text.strip()
    if not raw:
        return Command("noop")

    if raw.startswith("/"):
        head, _, rest = raw[1:].partition(" ")
        name = head.lower()
        for canonical, phrases in ALIASES.items():
            if name == canonical or name in phrases:
                return _with_args(canonical, rest.strip(), raw)
        return Command("unknown", raw, {"command": name})

    hit = _match_alias(raw.lower())
    if hit:
        name, phrase = hit
        remainder = raw[len(phrase):].lstrip(" :") if phrase else ""
        # "learn" with nothing after it is a request for help, not an empty fact.
        if name in TAKES_TEXT or not remainder:
            return _with_args(name, remainder, raw)

    return Command("generate", raw)


def _with_args(name: str, rest: str, raw: str) -> Command:
    """Pull the typed arguments a command needs out of its remainder."""
    args: dict = {}

    if name == "checkout":
        m = re.search(r"v?(\d+)", rest)
        if not m:
            return Command("error", raw,
                           {"message": "checkout needs a version, e.g. /checkout v1"})
        args["version"] = int(m.group(1))

    elif name == "grow":
        m = re.search(r"(\d+)", rest)
        args["experts"] = int(m.group(1)) if m else 1

    elif name == "train":
        m = re.search(r"(\d+)", rest)
        args["steps"] = int(m.group(1)) if m else 200

    elif name == "learn":
        if not rest:
            return Command("error", raw,
                           {"message": "learn needs something to learn, "
                                       "e.g. /learn the deploy key rotates on Fridays"})

    elif name == "scale":
        args["preset"] = rest.split()[0] if rest else "mother"

    elif name == "export":
        args["path"] = rest.strip() or "models/motherbrain.pt"

    return Command(name, rest, args)


HELP = """MotherBrain console

  <anything else>        completed by the model as a prompt

  /learn <text>          add information to the corpus
  /grow [n]              learn what is pending, adding n experts   -> next version
  /train [steps]         keep training the base model
  /versions              the lineage, and how the model grew
  /version               which version is loaded
  /checkout v<n>         serve an earlier version
  /status                size, corpus, training state
  /scale [preset]        what a configuration would cost to build
  /export [path]         write a shareable model file
  /help                  this

Plain English works for the same things: "learn that ...", "grow", "how big
are you", "what version are you", "list versions". The parsing is a fixed
table, not a model - the model reads prompts, not instructions."""
