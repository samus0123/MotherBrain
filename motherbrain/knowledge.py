"""Knowing things, and working out what follows from them.

A language model predicts text. Told that Socrates is a man and that all men
are mortal, it can produce the sentence "Socrates is mortal" because that
sentence is likely - not because it followed from anything. The two are
indistinguishable from outside until you tell it something the training data
never said, and then only one of them still works.

So this is the other half, and it is deliberately not neural. Facts and rules
go in, forward chaining derives everything that follows, and every answer
carries the chain that produced it. It is exact, it explains itself, and when
nothing follows it says so instead of guessing - which is the property the
language model cannot have.

Small and closed on purpose. Subject-predicate-object triples, universal rules
over one variable, and negation as failure. No quantifier nesting, no
disjunction, nothing whose limits would be hard to state honestly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    """One thing held to be true: subject, relation, object."""

    subject: str
    relation: str
    object: str

    # A rule is told in the plural ("all devices need power") and a fact
    # derived from it is about one thing, so the verb has to come back to the
    # singular or a correct derivation reads as broken English.
    _AGREES = {"have": "has", "need": "needs", "cannot": "cannot",
               "like": "likes", "know": "knows", "own": "owns",
               "use": "uses", "contain": "contains", "beat": "beats"}

    def __str__(self) -> str:
        if self.relation == "is":
            return f"{self.subject} is {self.object}"
        if self.relation == "is a":
            return f"{self.subject} is a {self.object}"
        verb = self._AGREES.get(self.relation, self.relation)
        return f"{self.subject} {verb} {self.object}"

    def as_list(self) -> list[str]:
        return [self.subject, self.relation, self.object]


@dataclass(frozen=True)
class Rule:
    """If a thing is `when`, then it is `then`.

    One variable, universally quantified: "all men are mortal" is
    Rule("man", "mortal"). That covers most of what people actually tell a
    system in passing, and refusing everything else is better than half-doing
    quantifiers.
    """

    when: str
    then: str
    relation: str = "is"

    def __str__(self) -> str:
        # Stored as told, shown as read: "every men is mortal" is what the
        # plural gives you, and it makes a correct derivation look broken.
        verb = {"is": "is", "can": "can", "cannot": "cannot",
                "have": "has", "has": "has", "need": "needs",
                "needs": "needs"}.get(self.relation, self.relation)
        return f"every {singular(self.when)} {verb} {self.then}"


@dataclass
class Derivation:
    """An answer, and the steps that produced it."""

    holds: bool
    known: bool                      # False means "nothing follows either way"
    steps: list[str] = field(default_factory=list)
    # An open question ("what is a modem") is not a yes/no question, and
    # answering one with "Yes." reads as a machine that did not understand it.
    lead: str = ""

    def render(self) -> str:
        if not self.known:
            return ("I do not know. Nothing I have been told settles it, and I "
                    "will not guess.")
        head = self.lead or ("Yes." if self.holds else "No.")
        if not self.steps:
            return head
        return head + "\n  " + "\n  ".join(self.steps)


# ---- reading English, narrowly ---------------------------------------------

# Each pattern is literal. A parser that guesses would put things into the
# knowledge base that were never said, and a wrong fact propagates through
# every rule that touches it.
_RULE_PATTERNS = [
    (r"^(?:all|every)\s+(?P<when>[\w -]+?)\s+(?:are|is)\s+"
     r"(?:an?\s+)?(?P<then>[\w -]+?)$", "is"),
    # "all birds can fly" is the same shape of claim as "all men are mortal",
    # and refusing it sent a perfectly ordinary sentence to the text model.
    (r"^(?:all|every)\s+(?P<when>[\w -]+?)\s+(?P<rel>can|cannot|have|has|need|"
     r"needs)\s+(?P<then>[\w -]+?)$", None),
    (r"^if\s+(?:something|it)\s+is\s+(?:an?\s+)?(?P<when>[\w -]+?)\s*,?\s*"
     r"(?:then\s+)?it\s+is\s+(?:an?\s+)?(?P<then>[\w -]+?)$", "is"),
]

_FACT_PATTERNS = [
    (r"^(?P<subject>[\w -]+?)\s+is\s+an?\s+(?P<object>[\w -]+?)$", "is a"),
    (r"^(?P<subject>[\w -]+?)\s+is\s+(?P<object>[\w -]+?)$", "is"),
    (r"^(?P<subject>[\w -]+?)\s+(?P<relation>can|cannot|likes|knows|owns|made|"
     r"contains|needs|has|have|uses|beats)\s+(?P<object>[\w -]+?)$", None),
]

_NEGATIVE = re.compile(r"^(?P<subject>[\w -]+?)\s+is\s+not\s+"
                       r"(?:an?\s+)?(?P<object>[\w -]+?)$")


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().rstrip(".!").lower())


def _bare(name: str) -> str:
    """A name without its article, or its empty noun.

    "a raven" and "raven" are the same thing, and storing them apart means
    being told one and asked the other. "fizzy things" is the same as "fizzy":
    the noun carries nothing, and leaving it in stops the rule ever matching a
    fact that simply says something is fizzy.
    """
    name = re.sub(r"^(?:a|an|the)\s+", "", name.strip().lower()).strip()
    return re.sub(r"\s+(?:things?|stuff|ones?|objects?)$", "", name).strip()


# English plurals are not a suffix rule, and pretending they are breaks the
# first example anybody tries: "all men are mortal" must fire on "socrates is
# a man", and stripping a trailing s turns "men" into "men".
_IRREGULAR = {
    "men": "man", "women": "woman", "people": "person", "children": "child",
    "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
    "oxen": "ox", "sheep": "sheep", "fish": "fish", "wolves": "wolf",
    "lives": "life", "knives": "knife", "leaves": "leaf",
}


def singular(word: str) -> str:
    """A canonical form, so a rule about a class matches members of it.

    Only used for comparison - nothing is stored in this form, so what you
    were told is still what you get back.
    """
    word = word.strip().lower()
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


# A question is never a statement. "why is the sky blue" matches the shape of
# "X is Y" perfectly, and storing Fact(why, is, sky blue) is the parser
# guessing - which is the one thing it must not do, because a wrong fact then
# propagates through every rule that touches it.
_INTERROGATIVE = re.compile(r"^(?:why|what|how|who|whom|whose|when|where|"
                            r"which|is|are|can|does|do|did|should|would|"
                            r"could|will)\b")


def parse_statement(text: str):
    """A Fact, a Rule, or None. None means it was not understood, and saying
    so is better than storing an approximation of it."""
    line = _tidy(text)
    if not line:
        return None
    if text.strip().endswith("?") or _INTERROGATIVE.match(line):
        return None

    for pattern, relation in _RULE_PATTERNS:
        m = re.match(pattern, line)
        if m:
            groups = m.groupdict()
            return Rule(when=_bare(groups["when"]),
                        then=_bare(groups["then"]),
                        relation=relation or groups.get("rel", "is").strip())

    m = _NEGATIVE.match(line)
    if m:
        return Fact(_bare(m.group("subject")), "is not",
                    _bare(m.group("object")))

    for pattern, relation in _FACT_PATTERNS:
        m = re.match(pattern, line)
        if m:
            groups = m.groupdict()
            return Fact(_bare(groups["subject"]),
                        relation or groups["relation"].strip(),
                        _bare(groups["object"]))
    return None


_QUESTION = re.compile(
    r"^(?:is|are)\s+(?:an?\s+)?(?P<subject>[\w -]+?)\s+"
    r"(?:an?\s+)?(?P<object>[\w -]+?)\??$"
    r"|^(?:can|does)\s+(?:an?\s+)?(?P<subject2>[\w -]+?)\s+"
    r"(?P<object2>[\w -]+?)\??$")


# An open question: not "is a modem a device" but "what is a modem" - tell
# me everything you have about this one thing. It is the first thing anybody
# types after telling it something, and answering it with generated prose is
# precisely the dishonesty this module exists to avoid.
_ABOUT = re.compile(
    r"^(?:what(?:'?s| is| are)\s+(?:an?\s+)?(?P<a>[\w -]+?)"
    r"|(?:tell me|what do you know)\s+about\s+(?:an?\s+)?(?P<b>[\w -]+?)"
    r"|who\s+(?:is|was)\s+(?P<c>[\w -]+?))\??$", re.IGNORECASE)


def parse_about(text: str) -> str | None:
    """The subject of an open question, or None if it is not one."""
    m = _ABOUT.match(_tidy(text).strip())
    if not m:
        return None
    name = m.group("a") or m.group("b") or m.group("c") or ""
    name = _bare(name)
    # "what is it", "what are you" - pronouns are not subjects it holds
    # facts about, and answering them from the knowledge base would be worse
    # than passing them on.
    if name in ("it", "this", "that", "you", "i", "me", "there", "the time"):
        return None
    return name or None


def parse_question(text: str):
    """A question about one fact, or None."""
    line = _tidy(text).rstrip("?")
    m = _QUESTION.match(line + "?") or _QUESTION.match(line)
    if not m:
        return None
    subject = m.group("subject") or m.group("subject2")
    obj = m.group("object") or m.group("object2")
    return _bare(subject), _bare(obj)


# ---- the base --------------------------------------------------------------


class Knowledge:
    """What MotherBrain has been told, and what follows from it.

    Kept beside the model in the run directory, so knowledge survives a
    restart the way weights do. Facts derived by inference are not stored -
    they are recomputed, so that retracting a premise retracts everything
    built on it rather than leaving orphans behind.
    """

    def __init__(self, run_dir, create: bool = True) -> None:
        self.run = Path(run_dir)
        self.path = self.run / "knowledge.json"
        self.facts: list[Fact] = []
        self.rules: list[Rule] = []
        if self.path.exists():
            self.load()
        elif create:
            self.run.mkdir(parents=True, exist_ok=True)

    # -- persistence --

    def load(self) -> None:
        with open(self.path) as fh:
            data = json.load(fh)
        self.facts = [Fact(*f) for f in data.get("facts", [])]
        self.rules = [Rule(*r) for r in data.get("rules", [])]

    def save(self) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"facts": [f.as_list() for f in self.facts],
                       "rules": [[r.when, r.then, r.relation]
                                 for r in self.rules]}, fh, indent=2)

    # -- telling it things --

    def tell(self, text: str):
        """Take in one statement. Returns what was understood, or None."""
        parsed = parse_statement(text)
        if parsed is None:
            return None
        if isinstance(parsed, Rule):
            if parsed not in self.rules:
                self.rules.append(parsed)
        elif parsed not in self.facts:
            self.facts.append(parsed)
        self.save()
        return parsed

    def forget(self, text: str) -> bool:
        """Retract a statement. Anything derived from it goes with it, because
        derivations are recomputed rather than stored."""
        parsed = parse_statement(text)
        if parsed is None:
            return False
        before = len(self.facts) + len(self.rules)
        self.facts = [f for f in self.facts if f != parsed]
        self.rules = [r for r in self.rules if r != parsed]
        self.save()
        return len(self.facts) + len(self.rules) < before

    # -- working out what follows --

    def closure(self) -> tuple[set[Fact], dict]:
        """Everything that follows, and one reason for each derived fact.

        Forward chaining to a fixed point. The world is small enough that
        deriving everything up front is cheaper than searching backwards per
        question, and it makes "what do you know?" answerable.
        """
        known = set(self.facts)
        because: dict[Fact, str] = {}
        changed = True
        while changed:
            changed = False
            for fact in list(known):
                if fact.relation not in ("is a", "is"):
                    continue
                for rule in self.rules:
                    if singular(fact.object) != singular(rule.when):
                        continue
                    derived = Fact(fact.subject, rule.relation, rule.then)
                    if derived not in known:
                        known.add(derived)
                        because[derived] = (f"{fact} , and {rule}"
                                            .replace(" ,", ","))
                        changed = True
        return known, because

    def ask(self, subject: str, obj: str) -> Derivation:
        """Does this hold? Yes, no, or - honestly - unknown."""
        subject, obj = subject.strip().lower(), obj.strip().lower()
        known, because = self.closure()

        denied = Fact(subject, "is not", obj)
        if denied in known:
            return Derivation(False, True, [f"I was told {denied}."])

        for relation in ("is", "is a", "can", "cannot", "has", "have",
                         "needs", "need"):
            target = Fact(subject, relation, obj)
            if target not in known:
                continue
            if target in self.facts:
                return Derivation(True, True, [f"I was told {target}."])
            chain = [f"{because[target]},"]
            chain.append(f"so {target}.")
            return Derivation(True, True, chain)

        return Derivation(False, False)

    # -- reporting --

    def about(self, name: str) -> Derivation:
        """Everything held about one subject, told and derived, with reasons."""
        name = _bare(name)
        known, because = self.closure()
        told, follows = [], []
        for fact in sorted(known, key=str):
            if singular(fact.subject) != singular(name):
                continue
            if fact in self.facts:
                told.append(f"I was told {fact}.")
            else:
                follows.append(f"{because[fact]}, so {fact}.")
        # A rule about the thing itself is worth saying even with no facts.
        for rule in self.rules:
            if singular(rule.when) == singular(name):
                told.append(f"I was told {rule}.")
        if not told and not follows:
            return Derivation(False, False)
        return Derivation(True, True, told + follows,
                          lead=f"Here is everything I have been told about "
                               f"{name}, and what follows from it.")

    def summary(self) -> str:
        known, because = self.closure()
        derived = [f for f in known if f not in set(self.facts)]
        lines = [f"{len(self.facts)} fact(s) told, {len(self.rules)} rule(s), "
                 f"{len(derived)} thing(s) that follow."]
        for fact in self.facts[:12]:
            lines.append(f"  told     {fact}")
        for rule in self.rules[:12]:
            lines.append(f"  rule     {rule}")
        for fact in derived[:12]:
            lines.append(f"  follows  {fact}")
        return "\n".join(lines)
