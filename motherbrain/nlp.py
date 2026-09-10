"""Understanding a sentence, and answering it in one.

A 52M-parameter base model trained on source code cannot answer a
question. Sampled freely it produces fluent English about nothing, which is
the worst failure available to a chat box: it looks exactly like an answer.

The way out is not a bigger model. It is to stop asking the model to do the
one thing it cannot, and to build the part it was never going to supply -
the part that reads a sentence, works out what is being asked, finds
something that is actually true, and composes a reply that is grammatical
and follows from the evidence.

That is what this is. Four stages, all of them inspectable:

1. **Analysis.** Tokenise, tag, lemmatise. The tagger is rule-based - a
   closed-class lexicon plus suffix rules - because a statistical tagger
   trained on nothing is worse than a table, and a table can be read.

2. **Interpretation.** What kind of utterance is it - a question, a
   statement, a command, a greeting - and what is its subject, its verb,
   its object, and the thing it is actually asking about.

3. **Evidence.** Where the answer comes from: arithmetic that can be
   computed, something MotherBrain was told, something it can read off its
   own state, or a sentence it has actually read in its corpus. Never the
   sampler.

4. **Composition.** The reply is built from the evidence by a small
   grammar that gets agreement, number and pronouns right, so what comes
   out is a sentence rather than a fragment - and is true, because it was
   assembled from something true rather than sampled.

When there is no evidence, the honest output is "I do not know", and this
says it. That is the entire difference between this and a chat box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- tokens -----------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def tokenise(text: str) -> list[str]:
    """Words, numbers and punctuation, kept separate. Contractions stay whole."""
    return _TOKEN.findall(text)


def sentences(text: str) -> list[str]:
    """Split on terminal punctuation, without splitting decimals or e.g."""
    out, current = [], []
    protected = {"e.g", "i.e", "mr", "mrs", "dr", "vs", "etc", "no"}
    for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not piece:
            continue
        # A piece that is nothing but punctuation has no last word.
        words = piece.rstrip(".!?").split()
        word = words[-1].lower() if words else ""
        current.append(piece)
        if word in protected:
            continue
        out.append(" ".join(current))
        current = []
    if current:
        out.append(" ".join(current))
    return [s for s in out if s.strip()]


# ---- tagging ----------------------------------------------------------------

# The closed classes, in full. These are the words a language does not add
# to, which is exactly why listing them works where guessing does not.
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my",
               "your", "his", "her", "its", "our", "their", "some", "any",
               "no", "every", "each", "another", "both", "all"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us",
            "them", "who", "whom", "what", "which", "myself", "yourself",
            "itself", "something", "anything", "nothing", "everything"}
PREPOSITIONS = {"of", "in", "to", "for", "with", "on", "at", "by", "from",
                "about", "into", "over", "after", "under", "between",
                "through", "during", "without", "within", "against",
                "before", "than", "like", "as"}
CONJUNCTIONS = {"and", "or", "but", "if", "because", "while", "although",
                "unless", "since", "so", "that", "when", "whether"}
BE = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re"}
HAVE = {"have", "has", "had", "'ve"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may",
          "might", "must", "do", "does", "did", "cannot", "won't", "don't"}
WH = {"what", "who", "whom", "whose", "which", "when", "where", "why",
      "how"}
ADVERBS = {"not", "n't", "very", "quite", "really", "always", "never",
           "often", "sometimes", "here", "there", "now", "then", "too",
           "also", "just", "only", "still", "well", "much", "more", "most"}

# Suffixes that give a word away. Ordered longest-first so -ation beats -on.
_SUFFIX_TAGS = [
    ("ational", "ADJ"), ("iveness", "NOUN"), ("fulness", "NOUN"),
    ("ousness", "NOUN"), ("ization", "NOUN"), ("isation", "NOUN"),
    ("ability", "NOUN"), ("ibility", "NOUN"),
    ("ation", "NOUN"), ("ition", "NOUN"), ("ement", "NOUN"),
    ("ness", "NOUN"), ("ship", "NOUN"), ("ment", "NOUN"), ("ance", "NOUN"),
    ("ence", "NOUN"), ("hood", "NOUN"), ("ity", "NOUN"), ("ist", "NOUN"),
    ("ism", "NOUN"), ("ure", "NOUN"), ("age", "NOUN"), ("ery", "NOUN"),
    ("ing", "VERB"), ("ise", "VERB"), ("ize", "VERB"), ("ify", "VERB"),
    ("ate", "VERB"),
    ("able", "ADJ"), ("ible", "ADJ"), ("ical", "ADJ"), ("ive", "ADJ"),
    ("ous", "ADJ"), ("ful", "ADJ"), ("less", "ADJ"), ("ish", "ADJ"),
    ("ic", "ADJ"), ("al", "ADJ"), ("ed", "VERB"),
    ("ly", "ADV"),
]


def tag(tokens: list[str]) -> list[tuple[str, str]]:
    """(token, tag) for each token. Closed classes first, then suffixes.

    The tags are the coarse universal set - NOUN, VERB, ADJ, ADV, DET,
    PRON, ADP, CONJ, NUM, PUNCT, AUX - because that is as fine as a
    rule-based tagger can be right about, and a tagger that claims more
    than it knows makes the parser downstream confidently wrong.
    """
    out: list[tuple[str, str]] = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        if not token[:1].isalnum():
            out.append((token, "PUNCT"))
        elif token[0].isdigit():
            out.append((token, "NUM"))
        elif lower in BE or lower in HAVE:
            out.append((token, "AUX"))
        elif lower in MODALS:
            out.append((token, "AUX"))
        elif lower in DETERMINERS:
            out.append((token, "DET"))
        elif lower in PRONOUNS:
            out.append((token, "PRON"))
        elif lower in PREPOSITIONS:
            out.append((token, "ADP"))
        elif lower in CONJUNCTIONS:
            out.append((token, "CONJ"))
        elif lower in ADVERBS:
            out.append((token, "ADV"))
        else:
            out.append((token, _guess(lower, out)))
    return out


def _guess(word: str, so_far: list[tuple[str, str]]) -> str:
    """A word not in any closed class. Its shape, and what came before it."""
    for suffix, guess in _SUFFIX_TAGS:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            # "the meaning" is a noun; "is meaning" is a verb.
            if guess == "VERB" and so_far and so_far[-1][1] == "DET":
                return "NOUN"
            return guess
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return "NOUN"
    # After a determiner or an adjective, the default is a noun; after a
    # pronoun, a verb. That one rule fixes most of what is left.
    if so_far:
        if so_far[-1][1] in ("DET", "ADJ"):
            return "NOUN"
        if so_far[-1][1] == "PRON":
            return "VERB"
    return "NOUN"


_IRREGULAR = {
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be",
    "been": "be", "being": "be", "has": "have", "had": "have",
    "does": "do", "did": "do", "went": "go", "gone": "go", "made": "make",
    "said": "say", "knew": "know", "known": "know", "thought": "think",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "data": "datum", "criteria": "criterion",
}


def lemma(word: str, pos: str = "NOUN") -> str:
    """The dictionary form. Small, honest, and wrong about the same words
    every English stemmer is wrong about."""
    lower = word.lower()
    if lower in _IRREGULAR:
        return _IRREGULAR[lower]
    if pos == "VERB":
        for suffix, add in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""),
                            ("s", "")):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                stem = lower[:-len(suffix)] + add
                if suffix in ("ing", "ed") and len(stem) > 2 \
                        and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                    stem = stem[:-1]          # running -> run
                return stem
        return lower
    if pos not in ("NOUN", "PRON", ""):
        # Only nouns take a plural off. "conscious" is not "consciou", and
        # a lemmatiser that mangles adjectives poisons every keyword match
        # downstream of it.
        return lower
    for suffix, add in (("ies", "y"), ("ses", "s"), ("xes", "x"),
                        ("zes", "z"), ("ches", "ch"), ("shes", "sh"),
                        ("s", "")):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            if suffix == "s" and lower.endswith("ss"):
                return lower
            return lower[:-len(suffix)] + add
    return lower


# ---- what the sentence is ---------------------------------------------------

STOP = (DETERMINERS | PRONOUNS | PREPOSITIONS | CONJUNCTIONS | BE | HAVE
        | MODALS | ADVERBS | {"please", "tell", "me", "give", "say"})

GREETINGS = {"hello", "hi", "hey", "greetings", "yo", "howdy", "morning",
             "afternoon", "evening"}
THANKS = {"thanks", "thank", "cheers", "ta"}
FAREWELL = {"bye", "goodbye", "farewell", "later", "night"}


@dataclass
class Utterance:
    """One sentence, read."""

    text: str
    tokens: list[str] = field(default_factory=list)
    tagged: list[tuple[str, str]] = field(default_factory=list)
    kind: str = "statement"        # question, statement, command, greeting…
    wh: str = ""                   # what/who/why/how, when it is a question
    subject: str = ""
    verb: str = ""
    object: str = ""
    topic: str = ""                # the content word it is really about
    keywords: list[str] = field(default_factory=list)
    negated: bool = False
    about_self: bool = False


def analyse(text: str) -> Utterance:
    """Read one sentence: what kind it is, and what it is about."""
    stripped = text.strip()
    tokens = tokenise(stripped)
    tagged = tag(tokens)
    u = Utterance(text=stripped, tokens=tokens, tagged=tagged)
    if not tokens:
        return u

    words = [t.lower() for t in tokens if t[:1].isalnum()]
    if not words:
        return u
    u.negated = any(w in ("not", "n't", "never", "no") for w in words)
    u.keywords = [lemma(token.lower(), pos) for token, pos in tagged
                  if token.lower() not in STOP
                  and pos in ("NOUN", "VERB", "ADJ", "NUM")]

    first = words[0]
    if first in GREETINGS:
        u.kind = "greeting"
    elif first in THANKS:
        u.kind = "thanks"
    elif first in FAREWELL:
        u.kind = "farewell"
    elif first in WH:
        u.kind, u.wh = "question", first
    elif first in BE or first in MODALS or first in HAVE:
        u.kind = "question"
    elif stripped.endswith("?"):
        u.kind = "question"
        u.wh = next((w for w in words if w in WH), "")
    elif tagged[0][1] == "VERB" and first not in BE:
        u.kind = "command"

    # Subject, verb, object: the first noun phrase, the first verb after it,
    # and whatever follows. Enough for the sentences people actually type.
    for i, (token, pos) in enumerate(tagged):
        if pos in ("NOUN", "PRON") and not u.subject:
            u.subject = token.lower()
        elif pos in ("VERB", "AUX") and u.subject and not u.verb:
            u.verb = lemma(token, "VERB")
        elif pos in ("NOUN", "PRON", "ADJ") and u.verb and not u.object:
            u.object = token.lower()

    u.about_self = any(w in ("you", "your", "yourself", "motherbrain")
                       for w in words)
    # The topic is the last content word, which in English is usually the
    # one being asked about: "what is a modem" -> modem.
    content = [lemma(t.lower(), pos) for t, pos in tagged
               if pos in ("NOUN", "ADJ") and t.lower() not in STOP]
    u.topic = content[-1] if content else ""
    return u


# ---- composition ------------------------------------------------------------

_PLURAL_PRONOUNS = {"they", "we", "you", "these", "those"}


def agree(subject: str, verb: str) -> str:
    """`be` and third-person -s, which is where a generated sentence gives
    itself away."""
    plural = (subject.lower() in _PLURAL_PRONOUNS
              or (subject.lower().endswith("s")
                  and not subject.lower().endswith("ss")))
    if verb == "be":
        if subject.lower() == "i":
            return "am"
        return "are" if plural else "is"
    if plural or subject.lower() in ("i", "you"):
        return verb
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def article(word: str) -> str:
    """a or an, by sound rather than by letter where it matters."""
    if not word:
        return "a"
    lower = word.lower()
    if lower[0] in "aeiou" and not lower.startswith(("eu", "one", "uni",
                                                     "use", "user")):
        return "an"
    if lower.startswith(("hour", "honest", "heir")):
        return "an"
    return "a"


def sentence(text: str) -> str:
    """One capital at the front, one full stop at the end, no double spaces."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def join(items: list[str], conjunction: str = "and") -> str:
    """A list, in English: a, b and c."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


# ---- retrieval --------------------------------------------------------------

def hits(query: list[str], candidate: str) -> int:
    """How many of the query's content words the sentence actually contains."""
    words = {lemma(w.lower(), "NOUN") for w in tokenise(candidate)
             if w[:1].isalnum() and w.lower() not in STOP}
    return sum(1 for term in query if term in words)


def relevance(query: list[str], candidate: str) -> float:
    """How well a sentence answers a query: shared content words, normalised.

    Deliberately simple and deliberately explainable. The point of grounding
    an answer in a sentence the model has actually read is that you can be
    shown the sentence, and a score you can recompute by hand is part of
    that.
    """
    if not query:
        return 0.0
    words = {lemma(w.lower(), "NOUN") for w in tokenise(candidate)
             if w[:1].isalnum() and w.lower() not in STOP}
    if not words:
        return 0.0
    hits = sum(1 for term in query if term in words)
    # Reward density as well as coverage, so a short sentence that is
    # entirely on the subject beats a long one that mentions it once.
    return hits / len(query) * (1.0 + hits / (len(words) + hits))


def search(query: list[str], index, limit: int = 3,
           floor: float = 0.5) -> list[tuple[float, str]]:
    """The best sentences for a query. Only the ones that could match."""
    scored: list[tuple[float, str]] = []
    if isinstance(index, Index):
        lines = (index.lines[i] for i in index.candidates(query))
    else:
        lines = index
    # One word in common is a coincidence: "fly me to the moon" matches
    # every sentence about doing something on the fly. Two is a subject.
    needed = min(2, len(query))
    for line in lines:
        if hits(query, line) < needed:
            continue
        score = relevance(query, line)
        if score >= floor:
            scored.append((score, line.strip()))
    scored.sort(key=lambda pair: -pair[0])
    out: list[tuple[float, str]] = []
    seen: set[str] = set()
    for score, line in scored:
        key = line.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append((score, line))
        if len(out) >= limit:
            break
    return out


# ---- answering --------------------------------------------------------------

@dataclass
class Answer:
    """A reply, and where it came from. The source is never decoration."""

    source: str                     # exact | known | self | read | none | social
    text: str
    evidence: list[str] = field(default_factory=list)

    def render(self) -> str:
        body = self.text
        if self.evidence:
            body += "\n" + "\n".join(f"  {line}" for line in self.evidence)
        return body


# What a greeting gets. Not generated: there is nothing to work out, and a
# sampled hello is the one place fluency costs nothing and buys nothing.
SOCIAL = {
    "greeting": "Hello. Ask me something I can work out, or tell me "
                "something and I will remember it.",
    "thanks": "You are welcome.",
    "farewell": "Goodbye.",
}


# Scanning a quarter of a billion characters per question would make every
# reply cost seconds. So it is read once into an inverted index - word to
# the sentences containing it - and a question then touches only the
# sentences that contain one of its words, which is a few hundred rather
# than a few hundred thousand.
class Index:
    """Sentences from the corpus, and which words are in which."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.postings: dict[str, list[int]] = {}

    def add(self, line: str) -> None:
        position = len(self.lines)
        self.lines.append(line)
        for word in {lemma(w.lower(), "NOUN") for w in tokenise(line)
                     if w[:1].isalnum() and w.lower() not in STOP}:
            self.postings.setdefault(word, []).append(position)

    def candidates(self, query: list[str]) -> list[int]:
        """Only the sentences that contain at least one word of the query."""
        found: set[int] = set()
        for term in query:
            found.update(self.postings.get(term, ()))
            if len(found) > 40_000:                # a word in everything is
                break                              # no use for narrowing
        return sorted(found)

    def __len__(self) -> int:
        return len(self.lines)


_INDEX: dict[str, Index] = {}
_INDEX_LIMIT = 250_000


def corpus_index(corpus_dir, documents: int = 12_000) -> Index:
    """The inverted index over the corpus, built once and kept."""
    key = str(corpus_dir)
    cached = _INDEX.get(key)
    if cached is not None:
        return cached

    index = Index()
    try:
        from motherbrain.data import Corpus

        corpus = Corpus(corpus_dir)
        for i, document in enumerate(corpus.texts()):
            if i >= documents or len(index) >= _INDEX_LIMIT:
                break
            for line in sentences(document):
                line = line.strip()
                if 20 <= len(line) <= 400:
                    index.add(line)
    except Exception:                                     # noqa: BLE001
        pass
    _INDEX[key] = index
    return index


def forget_index() -> None:
    """Drop the cache - after a patch, when the corpus has grown."""
    _INDEX.clear()


def answer(text: str, *, run_dir=None, corpus_dir=None, stats=None,
           ground: bool = True) -> Answer:
    """Read the sentence, find evidence, and compose an English reply.

    The order is the order of certainty: arithmetic that can be computed,
    then something MotherBrain was told, then something it can read off its
    own state, then a sentence it has actually read. Generation is not in
    this list, and that is the point - everything here is either true or
    labelled as somebody else's words.
    """
    u = analyse(text)
    if not u.tokens:
        return Answer("none", "You did not say anything.")

    if u.kind in SOCIAL:
        return Answer("social", SOCIAL[u.kind])

    # 1. Anything exactly computable.
    try:
        from motherbrain.logic import solve

        exact = solve(u.text)
        if exact is not None:
            return Answer("exact", exact.render())
    except Exception:                                     # noqa: BLE001
        pass

    def from_state():
        """What it can read off its own state, if this is that question."""
        if stats is None:
            return None
        try:
            from motherbrain.chat import answer_about_self, classify

            kind = classify(u.text)
            if kind:
                return Answer("self", answer_about_self(kind, stats))
        except Exception:                                 # noqa: BLE001
            return None
        return None

    # 2. A question about itself is answered from itself first. The
    #    knowledge base holds facts about the world; sending "are you an AI"
    #    to it means answering "I do not know" about the one subject it has
    #    complete information on.
    if u.about_self:
        mine = from_state()
        if mine is not None:
            return mine

    # 3. Anything it was told, or that follows from it.
    if run_dir is not None:
        try:
            from motherbrain.chat import consider

            considered = consider(u.text, run_dir)
            if considered is not None:
                return Answer("known", considered[1])
        except Exception:                                 # noqa: BLE001
            pass

    mine = from_state()
    if mine is not None:
        return mine

    # 4. Anything it has actually read. Quoted, attributed to the corpus,
    #    and never paraphrased - paraphrasing is where a retrieval system
    #    starts inventing.
    if ground and corpus_dir is not None and u.keywords:
        found = search(u.keywords, corpus_index(corpus_dir))
        if found:
            lead = _lead_in(u)
            return Answer("read", lead,
                          [f'"{line}"' for _score, line in found])

    return Answer("none", _admit(u))


def _lead_in(u: Utterance) -> str:
    """The sentence that introduces quoted evidence, agreeing with the query."""
    subject = u.topic or "that"
    if u.kind == "question":
        return sentence(
            f"I have not been told anything about {subject}, but this is "
            f"what my corpus says")
    return sentence(f"This is what my corpus says about {subject}")


def _admit(u: Utterance) -> str:
    """Saying so. The whole design turns on this being a real answer."""
    if u.about_self:
        return sentence(
            "I cannot establish that about myself. I can tell you my "
            "version, my size, what I have learned and how well I can see")
    if u.kind == "question":
        topic = u.topic or "that"
        return sentence(
            f"I do not know. Nothing I have been told settles {topic}, and "
            f"I will not make something up to fill the gap. Teach me with "
            f'"{topic} is ..." and I will keep it')
    if u.kind == "command":
        return sentence(
            f"I cannot {u.verb or 'do that'}. I am a language model with a "
            f"command table in front of it, and that is not on the table")
    return sentence("I have nothing true to say about that")
