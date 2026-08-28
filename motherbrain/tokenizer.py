"""Byte-level BPE tokenizer.

Pure Python and dependency-free. Every token maps to a byte string, so the
tokenizer never fails on unseen input: worst case it falls back to single
bytes. Special tokens are held outside the merge table and are matched before
the regex pre-tokenizer runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# A pre-tokenizer split that keeps words, numbers and punctuation apart, so BPE
# never learns merges that span a word boundary. Two details matter: `[^\W\d]`
# spells "letter or underscore" (Python's `\w` counts `_`, so the more obvious
# `[^\W\d_]` silently drops underscores), and the trailing `.` is a catch-all.
# Together they make the split *total* — re-joining the pieces reproduces the
# input exactly — which is what `encode`/`decode` rely on to be lossless.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+|.""",
    re.UNICODE | re.DOTALL,
)

END_OF_TEXT = "<|endoftext|>"
PAD = "<|pad|>"
DEFAULT_SPECIALS = (END_OF_TEXT, PAD)


class Tokenizer:
    """A trained byte-level BPE tokenizer.

    Attributes:
        merges: {(id_a, id_b): new_id} in the order they were learned.
        vocab: {id: bytes}.
        special_tokens: {text: id}.
    """

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.merges = dict(merges or {})
        self.special_tokens = dict(special_tokens or {})
        self.vocab = self._build_vocab()
        self._special_re = self._build_special_re()
        self._cache: dict[str, list[int]] = {}

    # -- construction -----------------------------------------------------
    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        for text, idx in self.special_tokens.items():
            vocab[idx] = text.encode("utf-8")
        return vocab

    def _build_special_re(self) -> re.Pattern[str] | None:
        if not self.special_tokens:
            return None
        # Longest first so <|endoftext|> wins over any prefix of itself.
        pattern = "|".join(
            re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
        )
        return re.compile(f"({pattern})")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        return self.special_tokens.get(END_OF_TEXT, 0)

    # -- training ---------------------------------------------------------
    @classmethod
    def train(
        cls,
        text: str | Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = DEFAULT_SPECIALS,
        verbose: bool = False,
    ) -> "Tokenizer":
        """Learn merges until the vocabulary reaches `vocab_size`."""
        specials = list(special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is too small for 256 byte tokens "
                f"plus {len(specials)} special tokens"
            )

        chunks: Counter[tuple[int, ...]] = Counter()
        texts = [text] if isinstance(text, str) else text
        for doc in texts:
            for piece in SPLIT_PATTERN.findall(doc):
                chunks[tuple(piece.encode("utf-8"))] += 1

        words = {word: list(word) for word in chunks}
        merges: dict[tuple[int, int], int] = {}

        for i in range(n_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, ids in words.items():
                freq = chunks[word]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            new_id = 256 + i
            merges[best] = new_id
            words = {word: _merge_ids(ids, best, new_id) for word, ids in words.items()}
            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i + 1}/{n_merges}: {best} -> {new_id} (count {count})")

        base = 256 + len(merges)
        special_map = {tok: base + j for j, tok in enumerate(specials)}
        return cls(merges=merges, special_tokens=special_map)

    # -- encoding ---------------------------------------------------------
    def _encode_chunk(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned applicable merge, mirroring training order.
            best = min(
                (self.merges.get(pair, float("inf")) for pair in zip(ids, ids[1:])),
                default=float("inf"),
            )
            if best == float("inf"):
                break
            pair = next(p for p in zip(ids, ids[1:]) if self.merges.get(p) == best)
            ids = _merge_ids(ids, pair, int(best))
        if len(piece) < 64:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not text:
            return []
        if allowed_special and self._special_re is not None:
            out: list[int] = []
            for part in self._special_re.split(text):
                if not part:
                    continue
                if part in self.special_tokens:
                    out.append(self.special_tokens[part])
                else:
                    out.extend(self._encode_ordinary(part))
            return out
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(piece))
        return out

    # -- decoding ---------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        parts = [self.vocab.get(int(i), b"") for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        p.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        raw = json.loads(Path(path).read_text())
        merges = {(a, b): idx for a, b, idx in raw["merges"]}
        return cls(merges=merges, special_tokens=raw.get("special_tokens", {}))


class ByteTokenizer:
    """A trivial 256-symbol fallback, handy for tests and tiny runs."""

    vocab_size = 257
    eot_id = 256
    special_tokens = {END_OF_TEXT: 256}

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
