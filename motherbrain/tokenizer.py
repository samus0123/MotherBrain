"""A byte-level BPE tokenizer that MotherBrain trains on your own corpus.

Byte-level means there is no such thing as an unknown character: any byte
sequence round-trips exactly, including emoji, code, and binary-ish junk. The
merge table is learned from whatever you feed in, so a corpus of Python and a
corpus of Portuguese produce different, well-fitted vocabularies.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

# Special tokens occupy the first ids so they never collide with learned merges.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]
PAD_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID = range(len(SPECIAL_TOKENS))

# GPT-2-style pre-tokenization: keeps words, numbers and runs of whitespace from
# merging across each other, which keeps the learned vocabulary sane. Written
# with stdlib `re` classes rather than the \p{...} properties of `regex`.
# The branches must cover every character: [^\W\d] is letters *and* underscore,
# so identifiers like n_heads survive intact. tests/test_tokenizer.py asserts
# that concatenating the split pieces reproduces the input exactly.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class Tokenizer:
    """Byte-level BPE. Train it, save it, load it, use it."""

    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 vocab_size: int | None = None) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.specials = list(SPECIAL_TOKENS)
        self.n_special = len(self.specials)
        if merges:
            for rank, pair in enumerate(merges):
                self.merges[tuple(pair)] = self.n_special + 256 + rank
        self.vocab_size = vocab_size or (self.n_special + 256 + len(self.merges))
        self._vocab: dict[int, bytes] | None = None
        self._encode_cache: dict[str, list[int]] = {}

    # ---- construction ------------------------------------------------------

    @property
    def base_offset(self) -> int:
        """First id used for a raw byte."""
        return self.n_special

    def _byte_id(self, b: int) -> int:
        return self.base_offset + b

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096,
              verbose: bool = False) -> "Tokenizer":
        """Learn a merge table from a corpus.

        Counts are accumulated over pre-tokenized word types rather than raw
        positions, so cost scales with vocabulary diversity, not corpus length.
        Pair statistics are then maintained incrementally: applying a merge only
        touches the words that actually contained the merged pair, which keeps
        training on a multi-megabyte corpus to seconds rather than hours.
        """
        n_special = len(SPECIAL_TOKENS)
        n_merges = max(0, vocab_size - n_special - 256)

        word_freqs: Counter[str] = Counter()
        for text in texts:
            word_freqs.update(SPLIT_PATTERN.findall(text))
        if not word_freqs:
            raise ValueError("cannot train a tokenizer on an empty corpus")

        # Each distinct word becomes a list of byte ids, carrying its frequency.
        words: list[list[int]] = []
        freqs: list[int] = []
        for word, freq in word_freqs.items():
            words.append([n_special + b for b in word.encode("utf-8")])
            freqs.append(freq)

        # pair -> total frequency, and pair -> which words currently contain it.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[int]] = {}

        def add_word_pairs(i: int, sign: int) -> None:
            seq, freq = words[i], freqs[i]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += sign * freq
                if sign > 0:
                    pair_words.setdefault(pair, set()).add(i)

        for i in range(len(words)):
            add_word_pairs(i, 1)

        merges: list[tuple[int, int]] = []
        next_id = n_special + 256

        for step in range(n_merges):
            # Highest count wins; ties break on the pair itself so that training
            # the same corpus twice always yields the same merge table.
            best, count = None, 1
            for pair, c in pair_counts.items():
                if c > count or (c == count and best is not None and pair < best):
                    best, count = pair, c
            if best is None:
                break  # nothing left worth merging

            merges.append(best)
            a, b = best

            for i in list(pair_words.get(best, ())):
                seq = words[i]
                if len(seq) < 2:
                    continue
                add_word_pairs(i, -1)  # retract this word's old statistics
                out: list[int] = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                        out.append(next_id)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                words[i] = out
                add_word_pairs(i, 1)  # and post its new ones

            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            next_id += 1
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}  (pair seen {count:,}x)", flush=True)

        return cls(merges=merges, vocab_size=n_special + 256 + len(merges))

    # ---- encoding / decoding ----------------------------------------------

    def _encode_word(self, word: str) -> list[int]:
        cached = self._encode_cache.get(word)
        if cached is not None:
            return cached
        ids = [self._byte_id(b) for b in word.encode("utf-8")]
        while len(ids) >= 2:
            # Apply the lowest-ranked (earliest learned) applicable merge.
            best_rank = None
            best_pos = -1
            for i, pair in enumerate(zip(ids, ids[1:])):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                break
            ids[best_pos:best_pos + 2] = [best_rank]
        if len(self._encode_cache) < 200_000:
            self._encode_cache[word] = ids
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        out: list[int] = [BOS_ID] if bos else []
        for word in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_word(word))
        if eos:
            out.append(EOS_ID)
        return out

    @property
    def vocab(self) -> dict[int, bytes]:
        """id -> byte string, built lazily and cached."""
        if self._vocab is None:
            v: dict[int, bytes] = {i: s.encode("utf-8") for i, s in enumerate(self.specials)}
            for b in range(256):
                v[self._byte_id(b)] = bytes([b])
            for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
                v[idx] = v[a] + v[b]
            self._vocab = v
        return self._vocab

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        vocab = self.vocab
        chunks: list[bytes] = []
        for i in ids:
            i = int(i)
            if skip_special and i < self.n_special:
                continue
            piece = vocab.get(i)
            if piece is not None:
                chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "specials": self.specials,
                    "merges": [list(p) for p, _ in
                               sorted(self.merges.items(), key=lambda kv: kv[1])],
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path) as fh:
            d = json.load(fh)
        tok = cls(merges=[tuple(p) for p in d["merges"]], vocab_size=d["vocab_size"])
        tok.specials = d.get("specials", list(SPECIAL_TOKENS))
        tok.n_special = len(tok.specials)
        return tok

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
