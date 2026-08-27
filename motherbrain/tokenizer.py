"""Byte-level tokenizer.

Bytes mean any text can be fed in without training a vocabulary first: the
tokenizer is fixed, so a checkpoint from one corpus stays readable by the
next. Ids 0-255 are raw bytes; the three ids above them are control tokens.
"""

from __future__ import annotations

BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 259

SPECIAL_TOKENS = {"<bos>": BOS, "<eos>": EOS, "<pad>": PAD}


class ByteTokenizer:
    """Encodes text as UTF-8 bytes plus control tokens."""

    vocab_size = VOCAB_SIZE
    bos_id = BOS
    eos_id = EOS
    pad_id = PAD

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if bos:
            ids.insert(0, BOS)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids, *, skip_special: bool = True) -> str:
        payload = bytes(i for i in ids if i < 256)
        if not skip_special:
            payload = bytes(i if i < 256 else 63 for i in ids)  # '?' for specials
        # Generation can cut a multi-byte character in half; don't crash on it.
        return payload.decode("utf-8", errors="replace")
