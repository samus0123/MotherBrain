"""Byte-level BPE tokenizer built on HuggingFace ``tokenizers``.

The trained vocabulary is a single portable ``tokenizer.json``. This module is a
thin wrapper that pins the special-token contract the rest of the stack relies
on: ids 0/1/2 are ``<pad>``/``<bos>``/``<eos>``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["Tokenizer", "PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "SPECIAL_TOKENS"]

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


def _require_tokenizers():  # pragma: no cover - trivial import shim
    try:
        import tokenizers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the `tokenizers` package is required for tokenizer support; "
            "install it with `pip install 'motherbrain[tokenizer]'`"
        ) from exc
    return tokenizers


class Tokenizer:
    """Encode/decode text with a trained byte-level BPE vocabulary."""

    def __init__(self, backend) -> None:
        self._tok = backend
        self.pad_id = self._token_id(PAD_TOKEN)
        self.bos_id = self._token_id(BOS_TOKEN)
        self.eos_id = self._token_id(EOS_TOKEN)

    def _token_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise ValueError(f"tokenizer is missing the required special token {token!r}")
        return int(tid)

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        tokenizers = _require_tokenizers()
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"tokenizer not found: {path}")
        return cls(tokenizers.Tokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        files: Iterable[str | Path],
        vocab_size: int,
        out_path: str | Path,
        min_frequency: int = 2,
    ) -> Tokenizer:
        """Train byte-level BPE over ``files`` and write ``tokenizer.json``."""
        tokenizers = _require_tokenizers()
        from tokenizers import decoders, models, pre_tokenizers, trainers

        files = [str(f) for f in files]
        if not files:
            raise ValueError("no training files given")

        backend = tokenizers.Tokenizer(models.BPE(unk_token=None))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        backend.train(files, trainer)

        out_path = Path(out_path)
        if out_path.is_dir():
            out_path = out_path / "tokenizer.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        backend.save(str(out_path))
        return cls(backend)

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        encoded = self._tok.encode_batch(texts, add_special_tokens=False)
        out = []
        for enc in encoded:
            ids = list(enc.ids)
            if add_bos:
                ids = [self.bos_id, *ids]
            if add_eos:
                ids = [*ids, self.eos_id]
            out.append(ids)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def stream_encode(
        self, texts: Iterable[str], add_bos: bool = True, add_eos: bool = True, chunk: int = 1000
    ) -> Iterator[list[int]]:
        """Encode an iterable of documents in batches, yielding one id list each."""
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= chunk:
                yield from self.encode_batch(batch, add_bos, add_eos)
                batch = []
        if batch:
            yield from self.encode_batch(batch, add_bos, add_eos)
