"""Token packing and batch loading.

Text is tokenized once into a flat `uint16`/`uint32` binary file per split. At
train time the file is memory-mapped, so the dataset never has to fit in RAM and
startup is instant regardless of corpus size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np
import torch


class SupportsEncode(Protocol):
    def encode(self, text: str, allowed_special: bool = True) -> list[int]: ...


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Smallest unsigned integer type that can hold every token id."""
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size} is too large")


def pack_documents(
    documents: Iterable[str],
    tokenizer: SupportsEncode,
    out_path: str | Path,
    vocab_size: int,
    eot_id: int | None = None,
    flush_every: int = 1024,
) -> int:
    """Tokenize `documents` into a flat binary file. Returns the token count.

    An end-of-text token is appended after each document so the model learns
    where documents stop instead of running them together.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = dtype_for_vocab(vocab_size)

    total = 0
    buffer: list[int] = []
    with open(out_path, "wb") as f:
        for i, doc in enumerate(documents):
            ids = tokenizer.encode(doc)
            if eot_id is not None:
                ids = ids + [eot_id]
            buffer.extend(ids)
            if (i + 1) % flush_every == 0 and buffer:
                np.asarray(buffer, dtype=dtype).tofile(f)
                total += len(buffer)
                buffer = []
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(f)
            total += len(buffer)
    return total


def write_meta(out_dir: str | Path, vocab_size: int, splits: dict[str, int]) -> None:
    meta = {"vocab_size": vocab_size, "dtype": str(dtype_for_vocab(vocab_size)), "splits": splits}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `motherbrain data` to build a token dataset first."
        )
    return json.loads(path.read_text())


class TokenDataset:
    """A memory-mapped flat array of token ids.

    `sample_batch` draws uniformly random windows, which is the standard recipe
    for pretraining on a shuffled corpus and avoids holding an index of
    document boundaries.
    """

    def __init__(self, path: str | Path, dtype: np.dtype | str = np.uint16) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token file {self.path} does not exist")
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
        generator: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, targets), each (batch_size, seq_len)."""
        if len(self) < seq_len + 1:
            raise ValueError(
                f"dataset has {len(self)} tokens, need at least seq_len+1 = {seq_len + 1}"
            )
        rng = generator or np.random.default_rng()
        starts = rng.integers(0, len(self) - seq_len - 1, size=batch_size)
        # Copy out of the memmap before making tensors; np.stack materialises the
        # windows so the tensors do not alias the mapped file.
        x = np.stack([self.tokens[s : s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if str(device) != "cpu":
            xt = xt.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
        return xt, yt

    def iter_sequential(self, batch_size: int, seq_len: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Walk the split front to back, for deterministic evaluation."""
        stride = batch_size * seq_len
        limit = (len(self) - 1) // stride * stride
        for start in range(0, limit, stride):
            chunk = np.asarray(self.tokens[start : start + stride + 1], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1].reshape(batch_size, seq_len).copy())
            y = torch.from_numpy(chunk[1:].reshape(batch_size, seq_len).copy())
            yield x, y


def load_splits(data_dir: str | Path) -> tuple[dict[str, TokenDataset], dict]:
    meta = read_meta(data_dir)
    datasets = {}
    for split in meta["splits"]:
        path = Path(data_dir) / f"{split}.bin"
        if path.exists():
            datasets[split] = TokenDataset(path, meta["dtype"])
    return datasets, meta
