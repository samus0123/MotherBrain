"""Memory-mapped token shards and the batch loader that walks them.

Data prep writes flat ``.bin`` files of token ids plus a ``meta.json`` index.
Training never loads a shard into RAM -- windows are sliced straight out of a
``np.memmap``, so the corpus can be far larger than memory.

The loader is deterministic and resumable: every window in a split gets a stable
global index, epoch ``e`` visits them in the order given by a permutation seeded
with ``(seed, e)``, and each rank strides through its own slice. Restoring
``samples_seen`` reproduces the exact remaining stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["ShardIndex", "TokenDataset", "BatchLoader", "DTYPES"]

# Token id width on disk. uint16 covers vocabularies up to 65535.
DTYPES: dict[str, Any] = {"uint16": np.uint16, "uint32": np.uint32}


@dataclass
class ShardIndex:
    """The ``meta.json`` sitting next to a set of token shards."""

    vocab_size: int
    dtype: str
    splits: dict[str, list[dict[str, Any]]]

    @classmethod
    def load(cls, data_dir: str | Path) -> ShardIndex:
        path = Path(data_dir) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no meta.json in {data_dir}; run scripts/prepare_data.py first"
            )
        with path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            vocab_size=int(meta["vocab_size"]),
            dtype=str(meta["dtype"]),
            splits={k: list(v) for k, v in meta["splits"].items()},
        )

    def save(self, data_dir: str | Path) -> None:
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {"vocab_size": self.vocab_size, "dtype": self.dtype, "splits": self.splits},
                fh,
                indent=2,
            )

    def tokens_in(self, split: str) -> int:
        if split not in self.splits:
            raise KeyError(f"split {split!r} not in {sorted(self.splits)}")
        return sum(int(s["tokens"]) for s in self.splits[split])


class TokenDataset:
    """Non-overlapping ``seq_len + 1`` windows over the shards of one split."""

    def __init__(self, data_dir: str | Path, split: str, seq_len: int) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.index = ShardIndex.load(self.data_dir)
        if split not in self.index.splits:
            raise KeyError(f"split {split!r} not in {sorted(self.index.splits)}")
        if self.index.dtype not in DTYPES:
            raise ValueError(f"unsupported shard dtype {self.index.dtype!r}")
        self.np_dtype = DTYPES[self.index.dtype]

        self._shard_paths: list[Path] = []
        # Windows per shard; a window needs seq_len inputs plus one shifted target.
        counts: list[int] = []
        for shard in self.index.splits[split]:
            self._shard_paths.append(self.data_dir / shard["path"])
            counts.append(max(0, (int(shard["tokens"]) - 1) // seq_len))
        self._counts = np.asarray(counts, dtype=np.int64)
        # cum[i] is the number of windows before shard i, so a global window index
        # maps to a shard with one searchsorted.
        self._cum = np.concatenate([[0], np.cumsum(self._counts)])
        self._memmaps: dict[int, np.memmap] = {}

        if self.num_windows == 0:
            raise ValueError(
                f"split {split!r} has no complete windows at seq_len={seq_len}; "
                "the shards are shorter than one sequence"
            )

    @property
    def num_windows(self) -> int:
        return int(self._cum[-1])

    def __len__(self) -> int:
        return self.num_windows

    def _memmap(self, shard_id: int) -> np.memmap:
        mm = self._memmaps.get(shard_id)
        if mm is None:
            mm = np.memmap(self._shard_paths[shard_id], dtype=self.np_dtype, mode="r")
            self._memmaps[shard_id] = mm
        return mm

    def locate(self, window: int) -> tuple[int, int]:
        """Map a global window index to ``(shard_id, token_offset)``."""
        if not 0 <= window < self.num_windows:
            raise IndexError(f"window {window} out of range [0, {self.num_windows})")
        shard_id = int(np.searchsorted(self._cum, window, side="right") - 1)
        local = window - int(self._cum[shard_id])
        return shard_id, local * self.seq_len

    def __getitem__(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        shard_id, offset = self.locate(window)
        chunk = np.asarray(self._memmap(shard_id)[offset : offset + self.seq_len + 1])
        # Cast to int64 for the embedding lookup; copy so torch owns the memory.
        chunk = chunk.astype(np.int64)
        return chunk[:-1], chunk[1:]


class BatchLoader:
    """An endless, resumable stream of ``(x, y)`` batches for one split."""

    def __init__(
        self,
        dataset: TokenDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)

        # Drop the tail so every rank sees the same number of windows per epoch.
        self.windows_per_rank = dataset.num_windows // world_size
        if self.windows_per_rank == 0:
            raise ValueError(
                f"split {dataset.split!r} has {dataset.num_windows} windows, "
                f"fewer than world_size={world_size}"
            )
        self.samples_seen = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.windows_per_rank // self.batch_size

    def _epoch_order(self, epoch: int) -> np.ndarray:
        """Global window ids assigned to this rank for ``epoch``, in visit order."""
        n = self.windows_per_rank * self.world_size
        if self.shuffle:
            rng = np.random.default_rng([self.seed, epoch])
            order = rng.permutation(self.dataset.num_windows)[:n]
        else:
            order = np.arange(n)
        return order[self.rank :: self.world_size]

    def __iter__(self) -> BatchLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs = np.empty((self.batch_size, self.dataset.seq_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.dataset.seq_len), dtype=np.int64)
        for i in range(self.batch_size):
            epoch, pos = divmod(self.samples_seen, self.windows_per_rank)
            window = int(self._epoch_order(epoch)[pos])
            xs[i], ys[i] = self.dataset[window]
            self.samples_seen += 1

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if self.device.type == "cuda":
            # Pinned + non_blocking lets the copy overlap the previous step.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def state_dict(self) -> dict[str, Any]:
        return {
            "samples_seen": self.samples_seen,
            "seed": self.seed,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # The stream is only reproducible if the shape of the run is unchanged.
        for key in ("seed", "world_size", "batch_size"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(
                    f"cannot resume loader: {key} was {state[key]} at checkpoint time "
                    f"but is {getattr(self, key)} now"
                )
        self.samples_seen = int(state["samples_seen"])
