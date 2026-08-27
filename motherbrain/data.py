"""Turning a pile of text into next-token training batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tokenizer import ByteTokenizer

TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".py", ".rst"}


def iter_text_files(source: Path) -> list[Path]:
    """Every text file under ``source`` (or ``source`` itself if it is a file)."""
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"no such file or directory: {source}")
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        raise ValueError(f"no text files found under {source}")
    return files


def load_corpus(source: Path | str, tokenizer: ByteTokenizer | None = None):
    """Read every text file under ``source`` into one array of token ids."""
    tokenizer = tokenizer or ByteTokenizer()
    chunks: list[np.ndarray] = []
    for path in iter_text_files(Path(source)):
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text, bos=True, eos=True)
        chunks.append(np.asarray(ids, dtype=np.int32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)


def make_windows(tokens: np.ndarray, seq_len: int):
    """Slice ``tokens`` into (input, target) pairs shifted by one position.

    Targets are the inputs shifted left, which is the whole training signal:
    predict the next token from everything before it.
    """
    usable = (len(tokens) - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(
            f"corpus has {len(tokens)} tokens, need more than {seq_len + 1} "
            "for a single training window"
        )
    inputs = tokens[:usable].reshape(-1, seq_len)
    targets = tokens[1 : usable + 1].reshape(-1, seq_len)
    return inputs, targets


def make_dataset(
    source: Path | str,
    seq_len: int,
    batch_size: int,
    *,
    validation_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 0,
):
    """Build shuffled train/validation ``tf.data`` datasets from a corpus."""
    import tensorflow as tf

    tokens = load_corpus(source)
    inputs, targets = make_windows(tokens, seq_len)

    n_val = int(len(inputs) * validation_split)
    if validation_split > 0 and n_val == 0 and len(inputs) > 1:
        n_val = 1  # always keep at least one held-out window if asked for any
    split = len(inputs) - n_val

    def build(x, y, do_shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if do_shuffle and len(x):
            dataset = dataset.shuffle(len(x), seed=seed)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train = build(inputs[:split], targets[:split], shuffle)
    validation = (
        build(inputs[split:], targets[split:], False) if n_val else None
    )
    return train, validation, len(tokens)
