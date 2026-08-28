#!/usr/bin/env python3
"""Tokenize a corpus into flat ``.bin`` shards plus a ``meta.json`` index.

Reads either plain text files (one document per file, or one per line with
``--line-docs``) or a HuggingFace dataset, and streams tokens to disk so the
corpus never has to fit in memory.

    python scripts/prepare_data.py --input data/raw/*.txt \
        --tokenizer data/tokenized/tokenizer.json --out data/tokenized

    python scripts/prepare_data.py --hf-dataset wikitext --hf-config wikitext-103-raw-v1 \
        --tokenizer data/tokenized/tokenizer.json --out data/tokenized
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motherbrain.data import DTYPES, ShardIndex  # noqa: E402
from motherbrain.logging_utils import format_count, setup_logging  # noqa: E402
from motherbrain.tokenizer import Tokenizer  # noqa: E402

logger = setup_logging()


def iter_text_files(paths: list[Path], line_docs: bool) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            if line_docs:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line
            else:
                yield fh.read()


def iter_hf_dataset(name: str, config: str | None, split: str, text_key: str) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "`datasets` is required for --hf-dataset; "
            "install it with `pip install 'motherbrain[data]'`"
        ) from exc
    stream = load_dataset(name, config, split=split, streaming=True)
    for row in stream:
        text = row.get(text_key)
        if text:
            yield text


class ShardWriter:
    """Buffer token ids and flush fixed-size shards to disk."""

    def __init__(self, out_dir: Path, split: str, dtype: str, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.np_dtype = DTYPES[dtype]
        self.shard_tokens = shard_tokens
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.shards: list[dict] = []

    def add(self, ids: list[int]) -> None:
        self.buffer.append(np.asarray(ids, dtype=self.np_dtype))
        self.buffered += len(ids)
        while self.buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count: int) -> None:
        if count == 0:
            return
        joined = np.concatenate(self.buffer)
        chunk, rest = joined[:count], joined[count:]
        path = self.out_dir / f"{self.split}_{len(self.shards):05d}.bin"
        chunk.tofile(path)
        self.shards.append({"path": path.name, "tokens": int(chunk.size)})
        logger.info("  wrote %s (%s tokens)", path.name, format_count(chunk.size))
        self.buffer = [rest] if rest.size else []
        self.buffered = int(rest.size)

    def close(self) -> list[dict]:
        self._flush(self.buffered)
        return self.shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenize a corpus into training shards.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", help="text file(s) to tokenize")
    source.add_argument("--hf-dataset", help="HuggingFace dataset name")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--line-docs", action="store_true", help="treat each line as a document")
    parser.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    parser.add_argument("--out", default="data/tokenized")
    parser.add_argument("--split", default="train", help="name of the split being written")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.005,
        help="hold out this fraction of documents as a val split (0 to disable)",
    )
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="stop after N documents")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.load(args.tokenizer)
    # uint16 holds ids up to 65535; wider vocabularies need 4 bytes per token.
    dtype = "uint16" if tokenizer.vocab_size <= np.iinfo(np.uint16).max else "uint32"
    logger.info("tokenizer vocab=%d -> shard dtype %s", tokenizer.vocab_size, dtype)

    if args.input:
        paths: list[Path] = []
        for pattern in args.input:
            path = Path(pattern)
            matches = sorted(path.parent.glob(path.name)) if "*" in path.name else [path]
            paths.extend(m for m in matches if m.is_file())
        if not paths:
            logger.error("no input files matched %s", args.input)
            return 1
        docs = iter_text_files(paths, args.line_docs)
    else:
        docs = iter_hf_dataset(args.hf_dataset, args.hf_config, args.hf_split, args.hf_text_key)

    if args.limit is not None:
        docs = (doc for _, doc in zip(range(args.limit), docs, strict=False))

    writers = {args.split: ShardWriter(out_dir, args.split, dtype, args.shard_tokens)}
    if args.val_fraction > 0:
        writers["val"] = ShardWriter(out_dir, "val", dtype, args.shard_tokens)

    rng = np.random.default_rng(args.seed)
    n_docs = 0
    n_tokens = 0
    for ids in tokenizer.stream_encode(docs, add_bos=True, add_eos=True):
        # Route whole documents to val so no sequence straddles the split.
        target = "val" if ("val" in writers and rng.random() < args.val_fraction) else args.split
        writers[target].add(ids)
        n_docs += 1
        n_tokens += len(ids)
        if n_docs % 10000 == 0:
            logger.info("  %s docs, %s tokens", format_count(n_docs), format_count(n_tokens))

    splits = {}
    for name, writer in writers.items():
        shards = writer.close()
        if shards:
            splits[name] = shards

    # Merge with any splits written by a previous invocation.
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        existing = ShardIndex.load(out_dir)
        if existing.dtype != dtype or existing.vocab_size != tokenizer.vocab_size:
            logger.error("existing meta.json was built with a different tokenizer; aborting")
            return 1
        merged = dict(existing.splits)
        merged.update(splits)
        splits = merged

    ShardIndex(vocab_size=tokenizer.vocab_size, dtype=dtype, splits=splits).save(out_dir)
    for name, shards in splits.items():
        logger.info(
            "split %-6s %s tokens across %d shard(s)",
            name,
            format_count(sum(s["tokens"] for s in shards)),
            len(shards),
        )
    logger.info("wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
