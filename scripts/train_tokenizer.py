#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer over a text corpus.

    python scripts/train_tokenizer.py --input data/raw/*.txt \
        --vocab-size 32000 --out data/tokenized/tokenizer.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motherbrain.logging_utils import format_count, setup_logging  # noqa: E402
from motherbrain.tokenizer import Tokenizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="text file(s) to train on")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", default="data/tokenized/tokenizer.json")
    args = parser.parse_args(argv)

    logger = setup_logging()
    files: list[Path] = []
    for pattern in args.input:
        path = Path(pattern)
        matches = sorted(path.parent.glob(path.name)) if "*" in path.name else [path]
        files.extend(m for m in matches if m.is_file())
    if not files:
        logger.error("no input files matched %s", args.input)
        return 1

    total_bytes = sum(f.stat().st_size for f in files)
    logger.info(
        "training BPE (vocab=%d) on %d file(s), %sB of text",
        args.vocab_size,
        len(files),
        format_count(total_bytes),
    )
    tokenizer = Tokenizer.train(files, args.vocab_size, args.out, args.min_frequency)
    logger.info("wrote %s with %d tokens", args.out, tokenizer.vocab_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
