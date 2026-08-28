#!/usr/bin/env python3
"""Build a small demo corpus so the pipeline can be run end to end offline.

Concatenates the Python standard library's own source files into
`data/corpus/`. It is a real, structured corpus with plenty of repeated
patterns, which makes it easy to see a small model actually learning.

Point `motherbrain tokenizer` / `motherbrain data` at your own text files to
train on something you care about instead.
"""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/corpus")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=6_000_000)
    args = ap.parse_args()

    stdlib = Path(sysconfig.get_paths()["stdlib"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in stdlib.glob("*.py") if p.stat().st_size > 2000)
    written, total = 0, 0
    chunks: list[str] = []
    for path in sources[: args.max_files]:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        chunks.append(text)
        total += len(text)
        written += 1
        if total >= args.max_bytes:
            break

    out_path = out_dir / "corpus.txt"
    # A blank-line-free sentinel between documents, so `--split-on` can recover them.
    out_path.write_text("\n<|document|>\n".join(chunks), encoding="utf-8")
    print(f"wrote {out_path}: {written} documents, {total:,} characters")


if __name__ == "__main__":
    main()
