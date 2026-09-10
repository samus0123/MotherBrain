"""Ingestion: turning whatever you feed MotherBrain into trainable tokens.

Text arrives as raw strings, files, or whole directory trees. It gets appended
to a corpus on disk, tokenized once into a flat uint32 array, and memory-mapped
at training time so corpora larger than RAM stream without ceremony.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from motherbrain.tokenizer import BOS_ID, EOS_ID, Tokenizer

# Every language MotherBrain will take in, by the extension it arrives
# under. The list is long on purpose: "teach it all programming languages"
# is a corpus problem, not a model problem, and a corpus that silently
# skipped Haskell because nobody had put ".hs" in a set would have made it
# look like the model could not learn Haskell.
#
# The name beside each suffix is what the census reports, so a person can
# see what it has actually read rather than what it was pointed at.
LANGUAGES = {
    ".py": "Python", ".pyx": "Cython", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala",
    ".groovy": "Groovy", ".clj": "Clojure", ".cljs": "Clojure",
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
    ".hpp": "C++", ".hh": "C++", ".m": "Objective-C", ".mm": "Objective-C",
    ".cs": "C#", ".fs": "F#", ".fsx": "F#", ".vb": "Visual Basic",
    ".rs": "Rust", ".go": "Go", ".zig": "Zig", ".d": "D", ".nim": "Nim",
    ".v": "V", ".cr": "Crystal", ".odin": "Odin",
    ".rb": "Ruby", ".pl": "Perl", ".pm": "Perl", ".php": "PHP",
    ".lua": "Lua", ".tcl": "Tcl", ".r": "R", ".jl": "Julia",
    ".hs": "Haskell", ".lhs": "Haskell", ".ml": "OCaml", ".mli": "OCaml",
    ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang", ".hrl": "Erlang",
    ".elm": "Elm", ".purs": "PureScript", ".rkt": "Racket",
    ".scm": "Scheme", ".ss": "Scheme", ".lisp": "Lisp", ".el": "Emacs Lisp",
    ".f": "Fortran", ".f90": "Fortran", ".f95": "Fortran",
    ".for": "Fortran",
    ".pas": "Pascal", ".pp": "Pascal", ".ada": "Ada", ".adb": "Ada",
    ".cob": "COBOL", ".cbl": "COBOL", ".bas": "BASIC", ".asm": "Assembly",
    ".s": "Assembly", ".S": "Assembly", ".vhd": "VHDL", ".sv": "SystemVerilog",
    ".swift": "Swift", ".dart": "Dart", ".sol": "Solidity",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell",
    ".ps1": "PowerShell", ".psm1": "PowerShell", ".bat": "Batch",
    ".cmd": "Batch", ".awk": "AWK", ".sed": "sed",
    ".sql": "SQL", ".psql": "SQL", ".hql": "SQL",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".sass": "Sass", ".less": "Less", ".vue": "Vue", ".svelte": "Svelte",
    ".json": "JSON", ".jsonl": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML", ".ini": "INI", ".cfg": "INI", ".xml": "XML",
    ".proto": "Protocol Buffers", ".graphql": "GraphQL", ".gql": "GraphQL",
    ".tf": "Terraform", ".hcl": "HCL", ".dockerfile": "Dockerfile",
    ".mk": "Make", ".cmake": "CMake", ".gradle": "Gradle",
    ".tex": "TeX", ".md": "Markdown", ".rst": "reStructuredText",
    ".txt": "text", ".csv": "data", ".tsv": "data", ".log": "text",
}

# Files with no extension that are source all the same.
NAMED_LANGUAGES = {
    "Makefile": "Make", "makefile": "Make", "GNUmakefile": "Make",
    "Dockerfile": "Dockerfile", "Containerfile": "Dockerfile",
    "Rakefile": "Ruby", "Gemfile": "Ruby", "Vagrantfile": "Ruby",
    "CMakeLists.txt": "CMake", "SConstruct": "Python", "BUILD": "Bazel",
    "WORKSPACE": "Bazel",
}

TEXT_SUFFIXES = set(LANGUAGES)


def language_of(path) -> str:
    """What language a file is, by its name. "unknown" if it is not source."""
    from pathlib import Path as _Path

    path = _Path(path)
    if path.name in NAMED_LANGUAGES:
        return NAMED_LANGUAGES[path.name]
    return LANGUAGES.get(path.suffix.lower(), "unknown")

TOKEN_DTYPE = np.uint32


@dataclass
class Corpus:
    """The on-disk home for everything the model has been fed.

    Layout under `root`:
        documents.jsonl   every ingested document, with its source and time
        tokens.bin        the whole corpus tokenized, flat
        tokenizer.json    the vocabulary learned from it
        meta.json         counts and timestamps
    """

    root: Path
    create: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        # Read-only callers (`mb status`) must not bring a workspace into
        # existence just by looking at one.
        if self.create:
            self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths -------------------------------------------------------------

    @property
    def docs_path(self) -> Path:
        return self.root / "documents.jsonl"

    @property
    def tokens_path(self) -> Path:
        return self.root / "tokens.bin"

    @property
    def tokenizer_path(self) -> Path:
        return self.root / "tokenizer.json"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    # ---- ingestion ---------------------------------------------------------

    def add_text(self, text: str, source: str = "inline") -> int:
        """Append one document. Returns its character count."""
        text = text.strip()
        if not text:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.docs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "source": source,
                "added_at": time.time(),
                "chars": len(text),
                "text": text,
            }, ensure_ascii=False) + "\n")
        return len(text)

    def add_file(self, path: str | Path) -> int:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return 0
        return self.add_text(text, source=str(path))

    def add_path(self, path: str | Path, recursive: bool = True) -> tuple[int, int]:
        """Ingest a file or a whole tree. Returns (files, characters)."""
        path = Path(path)
        if path.is_file():
            n = self.add_file(path)
            return (1 if n else 0), n
        files = chars = 0
        walker = path.rglob("*") if recursive else path.glob("*")
        for p in sorted(walker):
            if not p.is_file():
                continue
            if (p.suffix.lower() not in TEXT_SUFFIXES
                    and p.name not in NAMED_LANGUAGES):
                continue
            if any(part in {".git", "node_modules", "__pycache__", ".venv"}
                   for part in p.parts):
                continue
            n = self.add_file(p)
            if n:
                files += 1
                chars += n
        return files, chars

    # ---- reading -----------------------------------------------------------

    def documents(self) -> Iterator[dict]:
        if not self.docs_path.exists():
            return
        with open(self.docs_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def texts(self) -> Iterator[str]:
        for doc in self.documents():
            yield doc["text"]

    @property
    def n_documents(self) -> int:
        return sum(1 for _ in self.documents())

    @property
    def n_chars(self) -> int:
        return sum(d["chars"] for d in self.documents())

    # ---- preparation -------------------------------------------------------

    def build_tokenizer(self, vocab_size: int, verbose: bool = True) -> Tokenizer:
        if verbose:
            print(f"training tokenizer (target vocab {vocab_size}) ...", flush=True)
        tok = Tokenizer.train(self.texts(), vocab_size=vocab_size, verbose=verbose)
        tok.save(str(self.tokenizer_path))
        if verbose:
            print(f"  learned {len(tok.merges)} merges -> vocab {tok.vocab_size}", flush=True)
        return tok

    def load_tokenizer(self) -> Tokenizer:
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"no tokenizer at {self.tokenizer_path}; run `mb prepare` first"
            )
        return Tokenizer.load(str(self.tokenizer_path))

    def tokenize(self, tok: Tokenizer, verbose: bool = True) -> int:
        """Tokenize every document into tokens.bin. Returns the token count."""
        total = 0
        with open(self.tokens_path, "wb") as out:
            for i, text in enumerate(self.texts()):
                ids = tok.encode(text, bos=True, eos=True)
                np.asarray(ids, dtype=TOKEN_DTYPE).tofile(out)
                total += len(ids)
                if verbose and (i + 1) % 200 == 0:
                    print(f"  tokenized {i + 1} docs, {total:,} tokens", flush=True)
        self.write_meta(n_tokens=total, vocab_size=tok.vocab_size)
        if verbose:
            print(f"  {total:,} tokens written to {self.tokens_path}", flush=True)
        return total

    def prepare(self, vocab_size: int, verbose: bool = True) -> tuple[Tokenizer, int]:
        """Train a tokenizer on the corpus and tokenize it. The full pipeline."""
        if self.n_documents == 0:
            raise ValueError(f"corpus at {self.root} is empty; feed it something first")
        tok = self.build_tokenizer(vocab_size, verbose=verbose)
        n = self.tokenize(tok, verbose=verbose)
        return tok, n

    # ---- metadata ----------------------------------------------------------

    def write_meta(self, **kw) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = self.meta()
        meta.update(kw)
        meta["documents"] = self.n_documents
        meta["chars"] = self.n_chars
        meta["updated_at"] = time.time()
        with open(self.meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    def meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path) as fh:
                return json.load(fh)
        return {}

    @property
    def n_tokens(self) -> int:
        if not self.tokens_path.exists():
            return 0
        return self.tokens_path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize


class TokenStream:
    """Samples training batches from tokens.bin without loading it into RAM.

    The tail fraction is held out as a validation split so training loss can be
    checked against text the model has not been optimized on.
    """

    def __init__(self, path: str | Path, seq_len: int, val_fraction: float = 0.02,
                 seed: int = 1337) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no tokens at {path}; run `mb prepare` first")
        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        n = len(self.data)
        if n < seq_len + 2:
            raise ValueError(
                f"corpus has {n} tokens, too few for seq_len={seq_len}; "
                "feed it more text or lower --seq-len"
            )
        self.n_val = max(seq_len + 1, int(n * val_fraction))
        self.split_at = n - self.n_val
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def bounds(self, split: str) -> tuple[int, int]:
        return (0, self.split_at) if split == "train" else (self.split_at, len(self.data))

    def batch(self, batch_size: int, split: str = "train"):
        """One batch of (inputs, targets), each (batch_size, seq_len)."""
        import torch

        lo, hi = self.bounds(split)
        high = hi - self.seq_len - 1
        if high <= lo:
            lo, high = self.bounds("train")
            high -= self.seq_len + 1
        starts = self.rng.integers(lo, high, size=batch_size)
        x = np.stack([self.data[s:s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1:s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
