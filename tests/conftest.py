"""Shared fixtures: a tiny model config and a synthetic tokenized corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motherbrain.config import Config, ModelConfig
from motherbrain.data import ShardIndex

VOCAB = 64


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        max_seq_len=32,
        dropout=0.0,
    )


def write_corpus(
    data_dir, tokens_per_shard=(500, 500), val_tokens=200, vocab=VOCAB, dtype="uint16", seed=0
):
    """Write synthetic token shards plus meta.json into ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[dtype]
    splits: dict[str, list[dict]] = {"train": []}
    for i, n in enumerate(tokens_per_shard):
        name = f"train_{i:05d}.bin"
        rng.integers(0, vocab, size=n, dtype=np.int64).astype(np_dtype).tofile(data_dir / name)
        splits["train"].append({"path": name, "tokens": int(n)})
    if val_tokens:
        rng.integers(0, vocab, size=val_tokens, dtype=np.int64).astype(np_dtype).tofile(
            data_dir / "val_00000.bin"
        )
        splits["val"] = [{"path": "val_00000.bin", "tokens": int(val_tokens)}]
    ShardIndex(vocab_size=vocab, dtype=dtype, splits=splits).save(data_dir)
    return data_dir


@pytest.fixture
def write_corpus_fn():
    """The corpus builder itself, for tests that need a custom layout."""
    return write_corpus


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "data")


@pytest.fixture
def train_cfg(tmp_path, corpus, model_cfg) -> Config:
    cfg = Config()
    cfg.model = model_cfg
    cfg.data.data_dir = str(corpus)
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.train.batch_size = 2
    cfg.train.grad_accum_steps = 2
    cfg.train.max_steps = 8
    cfg.train.device = "cpu"
    cfg.train.dtype = "float32"
    cfg.train.compile = False
    cfg.train.eval_interval = 4
    cfg.train.eval_steps = 2
    cfg.train.log_interval = 100
    cfg.train.ckpt_interval = 4
    cfg.optim.warmup_steps = 2
    cfg.optim.lr = 1e-3
    return cfg
