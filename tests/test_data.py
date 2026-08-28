import numpy as np
import pytest
import torch

from motherbrain.data import (
    TokenDataset,
    dtype_for_vocab,
    load_splits,
    pack_documents,
    read_meta,
    write_meta,
)
from motherbrain.tokenizer import ByteTokenizer


def test_dtype_selection():
    assert dtype_for_vocab(256) == np.dtype(np.uint16)
    assert dtype_for_vocab(65536) == np.dtype(np.uint16)
    assert dtype_for_vocab(65537) == np.dtype(np.uint32)
    with pytest.raises(ValueError):
        dtype_for_vocab(2**33)


def test_pack_documents_writes_expected_token_count(tmp_path):
    tok = ByteTokenizer()
    docs = ["hello", "world!"]
    path = tmp_path / "train.bin"
    n = pack_documents(docs, tok, path, vocab_size=257, eot_id=256)
    # each document plus one end-of-text token
    assert n == len("hello") + 1 + len("world!") + 1
    assert TokenDataset(path).__len__() == n


def test_eot_separates_documents(tmp_path):
    tok = ByteTokenizer()
    path = tmp_path / "t.bin"
    pack_documents(["ab", "cd"], tok, path, vocab_size=257, eot_id=256)
    tokens = np.fromfile(path, dtype=np.uint16)
    assert list(tokens) == [97, 98, 256, 99, 100, 256]


def test_batch_shapes_and_target_shift(tmp_path):
    path = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    ds = TokenDataset(path)
    x, y = ds.sample_batch(batch_size=4, seq_len=16)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.int64
    # targets are the inputs shifted by exactly one position
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_sampling_is_reproducible_with_a_generator(tmp_path):
    path = tmp_path / "t.bin"
    np.arange(500, dtype=np.uint16).tofile(path)
    ds = TokenDataset(path)
    a, _ = ds.sample_batch(3, 8, generator=np.random.default_rng(0))
    b, _ = ds.sample_batch(3, 8, generator=np.random.default_rng(0))
    assert torch.equal(a, b)


def test_sequential_iteration_covers_the_split_in_order(tmp_path):
    path = tmp_path / "t.bin"
    np.arange(200, dtype=np.uint16).tofile(path)
    ds = TokenDataset(path)
    batches = list(ds.iter_sequential(batch_size=2, seq_len=10))
    assert batches
    assert torch.equal(batches[0][0][0], torch.arange(10))
    assert torch.equal(batches[0][1][0], torch.arange(1, 11))


def test_too_short_dataset_raises(tmp_path):
    path = tmp_path / "t.bin"
    np.arange(4, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        TokenDataset(path).sample_batch(1, 32)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenDataset(tmp_path / "nope.bin")
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path)


def test_meta_roundtrip_and_load_splits(tmp_path):
    tok = ByteTokenizer()
    n_train = pack_documents(["abc" * 50], tok, tmp_path / "train.bin", 257, eot_id=256)
    n_val = pack_documents(["xyz" * 50], tok, tmp_path / "val.bin", 257, eot_id=256)
    write_meta(tmp_path, 257, {"train": n_train, "val": n_val})
    datasets, meta = load_splits(tmp_path)
    assert set(datasets) == {"train", "val"}
    assert meta["vocab_size"] == 257
    assert len(datasets["train"]) == n_train
