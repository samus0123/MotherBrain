import numpy as np
import pytest
import torch

from motherbrain.data import BatchLoader, ShardIndex, TokenDataset


def test_index_roundtrip(tmp_path):
    index = ShardIndex(
        vocab_size=99, dtype="uint16", splits={"train": [{"path": "a.bin", "tokens": 10}]}
    )
    index.save(tmp_path)
    loaded = ShardIndex.load(tmp_path)
    assert loaded.vocab_size == 99 and loaded.tokens_in("train") == 10


def test_missing_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta.json"):
        ShardIndex.load(tmp_path)


def test_window_count_accounts_for_shifted_target(corpus):
    ds = TokenDataset(corpus, "train", seq_len=100)
    # Two 500-token shards: each yields floor((500 - 1) / 100) = 4 windows.
    assert ds.num_windows == 8


def test_windows_are_contiguous_and_shifted(corpus):
    ds = TokenDataset(corpus, "train", seq_len=10)
    x, y = ds[0]
    assert x.shape == (10,) and y.shape == (10,)
    # y is x shifted by one: the target at position i is the input at i+1.
    assert np.array_equal(x[1:], y[:-1])
    assert x.dtype == np.int64  # ready for the embedding lookup


def test_windows_do_not_straddle_shards(corpus):
    ds = TokenDataset(corpus, "train", seq_len=100)
    raw0 = np.fromfile(corpus / "train_00000.bin", dtype=np.uint16)
    raw1 = np.fromfile(corpus / "train_00001.bin", dtype=np.uint16)
    # Last window of shard 0 then first window of shard 1.
    assert ds.locate(3)[0] == 0 and ds.locate(4) == (1, 0)
    assert np.array_equal(ds[3][0], raw0[300:400].astype(np.int64))
    assert np.array_equal(ds[4][0], raw1[0:100].astype(np.int64))


def test_out_of_range_window_raises(corpus):
    ds = TokenDataset(corpus, "train", seq_len=100)
    with pytest.raises(IndexError):
        ds.locate(ds.num_windows)


def test_seq_len_longer_than_shards_raises(corpus):
    with pytest.raises(ValueError, match="no complete windows"):
        TokenDataset(corpus, "train", seq_len=5000)


def test_unknown_split_raises(corpus):
    with pytest.raises(KeyError):
        TokenDataset(corpus, "nope", seq_len=10)


def test_uint32_shards_are_supported(tmp_path, write_corpus_fn):
    data_dir = write_corpus_fn(tmp_path / "wide", vocab=70000, dtype="uint32")
    ds = TokenDataset(data_dir, "train", seq_len=10)
    x, _ = ds[0]
    assert x.max() < 70000


def test_batch_shapes_and_dtype(corpus):
    loader = BatchLoader(TokenDataset(corpus, "train", 10), batch_size=4)
    x, y = next(loader)
    assert x.shape == (4, 10) and y.shape == (4, 10)
    assert x.dtype == torch.long


def test_same_seed_gives_same_stream(corpus):
    ds = TokenDataset(corpus, "train", 10)
    a = next(BatchLoader(ds, 4, seed=7))[0]
    b = next(BatchLoader(ds, 4, seed=7))[0]
    assert torch.equal(a, b)


def test_different_seed_gives_different_stream(corpus):
    ds = TokenDataset(corpus, "train", 10)
    a = next(BatchLoader(ds, 4, seed=1))[0]
    b = next(BatchLoader(ds, 4, seed=2))[0]
    assert not torch.equal(a, b)


def test_shuffle_off_walks_windows_in_order(corpus):
    ds = TokenDataset(corpus, "train", 10)
    loader = BatchLoader(ds, 4, shuffle=False)
    x, _ = next(loader)
    for i in range(4):
        assert np.array_equal(x[i].numpy(), ds[i][0])


def test_ranks_see_disjoint_windows(corpus):
    """Every window in an epoch goes to exactly one rank."""
    ds = TokenDataset(corpus, "train", 10)
    loaders = [BatchLoader(ds, 2, seed=3, rank=r, world_size=4) for r in range(4)]
    seen = [set(ld._epoch_order(0).tolist()) for ld in loaders]
    assert all(len(s) == loaders[0].windows_per_rank for s in seen)
    union = set().union(*seen)
    assert len(union) == sum(len(s) for s in seen)  # pairwise disjoint


def test_epochs_reshuffle(corpus):
    loader = BatchLoader(TokenDataset(corpus, "train", 10), 2, seed=5)
    assert not np.array_equal(loader._epoch_order(0), loader._epoch_order(1))


def test_loader_wraps_around_forever(corpus):
    ds = TokenDataset(corpus, "train", 10)
    loader = BatchLoader(ds, 4)
    # Draw well past one epoch; the stream must not end.
    for _ in range(loader.batches_per_epoch * 3 + 1):
        x, _ = next(loader)
        assert x.shape == (4, 10)


def test_state_dict_resumes_exact_position(corpus):
    ds = TokenDataset(corpus, "train", 10)
    a = BatchLoader(ds, 2, seed=11)
    for _ in range(3):
        next(a)
    state = a.state_dict()
    expected = [next(a)[0] for _ in range(2)]

    b = BatchLoader(ds, 2, seed=11)
    b.load_state_dict(state)
    got = [next(b)[0] for _ in range(2)]
    assert all(torch.equal(e, g) for e, g in zip(expected, got, strict=True))


def test_resume_rejects_changed_run_shape(corpus):
    ds = TokenDataset(corpus, "train", 10)
    state = BatchLoader(ds, 2, seed=11).state_dict()
    with pytest.raises(ValueError, match="batch_size"):
        BatchLoader(ds, 4, seed=11).load_state_dict(state)


def test_invalid_rank_raises(corpus):
    ds = TokenDataset(corpus, "train", 10)
    with pytest.raises(ValueError, match="rank"):
        BatchLoader(ds, 2, rank=4, world_size=4)


def test_world_size_larger_than_dataset_raises(corpus):
    ds = TokenDataset(corpus, "train", 100)  # only 8 windows
    with pytest.raises(ValueError, match="fewer than world_size"):
        BatchLoader(ds, 1, rank=0, world_size=99)
