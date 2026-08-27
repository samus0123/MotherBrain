import numpy as np
import pytest

from motherbrain.data import load_corpus, make_windows


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("the quick brown fox " * 40, encoding="utf-8")
    (tmp_path / "b.md").write_text("jumps over the lazy dog " * 40, encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_load_reads_only_text_files(corpus):
    tokens = load_corpus(corpus)
    assert tokens.dtype == np.int32
    assert len(tokens) > 1000


def test_targets_are_inputs_shifted_by_one():
    tokens = np.arange(100, dtype=np.int32)
    inputs, targets = make_windows(tokens, seq_len=10)
    assert inputs.shape == targets.shape
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])
    assert targets[0, -1] == inputs[1, 0]


def test_windows_never_read_past_the_corpus():
    tokens = np.arange(25, dtype=np.int32)
    inputs, targets = make_windows(tokens, seq_len=10)
    assert inputs.shape == (2, 10)  # 25 tokens -> 2 full windows, remainder dropped
    assert int(targets.max()) <= int(tokens.max())


def test_too_short_a_corpus_is_reported(tmp_path):
    with pytest.raises(ValueError, match="need more than"):
        make_windows(np.arange(5, dtype=np.int32), seq_len=64)


def test_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_directory_without_text_files(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="no text files"):
        load_corpus(tmp_path)
