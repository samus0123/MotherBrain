import numpy as np
import pytest

from motherbrain.architecture import Architecture
from motherbrain.checkpoint import checkpoint_paths, default_checkpoint
from motherbrain.version import Version

TINY = Architecture(
    d_model=32, n_layers=1, n_heads=4, d_ff=64, vocab_size=259, context_length=64
)


@pytest.mark.parametrize(
    "given",
    ["dir/motherbrain-0.1.0", "dir/motherbrain-0.1.0.json", "dir/motherbrain-0.1.0.weights.h5"],
)
def test_paths_are_derived_from_one_stem(given):
    meta, weights = checkpoint_paths(given)
    assert meta.name == "motherbrain-0.1.0.json"
    assert weights.name == "motherbrain-0.1.0.weights.h5"


def test_a_version_stem_keeps_its_patch_number():
    """`.0` looks like a file extension; it must not be stripped."""
    meta, weights = checkpoint_paths(default_checkpoint(Version(1, 10, 0)))
    assert meta.stem.endswith("1.10.0")
    assert "1.10.0.weights.h5" in weights.name


@pytest.mark.slow
def test_save_then_load_restores_the_same_model(tmp_path):
    from motherbrain import checkpoint as ckpt
    from motherbrain.training import new_model

    model = new_model(TINY, seed=3)
    tokens = np.zeros((1, 8), dtype=np.int32)
    before = np.asarray(model(tokens))

    path = tmp_path / "motherbrain-0.1.0"
    ckpt.save(model, Version(0, 1, 0), TINY, path, steps_trained=7, tokens_seen=99)

    loaded = ckpt.load(path)
    assert loaded.version == Version(0, 1, 0)
    assert loaded.architecture == TINY
    assert loaded.steps_trained == 7 and loaded.tokens_seen == 99
    assert np.allclose(before, np.asarray(loaded.model(tokens)), atol=1e-6)


@pytest.mark.slow
def test_loading_a_missing_checkpoint(tmp_path):
    from motherbrain import checkpoint as ckpt

    with pytest.raises(FileNotFoundError):
        ckpt.load(tmp_path / "absent")
