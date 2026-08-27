"""Checks on the network itself. These build real Keras models, so they are
the slow tests in the suite -- kept to the smallest architecture that still
exercises every code path."""

import numpy as np
import pytest

from motherbrain import SEED
from motherbrain.architecture import Architecture
from motherbrain.growth import GrowthPolicy

pytestmark = pytest.mark.slow

TINY = Architecture(
    d_model=32,
    n_layers=2,
    n_heads=4,
    d_ff=64,
    vocab_size=259,
    context_length=64,
)


@pytest.fixture(scope="module")
def tiny_model():
    from motherbrain.model import build_model

    return build_model(TINY)


def test_weight_count_matches_the_formula(tiny_model):
    assert tiny_model.count_params() == TINY.parameter_count


@pytest.mark.parametrize(
    "arch",
    [
        TINY,
        Architecture.from_dict({**TINY.to_dict(), "tie_embeddings": False}),
        Architecture.from_dict({**TINY.to_dict(), "gated_mlp": False}),
    ],
)
def test_formula_tracks_architecture_options(arch):
    from motherbrain.model import build_model

    assert build_model(arch).count_params() == arch.parameter_count


def test_grown_models_really_do_have_more_weights():
    """The end-to-end version of the promise: growth is real in Keras too."""
    from motherbrain.model import build_model

    policy = GrowthPolicy()
    arch = TINY
    previous = build_model(arch).count_params()
    for kind in ("patch", "minor", "major"):
        arch = policy.grow(arch, kind)
        current = build_model(arch).count_params()
        assert current > previous
        assert current == arch.parameter_count
        previous = current


def test_output_shape(tiny_model):
    logits = np.asarray(tiny_model(np.zeros((2, 9), dtype=np.int32)))
    assert logits.shape == (2, 9, TINY.vocab_size)


def test_attention_cannot_see_the_future(tiny_model):
    """Editing a later token must not change any earlier position's logits."""
    tokens = np.random.default_rng(0).integers(0, 259, size=(1, 16)).astype(np.int32)
    before = np.asarray(tiny_model(tokens))

    edited = tokens.copy()
    edited[0, -1] = (edited[0, -1] + 1) % 259
    after = np.asarray(tiny_model(edited))

    assert np.allclose(before[0, :-1], after[0, :-1], atol=1e-6)
    assert not np.allclose(before[0, -1], after[0, -1])


def test_a_prefix_gives_the_same_logits_as_the_full_sequence(tiny_model):
    """Generation feeds a growing prefix; it must agree with a batch pass."""
    tokens = np.random.default_rng(1).integers(0, 259, size=(1, 16)).astype(np.int32)
    full = np.asarray(tiny_model(tokens))
    prefix = np.asarray(tiny_model(tokens[:, :8]))
    assert np.allclose(full[0, :8], prefix[0], atol=1e-6)


def test_seed_release_builds():
    from motherbrain.model import build_model

    model = build_model(SEED.architecture)
    assert model.count_params() == SEED.architecture.parameter_count
