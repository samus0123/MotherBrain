"""The core promise: every version bump makes the model bigger."""

import pytest

from motherbrain import SEED
from motherbrain.growth import GrowthPolicy, _grow_to_grid
from motherbrain.version import Version

BUMPS = ("major", "minor", "patch")


@pytest.fixture
def policy() -> GrowthPolicy:
    return GrowthPolicy()


@pytest.mark.parametrize("kind", BUMPS)
def test_every_bump_kind_increases_parameters(policy, kind):
    arch = SEED.architecture
    assert policy.grow(arch, kind).parameter_count > arch.parameter_count


@pytest.mark.parametrize("kind", BUMPS)
def test_no_dimension_ever_shrinks(policy, kind):
    arch = SEED.architecture
    grown = policy.grow(arch, kind)
    for field in ("d_model", "n_layers", "n_heads", "d_ff", "context_length"):
        assert getattr(grown, field) >= getattr(arch, field), field


@pytest.mark.parametrize("kind", BUMPS)
def test_repeated_bumps_keep_growing(policy, kind):
    arch = SEED.architecture
    for _ in range(12):
        grown = policy.grow(arch, kind)
        assert grown.parameter_count > arch.parameter_count
        arch = grown


def test_growth_holds_across_a_mixed_release_sequence(policy):
    arch = SEED.architecture
    version = SEED.version
    sequence = ["patch", "patch", "minor", "patch", "minor", "major", "patch"]
    for kind in sequence:
        grown = policy.grow(arch, kind)
        bumped = version.bump(kind)
        assert bumped > version
        assert grown.parameter_count > arch.parameter_count
        arch, version = grown, bumped


def test_bigger_bumps_grow_more(policy):
    arch = SEED.architecture
    sizes = {k: policy.grow(arch, k).parameter_count for k in BUMPS}
    assert sizes["patch"] < sizes["minor"] < sizes["major"]


def test_major_bump_extends_the_context_window(policy):
    arch = SEED.architecture
    assert policy.grow(arch, "major").context_length > arch.context_length


@pytest.mark.parametrize("kind", BUMPS)
def test_grown_widths_stay_divisible_by_head_count(policy, kind):
    arch = SEED.architecture
    for _ in range(8):
        arch = policy.grow(arch, kind)  # Architecture validates divisibility
        assert arch.d_model % arch.n_heads == 0


def test_grow_rejects_unknown_bump_kinds(policy):
    with pytest.raises(ValueError):
        policy.grow(SEED.architecture, "sideways")


def test_grid_rounding_always_moves_up():
    assert _grow_to_grid(256, 1.10) == 320
    assert _grow_to_grid(704, 1.06) == 768
    assert _grow_to_grid(64, 1.001) == 128  # rounds up even for a tiny factor


def test_growth_factors_below_one_are_refused():
    with pytest.raises(ValueError):
        _grow_to_grid(256, 0.5)


def test_a_no_op_step_is_reported_rather_than_returned():
    """A policy that does not grow must fail loudly, not shrink silently."""
    from motherbrain.growth import GrowthStep

    flat = GrowthPolicy(patch=GrowthStep())
    with pytest.raises(ValueError, match="did not grow"):
        flat.grow(SEED.architecture, "patch")
