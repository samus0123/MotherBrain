import pytest

from motherbrain.architecture import Architecture


def make(**overrides) -> Architecture:
    base = dict(
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=704,
        vocab_size=259,
        context_length=1024,
    )
    return Architecture(**{**base, **overrides})


def test_parameter_count_matches_hand_calculation():
    arch = make()
    per_layer = 4 * 256 * 256 + 3 * 256 * 704 + 2 * 256
    expected = 259 * 256 + 6 * per_layer + 256
    assert arch.parameter_count == expected


def test_untied_embeddings_add_an_output_head():
    assert (
        make(tie_embeddings=False).parameter_count
        == make().parameter_count + 259 * 256
    )


def test_learned_positions_add_a_position_table():
    assert (
        make(learned_positions=True).parameter_count
        == make().parameter_count + 1024 * 256
    )


@pytest.mark.parametrize("field", ["d_model", "n_layers", "n_heads", "d_ff"])
def test_dimensions_must_be_positive(field):
    with pytest.raises(ValueError):
        make(**{field: 0})


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError):
        make(d_model=256, n_heads=7)


def test_dict_roundtrip():
    arch = make()
    assert Architecture.from_dict(arch.to_dict()) == arch


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        Architecture.from_dict({**make().to_dict(), "d_qkv": 64})


def test_growing_any_dimension_adds_parameters():
    base = make().parameter_count
    for field, value in (
        ("d_model", 320),
        ("n_layers", 7),
        ("d_ff", 768),
        ("vocab_size", 300),
    ):
        assert make(**{field: value}).parameter_count > base
