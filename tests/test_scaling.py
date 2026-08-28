import pytest

from motherbrain.config import ModelConfig
from motherbrain.scaling import (
    PRESETS,
    chinchilla_tokens,
    count_active_parameters,
    count_parameters,
    flops_per_token,
    get_preset,
    human,
    kv_cache_bytes,
    ladder_table,
    memory_estimate,
    verify_counts,
)

VERIFY_CASES = [
    dict(),
    dict(tie_embeddings=False),
    dict(n_heads=8, n_kv_heads=1),
    dict(n_experts=8, n_experts_per_tok=2),
    dict(n_experts=8, n_experts_per_tok=2, n_shared_experts=1),
    dict(n_experts=8, n_experts_per_tok=2, moe_first_dense_layers=2, moe_layer_freq=2),
    dict(n_experts=4, n_experts_per_tok=1, moe_ffn_hidden=64, tie_embeddings=False),
]


@pytest.mark.parametrize("kw", VERIFY_CASES)
def test_analytic_counts_match_real_modules(kw):
    """The formulas must agree exactly with an instantiated model."""
    base = dict(vocab_size=311, dim=64, n_layers=5, n_heads=4, n_kv_heads=2, max_seq_len=32)
    base.update(kw)
    cfg = ModelConfig(**base)
    total, active = verify_counts(cfg)
    assert total == count_parameters(cfg).total
    assert active == count_active_parameters(cfg)


def test_dense_models_are_fully_active():
    cfg = ModelConfig(vocab_size=311, dim=64, n_layers=4, n_heads=4)
    assert count_active_parameters(cfg) == count_parameters(cfg).total


def test_experts_grow_total_but_not_active():
    """8x the experts means ~8x the parameters but the same experts run per token.

    Active parameters are not *exactly* equal: the router is a `dim x n_experts`
    matrix, so it genuinely grows with the expert count. Everything else the
    token touches is unchanged, so the difference is exactly the router.
    """
    base = dict(vocab_size=311, dim=64, n_layers=4, n_heads=4, n_experts_per_tok=2)
    small = ModelConfig(n_experts=8, **base)
    big = ModelConfig(n_experts=64, **base)
    assert count_parameters(big).total > 4 * count_parameters(small).total

    router_growth = count_parameters(big).router - count_parameters(small).router
    assert count_active_parameters(big) - count_active_parameters(small) == router_growth
    assert router_growth < 0.01 * count_parameters(small).total


def test_flops_scale_with_active_not_total():
    """16x the total parameters must not meaningfully change FLOPs per token."""
    base = dict(vocab_size=311, dim=64, n_layers=4, n_heads=4, n_experts_per_tok=2, max_seq_len=64)
    small = ModelConfig(n_experts=8, **base)
    big = ModelConfig(n_experts=128, **base)
    total_ratio = count_parameters(big).total / count_parameters(small).total
    flop_ratio = flops_per_token(big) / flops_per_token(small)
    assert total_ratio > 10
    assert flop_ratio < 1.1


def test_breakdown_sums_to_total():
    cfg = get_preset("moe-small")
    b = count_parameters(cfg)
    assert b.total == (
        b.embedding + b.attention + b.dense_ffn + b.moe_experts
        + b.moe_shared + b.router + b.norms + b.lm_head
    )


def test_memory_estimate_is_consistent():
    cfg = get_preset("small")
    mem = memory_estimate(cfg, dtype="bf16")
    total = count_parameters(cfg).total
    assert mem["weights_bytes"] == total * 2
    assert mem["train_total_bytes"] > mem["inference_bytes"]


def test_kv_cache_scales_with_kv_heads():
    wide = ModelConfig(dim=512, n_heads=8, n_kv_heads=8)
    grouped = ModelConfig(dim=512, n_heads=8, n_kv_heads=2)
    assert kv_cache_bytes(grouped, 1, 1024) * 4 == kv_cache_bytes(wide, 1, 1024)


def test_chinchilla_ratio():
    assert chinchilla_tokens(1_000_000) == 20_000_000


def test_ladder_is_monotonically_larger():
    totals = [count_parameters(cfg).total for cfg in PRESETS.values()]
    dense = totals[:7]  # the dense tier, nano -> xl
    assert dense == sorted(dense)
    assert count_parameters(PRESETS["motherbrain"]).total > 100e12


def test_every_preset_is_valid_and_describable():
    for name in PRESETS:
        cfg = get_preset(name)
        assert count_active_parameters(cfg) <= count_parameters(cfg).total
    assert "motherbrain" in ladder_table()


def test_get_preset_returns_a_copy():
    a = get_preset("nano")
    a.dim = 4096
    assert get_preset("nano").dim != 4096


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("does-not-exist")


def test_human_formatting():
    assert human(1_500_000_000_000) == "1.50T"
    assert human(2_500_000_000) == "2.50B"
    assert human(999) == "999"
