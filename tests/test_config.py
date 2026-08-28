import pytest

from motherbrain.config import Config, ModelConfig, apply_overrides, load_config


def test_derived_defaults():
    cfg = ModelConfig(d_model=768, n_head=12)
    assert cfg.n_kv_head == 12  # defaults to MHA
    assert cfg.head_dim == 64
    # 8/3 * 768 = 2048, already a multiple of 256
    assert cfg.d_ff == 2048


def test_d_ff_rounds_up_to_multiple():
    cfg = ModelConfig(d_model=1024, n_head=16, ffn_multiple_of=256)
    assert cfg.d_ff % 256 == 0
    assert cfg.d_ff >= int(8 * 1024 / 3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"d_model": 100, "n_head": 12}, "divisible by n_head"),
        ({"d_model": 64, "n_head": 8, "n_kv_head": 3}, "divisible by n_kv_head"),
        ({"n_layer": 0}, "must be positive"),
        ({"dropout": 1.5}, "dropout"),
    ],
)
def test_invalid_model_configs_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**kwargs)


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_dict({"model": {"nonexistent": 1}})


def test_overrides_coerce_types():
    cfg = Config()
    apply_overrides(
        cfg,
        [
            "train.max_steps=1e4",
            "train.compile=true",
            "optim.lr=3e-4",
            "optim.decay_steps=none",
            "train.out_dir=runs/x",
        ],
    )
    assert cfg.train.max_steps == 10000 and isinstance(cfg.train.max_steps, int)
    assert cfg.train.compile is True
    assert cfg.optim.lr == pytest.approx(3e-4)
    assert cfg.optim.decay_steps is None
    assert cfg.train.out_dir == "runs/x"


def test_override_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config"):
        apply_overrides(Config(), ["train.nope=1"])


def test_override_is_validated():
    with pytest.raises(ValueError):
        apply_overrides(Config(), ["train.batch_size=0"])


def test_yaml_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model:\n  n_layer: 4\n  n_head: 8\n  d_model: 64\noptim:\n  lr: 1.0e-3\n"
    )
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\nmodel:\n  n_layer: 8\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.model.n_layer == 8  # child wins
    assert cfg.model.d_model == 64  # inherited
    assert cfg.optim.lr == pytest.approx(1e-3)


def test_yaml_inherit_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("inherit: b.yaml\n")
    (tmp_path / "b.yaml").write_text("inherit: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_roundtrip_through_dict():
    cfg = Config()
    cfg.model.n_layer = 7
    assert Config.from_dict(cfg.to_dict()).model.n_layer == 7


def test_flat_dict_is_flat():
    flat = Config().flat_dict()
    assert flat["model.n_layer"] == 12
    assert all("." in k or not isinstance(flat[k], dict) for k in flat)


def test_shipped_configs_load():
    from pathlib import Path

    for path in sorted((Path(__file__).parents[1] / "configs").glob("*.yaml")):
        if path.name == "base.yaml":
            continue  # base is a fragment, meant to be inherited
        load_config(path).validate()


def test_derived_fields_follow_later_edits():
    """Overriding n_head/d_model must re-derive the fields that depend on them."""
    cfg = Config()
    apply_overrides(cfg, ["model.n_head=2", "model.d_model=32"])
    assert cfg.model.n_kv_head == 2
    assert cfg.model.d_ff == 256


def test_explicit_values_are_not_overwritten_by_derivation():
    cfg = Config()
    apply_overrides(cfg, ["model.n_kv_head=4", "model.n_head=8"])
    assert cfg.model.n_kv_head == 4  # explicit choice survives a later n_head edit

    pinned = ModelConfig(n_head=8, d_model=64, n_kv_head=2, d_ff=999)
    pinned.validate()
    assert (pinned.n_kv_head, pinned.d_ff) == (2, 999)


def test_setting_a_derived_field_to_none_restores_auto():
    cfg = ModelConfig(n_head=8, d_model=64, n_kv_head=2)
    cfg.n_kv_head = None
    cfg.validate()
    assert cfg.n_kv_head == 8


def test_derivation_bookkeeping_stays_out_of_serialization():
    assert not [k for k in Config().to_dict()["model"] if k.startswith("_")]
