import pytest
import torch
import torch.nn as nn

from motherbrain.checkpoint import (
    config_from_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    unwrap_model,
)
from motherbrain.config import Config, OptimConfig
from motherbrain.model import Transformer
from motherbrain.optim import build_optimizer


def _fixture(model_cfg):
    cfg = Config()
    cfg.model = model_cfg
    model = Transformer(model_cfg)
    return cfg, model, build_optimizer(model, OptimConfig(), "cpu")


def test_roundtrip_restores_weights_exactly(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    path = save_checkpoint(tmp_path / "c.pt", model, opt, 42, cfg)

    restored = Transformer(model_cfg)
    assert not torch.allclose(restored.tok_emb.weight, model.tok_emb.weight)
    ckpt = load_checkpoint(path, model=restored)
    assert ckpt["step"] == 42
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_optimizer_state_survives(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    x = torch.randint(0, model_cfg.vocab_size, (2, 8))
    model(x, targets=x)[1].backward()
    opt.step()

    path = save_checkpoint(tmp_path / "c.pt", model, opt, 1, cfg)
    fresh_model = Transformer(model_cfg)
    fresh_opt = build_optimizer(fresh_model, OptimConfig(), "cpu")
    load_checkpoint(path, model=fresh_model, optimizer=fresh_opt)
    # Adam's step counter and moments come back.
    state = next(iter(fresh_opt.state.values()))
    assert state["exp_avg"].abs().sum() > 0


def test_loader_state_and_best_loss_roundtrip(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    path = save_checkpoint(
        tmp_path / "c.pt",
        model,
        opt,
        5,
        cfg,
        best_val_loss=1.25,
        loader_state={"samples_seen": 99},
    )
    ckpt = load_checkpoint(path)
    assert ckpt["best_val_loss"] == 1.25
    assert ckpt["loader_state"]["samples_seen"] == 99


def test_config_is_recoverable(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    path = save_checkpoint(tmp_path / "c.pt", model, opt, 1, cfg)
    assert config_from_checkpoint(path).model.n_layer == model_cfg.n_layer


def test_ddp_and_compile_prefixes_are_stripped(tmp_path, model_cfg):
    """A checkpoint from a wrapped model must load into a bare model."""
    cfg, model, _ = _fixture(model_cfg)

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    path = save_checkpoint(tmp_path / "c.pt", Wrapper(model), None, 1, cfg)
    assert all(not k.startswith("module.") for k in torch.load(path, weights_only=False)["model"])
    load_checkpoint(path, model=Transformer(model_cfg))  # strict load must succeed


def test_unwrap_model_reaches_the_core(model_cfg):
    model = Transformer(model_cfg)

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    assert unwrap_model(Wrapper(Wrapper(model))) is model
    assert unwrap_model(model) is model


def test_save_is_atomic(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    save_checkpoint(tmp_path / "c.pt", model, opt, 1, cfg)
    # No .tmp scratch file is left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_latest_checkpoint_picks_highest_step(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    assert latest_checkpoint(tmp_path) is None
    for step in (5, 100, 20):
        save_checkpoint(tmp_path / f"ckpt_{step:07d}.pt", model, opt, step, cfg)
    save_checkpoint(tmp_path / "best.pt", model, opt, 1, cfg)  # must be ignored
    assert latest_checkpoint(tmp_path).name == "ckpt_0000100.pt"


def test_rotation_keeps_only_the_newest(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    for step in range(1, 6):
        save_checkpoint(tmp_path / f"ckpt_{step:07d}.pt", model, opt, step, cfg)
    save_checkpoint(tmp_path / "best.pt", model, opt, 1, cfg)
    rotate_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best.pt", "ckpt_0000004.pt", "ckpt_0000005.pt"]


def test_rotation_disabled_keeps_everything(tmp_path, model_cfg):
    cfg, model, opt = _fixture(model_cfg)
    for step in range(1, 4):
        save_checkpoint(tmp_path / f"ckpt_{step:07d}.pt", model, opt, step, cfg)
    rotate_checkpoints(tmp_path, keep_last_n=0)
    assert len(list(tmp_path.iterdir())) == 3
