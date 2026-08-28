import pytest
import torch

from motherbrain.config import OptimConfig
from motherbrain.model import Transformer
from motherbrain.optim import build_optimizer, get_lr, param_groups


def test_norms_and_biases_are_not_decayed(model_cfg):
    model = Transformer(model_cfg)
    decay, no_decay = param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_tied_weights_are_counted_once(model_cfg):
    model = Transformer(model_cfg)  # tie_embeddings defaults to True
    grouped = sum(len(g["params"]) for g in param_groups(model, 0.1))
    unique = len({id(p) for p in model.parameters()})
    assert grouped == unique


def test_every_parameter_lands_in_a_group(model_cfg):
    model = Transformer(model_cfg)
    grouped = {id(p) for g in param_groups(model, 0.1) for p in g["params"]}
    assert grouped == {id(p) for p in model.parameters()}


def test_build_optimizer_is_adamw(model_cfg):
    opt = build_optimizer(Transformer(model_cfg), OptimConfig(lr=1e-3), "cpu")
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    # The fused kernel is CUDA-only and must not be requested on CPU.
    assert not opt.param_groups[0].get("fused", False)


def test_warmup_rises_then_hits_peak():
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1)
    warm = [get_lr(s, cfg, 100) for s in range(10)]
    assert warm[0] > 0  # step 0 must still make progress
    assert warm == sorted(warm)
    assert warm[-1] < cfg.lr
    assert get_lr(10, cfg, 100) == pytest.approx(cfg.lr, rel=1e-6)


def test_cosine_decays_to_min_lr():
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1, schedule="cosine")
    assert get_lr(100, cfg, 100) == pytest.approx(1e-4)
    assert get_lr(500, cfg, 100) == pytest.approx(1e-4)  # clamped past the end
    mid = get_lr(55, cfg, 100)
    assert 1e-4 < mid < 1e-3


def test_cosine_is_monotonically_decreasing_after_warmup():
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, schedule="cosine")
    lrs = [get_lr(s, cfg, 100) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))


def test_linear_schedule_is_straight():
    cfg = OptimConfig(lr=1.0, warmup_steps=0, min_lr_ratio=0.0, schedule="linear")
    assert get_lr(50, cfg, 100) == pytest.approx(0.5, abs=0.02)


def test_constant_schedule_holds():
    cfg = OptimConfig(lr=1e-3, warmup_steps=5, schedule="constant")
    assert get_lr(50, cfg, 100) == pytest.approx(1e-3)
    assert get_lr(99, cfg, 100) == pytest.approx(1e-3)


def test_decay_steps_overrides_max_steps():
    """decay_steps lets a run reach min_lr before the last step (or after)."""
    cfg = OptimConfig(lr=1e-3, warmup_steps=0, min_lr_ratio=0.1, decay_steps=50)
    assert get_lr(50, cfg, 100) == pytest.approx(1e-4)
    assert get_lr(75, cfg, 100) == pytest.approx(1e-4)


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError, match="unknown schedule"):
        OptimConfig(schedule="magic").validate()


def test_optimizer_step_changes_weights(model_cfg):
    model = Transformer(model_cfg)
    opt = build_optimizer(model, OptimConfig(lr=1e-2), "cpu")
    before = model.blocks[0].mlp.down_proj.weight.detach().clone()
    x = torch.randint(0, model_cfg.vocab_size, (2, 8))
    _, loss = model(x, targets=x)
    loss.backward()
    opt.step()
    assert not torch.allclose(before, model.blocks[0].mlp.down_proj.weight)
