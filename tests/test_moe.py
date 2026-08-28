import torch

from motherbrain.config import ModelConfig
from motherbrain.model import MoEFeedForward, MotherBrain


def moe_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=97, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=32,
        n_experts=8, n_experts_per_tok=2,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_moe_layer_placement():
    cfg = moe_cfg(n_layers=6, moe_first_dense_layers=2, moe_layer_freq=2)
    assert [cfg.is_moe_layer(i) for i in range(6)] == [False, False, True, False, True, False]


def test_dense_when_single_expert():
    cfg = moe_cfg(n_experts=1, n_experts_per_tok=1)
    assert not any(cfg.is_moe_layer(i) for i in range(cfg.n_layers))
    m = MotherBrain(cfg)
    assert m(torch.randint(0, 97, (1, 8)), targets=torch.randint(0, 97, (1, 8))).aux_loss is None


def test_router_losses_are_reported():
    m = MotherBrain(moe_cfg())
    x = torch.randint(0, 97, (2, 8))
    out = m(x, targets=x)
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)
    assert out.router_z_loss is not None and torch.isfinite(out.router_z_loss)


def test_balanced_router_hits_the_aux_loss_floor():
    """Perfectly uniform routing gives an aux loss of ~1.0; skewed routing exceeds it."""
    cfg = moe_cfg()
    moe = MoEFeedForward(cfg)
    n_tokens, n_experts = 512, cfg.n_experts

    uniform = torch.full((n_tokens, n_experts), 1.0 / n_experts)
    idx = torch.arange(n_tokens)[:, None] % n_experts
    idx = torch.cat([idx, (idx + 1) % n_experts], dim=1)
    balanced = moe._load_balancing_loss(uniform, idx)
    assert abs(balanced.item() - 1.0) < 0.05

    skewed_probs = torch.zeros(n_tokens, n_experts)
    skewed_probs[:, 0] = 0.9
    skewed_probs[:, 1:] = 0.1 / (n_experts - 1)
    skewed_idx = torch.zeros(n_tokens, 2, dtype=torch.long)
    skewed_idx[:, 1] = 1
    assert moe._load_balancing_loss(skewed_probs, skewed_idx).item() > balanced.item()


def test_top_k_routing_weights_sum_to_one():
    cfg = moe_cfg(n_experts_per_tok=3)
    moe = MoEFeedForward(cfg).eval()
    x = torch.randn(2, 5, cfg.dim)
    logits = moe.gate(x.view(-1, cfg.dim)).float()
    probs = torch.softmax(logits, dim=-1)
    top, _ = torch.topk(probs, cfg.n_experts_per_tok, dim=-1)
    normalised = top / top.sum(dim=-1, keepdim=True)
    assert torch.allclose(normalised.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_only_selected_experts_affect_a_token():
    """Zeroing an unselected expert must not change that token's output."""
    torch.manual_seed(0)
    cfg = moe_cfg(n_experts=4, n_experts_per_tok=1, n_shared_experts=0)
    moe = MoEFeedForward(cfg).eval()
    x = torch.randn(1, 1, cfg.dim)
    with torch.no_grad():
        baseline = moe(x)
        chosen = int(torch.topk(torch.softmax(moe.gate(x.view(-1, cfg.dim)).float(), -1), 1, -1).indices)
        other = (chosen + 1) % cfg.n_experts
        moe.experts.w2.data[other].zero_()
        assert torch.allclose(baseline, moe(x), atol=1e-6)
        moe.experts.w2.data[chosen].zero_()
        assert not torch.allclose(baseline, moe(x), atol=1e-6)


def test_shared_expert_always_contributes():
    torch.manual_seed(0)
    cfg = moe_cfg(n_shared_experts=1)
    moe = MoEFeedForward(cfg).eval()
    x = torch.randn(2, 4, cfg.dim)
    with torch.no_grad():
        before = moe(x)
        moe.shared.w2.weight.data.zero_()
        after = moe(x)
    assert not torch.allclose(before, after)


def test_moe_output_shape_and_gradients():
    cfg = moe_cfg(n_shared_experts=1)
    m = MotherBrain(cfg)
    x = torch.randint(0, 97, (3, 10))
    out = m(x, targets=x)
    assert out.logits.shape == (3, 10, 97)
    out.loss.backward()
    assert m.blocks[1].ffn.experts.w1.grad is not None


def test_active_parameters_below_total():
    cfg = moe_cfg(n_experts=16, n_experts_per_tok=2)
    m = MotherBrain(cfg)
    assert m.num_active_parameters() < m.num_parameters()


def test_router_jitter_only_applies_in_training():
    torch.manual_seed(0)
    cfg = moe_cfg(router_jitter=0.5)
    moe = MoEFeedForward(cfg).eval()
    x = torch.randn(1, 4, cfg.dim)
    with torch.no_grad():
        assert torch.allclose(moe(x), moe(x))
