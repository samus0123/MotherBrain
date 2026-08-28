import pytest
import torch

from motherbrain.config import ModelConfig
from motherbrain.model import KVCache, MotherBrain, apply_rope, build_rope_cache


def tiny(**kw) -> ModelConfig:
    base = dict(
        vocab_size=97, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes_and_loss():
    m = MotherBrain(tiny())
    x = torch.randint(0, 97, (2, 8))
    out = m(x, targets=x)
    assert out.logits.shape == (2, 8, 97)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)


def test_untrained_loss_is_near_uniform():
    """A fresh model should sit close to ln(vocab_size) on unpredictable targets.

    The targets have to be independent of the inputs. Reusing the input as the
    target instead measures how easily the model copies the current token, which
    tied embeddings make easy even at init, and lands well below ln(V).
    """
    import math

    torch.manual_seed(0)
    m = MotherBrain(tiny(vocab_size=512)).eval()
    x = torch.randint(0, 512, (4, 16))
    y = torch.randint(0, 512, (4, 16))
    with torch.no_grad():
        loss = m(x, targets=y).loss.item()
    assert abs(loss - math.log(512)) < 0.5


def test_inference_returns_only_last_position():
    m = MotherBrain(tiny())
    out = m(torch.randint(0, 97, (2, 8)))
    assert out.logits.shape == (2, 1, 97)
    assert out.loss is None


def test_gqa_and_mha_both_run():
    for n_kv in (1, 2, 4):
        m = MotherBrain(tiny(n_kv_heads=n_kv))
        assert m(torch.randint(0, 97, (1, 6))).logits.shape == (1, 1, 97)


def test_bad_head_config_raises():
    with pytest.raises(ValueError):
        ModelConfig(dim=64, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(dim=65, n_heads=4)


def test_sequence_longer_than_context_raises():
    m = MotherBrain(tiny(max_seq_len=16))
    with pytest.raises(ValueError):
        m(torch.randint(0, 97, (1, 17)))


def test_causal_masking_future_tokens_do_not_leak():
    """Changing token t must not alter the prediction made at position t-1."""
    torch.manual_seed(0)
    m = MotherBrain(tiny()).eval()
    x = torch.randint(0, 97, (1, 10))
    with torch.no_grad():
        a = m(x, targets=x).logits
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 97
        b = m(x2, targets=x2).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    m = MotherBrain(tiny()).eval()
    x = torch.randint(0, 97, (2, 6))
    caches = m.make_caches(2)
    with torch.no_grad():
        cached = m(x, caches=caches, start_pos=0).logits
        assert torch.allclose(cached, m(x).logits, atol=1e-5)
        # one incremental step
        nxt = torch.randint(0, 97, (2, 1))
        step = m(nxt, caches=caches, start_pos=6).logits
        full = m(torch.cat([x, nxt], dim=1)).logits
    assert torch.allclose(step, full, atol=1e-4)


def test_kv_cache_overflow_raises():
    cache = KVCache.empty(1, 2, 4, 8, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):
        cache.update(torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8))


def test_rope_is_position_dependent():
    cos, sin = build_rope_cache(8, 16)
    x = torch.ones(1, 1, 16, 8)
    y = apply_rope(x, cos, sin)
    assert not torch.allclose(y[0, 0, 0], y[0, 0, 5])
    # RoPE is a rotation, so it preserves the norm of each pair.
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_tied_embeddings_share_storage():
    m = MotherBrain(tiny(tie_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    untied = MotherBrain(tiny(tie_embeddings=False))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()


def test_every_parameter_receives_gradient():
    m = MotherBrain(tiny(n_experts=4, n_experts_per_tok=2, n_shared_experts=1))
    x = torch.randint(0, 97, (4, 12))
    m(x, targets=x).loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == []


def test_gradient_checkpointing_matches_plain_forward():
    torch.manual_seed(0)
    m = MotherBrain(tiny())
    x = torch.randint(0, 97, (2, 8))
    plain = m(x, targets=x).loss
    m.set_grad_checkpointing(True)
    m.train()
    checkpointed = m(x, targets=x).loss
    assert torch.allclose(plain, checkpointed, atol=1e-5)


def test_logit_softcap_bounds_logits():
    m = MotherBrain(tiny(logit_softcap=5.0))
    logits = m(torch.randint(0, 97, (2, 8))).logits
    assert logits.abs().max() <= 5.0


def test_dropout_is_disabled_in_eval():
    torch.manual_seed(0)
    m = MotherBrain(tiny(dropout=0.5)).eval()
    x = torch.randint(0, 97, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(x).logits, m(x).logits)
