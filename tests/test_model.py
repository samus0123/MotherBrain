import math

import pytest
import torch

from motherbrain.config import ModelConfig
from motherbrain.model import RMSNorm, Transformer, apply_rope, build_rope_cache, repeat_kv


def test_forward_shapes_and_loss(model_cfg):
    model = Transformer(model_cfg)
    x = torch.randint(0, model_cfg.vocab_size, (3, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, 16, model_cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_no_targets_returns_no_loss(model_cfg):
    logits, loss = Transformer(model_cfg)(torch.randint(0, model_cfg.vocab_size, (1, 8)))
    assert loss is None and logits.shape[-1] == model_cfg.vocab_size


def test_init_loss_is_near_uniform():
    """A freshly initialized model should sit at ln(vocab_size) on random targets."""
    cfg = ModelConfig(vocab_size=512, n_layer=4, n_head=4, d_model=128, max_seq_len=64)
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    x = torch.randint(0, 512, (8, 64))
    y = torch.randint(0, 512, (8, 64))
    with torch.no_grad():
        _, loss = model(x, targets=y)
    assert loss.item() == pytest.approx(math.log(512), abs=0.15)


def test_attention_is_causal(model_cfg):
    """Changing the last token must not perturb any earlier position's logits."""
    torch.manual_seed(0)
    model = Transformer(model_cfg).eval()
    x = torch.randint(0, model_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % model_cfg.vocab_size
        perturbed, _ = model(x2)
    assert torch.allclose(base[:, :-1], perturbed[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], perturbed[:, -1], atol=1e-5)


def test_kv_cache_matches_full_forward(model_cfg):
    """Incremental decoding with a cache must equal a single full forward pass."""
    torch.manual_seed(0)
    model = Transformer(model_cfg).eval()
    x = torch.randint(0, model_cfg.vocab_size, (2, 20))
    with torch.no_grad():
        full, _ = model(x)
        caches = model.init_caches(2, model_cfg.max_seq_len)
        prefill, _ = model(x[:, :12], caches=caches, start_pos=0)
        pieces = [prefill]
        for i in range(12, 20):
            step, _ = model(x[:, i : i + 1], caches=caches, start_pos=i)
            pieces.append(step)
        incremental = torch.cat(pieces, dim=1)
    assert torch.allclose(full, incremental, atol=1e-4)


def test_kv_cache_overflow_raises(model_cfg):
    model = Transformer(model_cfg).eval()
    caches = model.init_caches(1, 8)
    with pytest.raises(ValueError, match="overflow"):
        model(torch.zeros(1, 9, dtype=torch.long), caches=caches, start_pos=0)


def test_sequence_longer_than_context_raises(model_cfg):
    model = Transformer(model_cfg)
    too_long = torch.zeros(1, model_cfg.max_seq_len + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)


def test_grouped_query_attention_shapes():
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=8, n_kv_head=2, d_model=64, max_seq_len=16)
    model = Transformer(cfg)
    attn = model.blocks[0].attn
    assert attn.n_rep == 4
    assert attn.k_proj.weight.shape == (2 * cfg.head_dim, 64)
    assert attn.q_proj.weight.shape == (8 * cfg.head_dim, 64)
    logits, _ = model(torch.randint(0, 32, (1, 16)))
    assert logits.shape == (1, 16, 32)


def test_repeat_kv_duplicates_heads():
    x = torch.randn(2, 2, 5, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 5, 4)
    # Each source head is repeated contiguously.
    assert torch.equal(out[:, 0], x[:, 0]) and torch.equal(out[:, 2], x[:, 0])
    assert torch.equal(out[:, 3], x[:, 1])


def test_repeat_kv_identity():
    x = torch.randn(1, 3, 4, 2)
    assert torch.equal(repeat_kv(x, 1), x)


def test_rope_preserves_norm():
    """A rotation changes direction, never magnitude."""
    cos, sin = build_rope_cache(16, 8, 10000.0)
    x = torch.randn(2, 3, 16, 8)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_rope_is_relative():
    """q.k after RoPE depends only on the offset between positions."""
    cos, sin = build_rope_cache(64, 16, 10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(pos_q, pos_k):
        qr = apply_rope(q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        kr = apply_rope(k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (qr * kr).sum()

    assert score(5, 3) == pytest.approx(score(20, 18), abs=1e-5)
    assert score(5, 3) != pytest.approx(score(20, 10), abs=1e-3)


def test_rope_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="even"):
        build_rope_cache(8, 7, 10000.0)


def test_rmsnorm_scales_to_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(4, 16) * 10
    rms = norm(x).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-3)


def test_tied_embeddings_share_storage():
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, max_seq_len=8)
    tied = Transformer(cfg)
    assert tied.lm_head.weight is tied.tok_emb.weight

    cfg_untied = ModelConfig(
        vocab_size=32, n_layer=1, n_head=2, d_model=16, max_seq_len=8, tie_embeddings=False
    )
    untied = Transformer(cfg_untied)
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.num_params(False) > tied.num_params(False)


def test_num_params_excludes_embeddings(model_cfg):
    model = Transformer(model_cfg)
    emb = model_cfg.vocab_size * model_cfg.d_model
    assert model.num_params(False) - model.num_params(True) == emb


def test_generate_extends_and_respects_budget(model_cfg):
    torch.manual_seed(0)
    model = Transformer(model_cfg).eval()
    prompt = torch.randint(0, model_cfg.vocab_size, (2, 4))
    out = model.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=8)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], prompt)  # prompt is preserved verbatim
    # Never exceeds the context window even when asked to.
    capped = model.generate(prompt, max_new_tokens=1000)
    assert capped.shape[1] <= model_cfg.max_seq_len


def test_greedy_generation_is_deterministic(model_cfg):
    torch.manual_seed(0)
    model = Transformer(model_cfg).eval()
    prompt = torch.randint(0, model_cfg.vocab_size, (1, 4))
    a = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_top_k_restricts_support(model_cfg):
    """With top_k=1 sampling must agree with greedy decoding."""
    torch.manual_seed(0)
    model = Transformer(model_cfg).eval()
    prompt = torch.randint(0, model_cfg.vocab_size, (1, 4))
    greedy = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    top1 = model.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1)


class _ScriptedModel(Transformer):
    """A Transformer whose sampler emits a fixed script, one row per column."""

    def set_script(self, script):
        self._script = script
        self._step = 0

    def _sample_next(self, logits, temperature, top_k, top_p, generator):
        row = self._script[min(self._step, len(self._script) - 1)]
        self._step += 1
        return torch.tensor(row, dtype=torch.long, device=logits.device).unsqueeze(-1)


def test_generate_stops_at_eos(model_cfg):
    eos = 7
    model = _ScriptedModel(model_cfg).eval()
    model.set_script([[3], [4], [eos], [5]])
    out = model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=10, eos_id=eos)
    # Generation halts on the EOS rather than running out the budget.
    assert out.shape[1] == 5
    assert out[0, -1].item() == eos


def test_generate_pads_finished_rows_with_eos(model_cfg):
    """When one sequence in a batch ends early, it is padded, not left to drift."""
    eos = 7
    model = _ScriptedModel(model_cfg).eval()
    model.set_script([[3, 4], [eos, 5], [9, eos]])
    out = model.generate(torch.zeros(2, 2, dtype=torch.long), max_new_tokens=10, eos_id=eos)
    assert out.shape == (2, 5)
    assert out[0, 2:].tolist() == [3, eos, eos]  # row 0 finished first, then padded
    assert out[1, 2:].tolist() == [4, 5, eos]


def test_generate_without_eos_runs_full_budget(model_cfg):
    model = _ScriptedModel(model_cfg).eval()
    model.set_script([[7]])
    out = model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=6)
    assert out.shape[1] == 8


def test_generate_restores_training_mode(model_cfg):
    model = Transformer(model_cfg)
    model.train()
    model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)
    assert model.training


def test_dropout_is_active_only_in_train_mode():
    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=2, d_model=16, max_seq_len=8, dropout=0.5)
    torch.manual_seed(0)
    model = Transformer(cfg)
    x = torch.randint(0, 32, (2, 8))
    model.train()
    assert not torch.allclose(model(x)[0], model(x)[0])
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)[0], model(x)[0])


def test_gradients_reach_every_parameter(model_cfg):
    model = Transformer(model_cfg)
    x = torch.randint(0, model_cfg.vocab_size, (2, 8))
    _, loss = model(x, targets=x)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_flops_per_token_is_positive(model_cfg):
    assert Transformer(model_cfg).flops_per_token() > 0
