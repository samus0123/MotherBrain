import torch

from motherbrain.config import ModelConfig
from motherbrain.model import MotherBrain
from motherbrain.sample import apply_repetition_penalty, filter_top_k_top_p, generate


def tiny_model(**kw) -> MotherBrain:
    base = dict(vocab_size=97, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=48)
    base.update(kw)
    return MotherBrain(ModelConfig(**base)).eval()


def test_generate_appends_requested_tokens():
    m = tiny_model()
    out = generate(m, torch.randint(0, 97, (2, 5)), max_new_tokens=10, seed=0)
    assert out.shape == (2, 15)


def test_generate_accepts_1d_prompt():
    m = tiny_model()
    out = generate(m, torch.randint(0, 97, (6,)), max_new_tokens=4, seed=0)
    assert out.shape == (1, 10)


def test_kv_cache_and_full_recompute_agree_greedily():
    torch.manual_seed(0)
    m = tiny_model()
    prompt = torch.randint(0, 97, (2, 6))
    a = generate(m, prompt, max_new_tokens=12, temperature=0.0, use_cache=True)
    b = generate(m, prompt, max_new_tokens=12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_greedy_decoding_is_deterministic():
    m = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    assert torch.equal(
        generate(m, prompt, max_new_tokens=8, temperature=0.0),
        generate(m, prompt, max_new_tokens=8, temperature=0.0),
    )


def test_seeded_sampling_is_reproducible():
    m = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    a = generate(m, prompt, max_new_tokens=8, temperature=1.0, seed=123)
    b = generate(m, prompt, max_new_tokens=8, temperature=1.0, seed=123)
    assert torch.equal(a, b)


def test_generation_stops_at_context_limit():
    m = tiny_model(max_seq_len=16)
    out = generate(m, torch.randint(0, 97, (1, 10)), max_new_tokens=100)
    assert out.shape[1] <= 16


def test_eos_halts_and_pads():
    """Generation stops once EOS is produced, instead of using the full budget."""
    torch.manual_seed(0)
    m = tiny_model()
    prompt = torch.randint(0, 97, (1, 3))

    # Ask what greedy decoding emits first, then declare that token to be EOS.
    first = generate(m, prompt, max_new_tokens=1, temperature=0.0)[0, -1].item()
    out = generate(m, prompt, max_new_tokens=20, temperature=0.0, eos_id=first)

    assert out[0, 3].item() == first
    assert out.shape[1] < 3 + 20  # stopped early


def test_eos_pads_finished_rows_in_a_batch():
    """A finished sequence keeps emitting EOS while its batch-mates continue."""
    torch.manual_seed(0)
    m = tiny_model()
    prompt = torch.randint(0, 97, (2, 4))
    first = generate(m, prompt, max_new_tokens=1, temperature=0.0)[:, -1]
    out = generate(m, prompt, max_new_tokens=6, temperature=0.0, eos_id=int(first[0]))
    tail = out[0, 4:]
    assert (tail == int(first[0])).all()


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = filter_top_k_top_p(logits, top_k=2)
    assert int(torch.isfinite(filtered).sum()) == 2
    assert torch.isfinite(filtered[0, 1]) and torch.isfinite(filtered[0, 4])


def test_top_p_keeps_the_nucleus():
    # probabilities ~ [0.64, 0.24, 0.09, 0.03] -> top_p=0.8 keeps the first two
    logits = torch.log(torch.tensor([[0.64, 0.24, 0.09, 0.03]]))
    filtered = filter_top_k_top_p(logits, top_p=0.8)
    assert torch.isfinite(filtered[0, 0]) and torch.isfinite(filtered[0, 1])
    assert torch.isinf(filtered[0, 2]) and torch.isinf(filtered[0, 3])


def test_top_p_always_keeps_at_least_one_token():
    logits = torch.log(torch.tensor([[0.99, 0.01]]))
    filtered = filter_top_k_top_p(logits, top_p=0.1)
    assert int(torch.isfinite(filtered).sum()) >= 1


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    seen = torch.tensor([[1]])
    out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
    assert out[0, 1] < out[0, 0]


def test_repetition_penalty_of_one_is_a_noop():
    logits = torch.randn(1, 5)
    assert torch.equal(apply_repetition_penalty(logits.clone(), torch.tensor([[2]]), 1.0), logits)


def test_moe_model_generates():
    m = tiny_model(n_experts=4, n_experts_per_tok=2, n_shared_experts=1)
    assert generate(m, torch.randint(0, 97, (1, 4)), max_new_tokens=6, seed=0).shape == (1, 10)
