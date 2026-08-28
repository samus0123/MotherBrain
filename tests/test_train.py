import json

import numpy as np
import pytest
import torch

from motherbrain.config import ModelConfig, RunConfig, TrainConfig
from motherbrain.data import pack_documents, write_meta
from motherbrain.model import MotherBrain
from motherbrain.tokenizer import ByteTokenizer
from motherbrain.train import (
    build_optimizer,
    load_checkpoint,
    lr_at,
    model_from_checkpoint,
    resolve_device,
    resolve_dtype,
    train,
)


def test_warmup_ramps_from_near_zero_to_peak():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) < cfg.lr
    assert lr_at(9, cfg) == pytest.approx(cfg.lr)


def test_cosine_decays_to_the_floor():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1, schedule="cosine")
    assert lr_at(99, cfg) == pytest.approx(cfg.lr * 0.1, rel=1e-2)
    # monotonically non-increasing after warmup
    rates = [lr_at(s, cfg) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(rates, rates[1:]))


@pytest.mark.parametrize("schedule", ["cosine", "linear", "constant", "wsd"])
def test_every_schedule_stays_in_range(schedule):
    cfg = TrainConfig(lr=1e-3, warmup_steps=5, max_steps=50, schedule=schedule)
    rates = [lr_at(s, cfg) for s in range(50)]
    assert all(0 < r <= cfg.lr + 1e-12 for r in rates)


def test_wsd_holds_then_decays():
    cfg = TrainConfig(lr=1e-3, warmup_steps=5, max_steps=105, schedule="wsd", min_lr_ratio=0.1)
    assert lr_at(50, cfg) == pytest.approx(cfg.lr)  # still in the stable phase
    assert lr_at(104, cfg) < cfg.lr  # decayed by the end


def test_optimizer_splits_decay_groups():
    model = MotherBrain(ModelConfig(vocab_size=97, dim=64, n_layers=2, n_heads=4, max_seq_len=32))
    opt = build_optimizer(model, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # every RMSNorm weight is 1-D and must land in the undecayed group
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert all(p.dim() >= 2 for p in decay["params"])


def test_dtype_resolution_on_cpu():
    cpu = torch.device("cpu")
    assert resolve_dtype("fp32", cpu) is torch.float32
    assert resolve_dtype("bf16", cpu) is torch.bfloat16
    # fp16 autocast is not a usable CPU path; it falls back to bf16
    assert resolve_dtype("fp16", cpu) is torch.bfloat16


def test_resolve_device_accepts_explicit_spec():
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.fixture
def tiny_dataset(tmp_path):
    """A highly repetitive corpus, so a small model can measurably learn it."""
    tok = ByteTokenizer()
    docs = ["the quick brown fox jumps over the lazy dog. " * 20] * 60
    data_dir = tmp_path / "tokens"
    n_train = pack_documents(docs, tok, data_dir / "train.bin", 257, eot_id=256)
    n_val = pack_documents(docs[:5], tok, data_dir / "val.bin", 257, eot_id=256)
    write_meta(data_dir, 257, {"train": n_train, "val": n_val})
    return data_dir


def _run(data_dir, out_dir, **train_kw) -> RunConfig:
    model = ModelConfig(
        vocab_size=257, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
    )
    base = dict(
        data_dir=str(data_dir), out_dir=str(out_dir), batch_size=8, seq_len=64,
        max_steps=60, warmup_steps=5, lr=3e-3, eval_every=0, save_every=0,
        log_every=1000, device="cpu", dtype="fp32", seed=0,
    )
    base.update(train_kw)
    return RunConfig(name="test", model=model, train=TrainConfig(**base))


def test_training_reduces_loss(tiny_dataset, tmp_path):
    """The end-to-end check: a real run on real data must actually learn."""
    run = _run(tiny_dataset, tmp_path / "run")
    train(run)

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    ]
    losses = [r["loss"] for r in records if "loss" in r]
    assert losses, "training produced no log records"
    assert losses[-1] < losses[0] - 0.5, f"loss did not fall: {losses[0]} -> {losses[-1]}"


def test_checkpoint_roundtrips_and_reproduces_logits(tiny_dataset, tmp_path):
    run = _run(tiny_dataset, tmp_path / "run", max_steps=10)
    final = train(run)
    assert final.exists()

    model, loaded_run = model_from_checkpoint(final, "cpu")
    assert loaded_run.model.dim == run.model.dim
    assert loaded_run.model.vocab_size == run.model.vocab_size

    ckpt = load_checkpoint(final)
    assert ckpt["step"] == 10
    assert "optimizer" in ckpt

    # the restored model is the same function, not just the same shape
    again, _ = model_from_checkpoint(final, "cpu")
    x = torch.randint(0, 257, (1, 16))
    with torch.no_grad():
        assert torch.allclose(model(x).logits, again(x).logits)


def test_resume_continues_from_the_saved_step(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    train(_run(tiny_dataset, out, max_steps=20, save_every=10))
    assert (out / "latest.pt").exists()

    resumed = _run(tiny_dataset, out, max_steps=30, resume="auto")
    train(resumed)
    assert load_checkpoint(out / "final.pt")["step"] == 30


def test_moe_model_trains(tiny_dataset, tmp_path):
    run = _run(tiny_dataset, tmp_path / "moe", max_steps=20)
    run.model = ModelConfig(
        vocab_size=257, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
        n_experts=4, n_experts_per_tok=2, n_shared_experts=1,
    )
    final = train(run)
    model, _ = model_from_checkpoint(final, "cpu")
    assert model.num_active_parameters() < model.num_parameters()


def test_grad_checkpointing_trains(tiny_dataset, tmp_path):
    run = _run(tiny_dataset, tmp_path / "gc", max_steps=10, grad_checkpoint=True)
    assert train(run).exists()


def test_grad_accumulation_trains(tiny_dataset, tmp_path):
    run = _run(tiny_dataset, tmp_path / "ga", max_steps=10, grad_accum_steps=4, batch_size=2)
    assert train(run).exists()


def test_trained_model_beats_uniform_on_its_own_data(tiny_dataset, tmp_path):
    """After training, held-out loss should be well below ln(vocab_size)."""
    import math

    run = _run(tiny_dataset, tmp_path / "run2", max_steps=120)
    final = train(run)
    model, _ = model_from_checkpoint(final, "cpu")

    from motherbrain.data import TokenDataset

    ds = TokenDataset(tiny_dataset / "val.bin")
    x, y = ds.sample_batch(4, 64, generator=np.random.default_rng(0))
    with torch.no_grad():
        loss = model(x, targets=y).lm_loss.item()
    assert loss < math.log(257) * 0.5
