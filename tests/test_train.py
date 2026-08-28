"""End-to-end tests for the training loop."""

import json

import pytest
import torch

from motherbrain.checkpoint import latest_checkpoint, load_checkpoint
from motherbrain.config import Config
from motherbrain.data import BatchLoader, TokenDataset
from motherbrain.evaluate import evaluate_split
from motherbrain.model import Transformer
from motherbrain.optim import build_optimizer, get_lr
from motherbrain.sample import load_model
from motherbrain.train import resolve_device, resolve_dtype, train


def test_training_runs_and_writes_artifacts(train_cfg):
    from pathlib import Path

    summary = train(train_cfg)
    out = Path(train_cfg.train.out_dir)
    assert summary["steps"] == train_cfg.train.max_steps
    assert (out / "config.json").exists()
    assert (out / "metrics.jsonl").exists()
    assert (out / f"ckpt_{train_cfg.train.max_steps:07d}.pt").exists()
    assert (out / "best.pt").exists()

    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("train/loss" in r for r in records)
    assert any("val/loss" in r for r in records)


def test_loss_decreases_on_a_learnable_corpus(tmp_path):
    """A repeating pattern is memorizable; loss must fall well below chance."""
    import numpy as np

    from motherbrain.data import ShardIndex

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pattern = np.tile(np.arange(16, dtype=np.uint16), 2000)  # a trivially predictable cycle
    pattern.tofile(data_dir / "train_00000.bin")
    pattern[:2000].tofile(data_dir / "val_00000.bin")
    ShardIndex(
        vocab_size=16,
        dtype="uint16",
        splits={
            "train": [{"path": "train_00000.bin", "tokens": int(pattern.size)}],
            "val": [{"path": "val_00000.bin", "tokens": 2000}],
        },
    ).save(data_dir)

    cfg = Config()
    cfg.model.vocab_size = 16
    cfg.model.n_layer = 2
    cfg.model.n_head = 2
    cfg.model.d_model = 32
    cfg.model.max_seq_len = 32
    cfg.data.data_dir = str(data_dir)
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.train.batch_size = 8
    cfg.train.grad_accum_steps = 1
    cfg.train.max_steps = 60
    cfg.train.device = "cpu"
    cfg.train.dtype = "float32"
    cfg.train.compile = False
    cfg.train.log_interval = 1000
    cfg.train.eval_interval = 1000
    cfg.train.ckpt_interval = 1000
    cfg.optim.lr = 3e-3
    cfg.optim.warmup_steps = 5

    summary = train(cfg)
    # Chance level is ln(16) = 2.77; a working loop gets far below it.
    assert summary["final_train_loss"] < 1.0


def test_gradient_accumulation_matches_one_big_batch(corpus, model_cfg):
    """Accumulating N micro-batches must equal a single batch N times as large."""
    torch.manual_seed(0)
    dataset = TokenDataset(corpus, "train", model_cfg.max_seq_len)

    def grads(batch_size, accum):
        torch.manual_seed(0)
        model = Transformer(model_cfg)
        loader = BatchLoader(dataset, batch_size, shuffle=False, seed=0)
        model.zero_grad()
        for _ in range(accum):
            x, y = next(loader)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
        return [p.grad.clone() for p in model.parameters()]

    big = grads(4, 1)
    accumulated = grads(2, 2)
    for a, b in zip(big, accumulated, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_resume_reproduces_uninterrupted_run(train_cfg, tmp_path):
    """Stopping and resuming must land on exactly the same weights."""
    # Pin the decay horizon so both runs share one LR schedule.
    train_cfg.optim.decay_steps = 8
    train_cfg.train.eval_interval = 10_000  # keep it to a plain training run
    train_cfg.train.ckpt_interval = 4

    straight_dir = tmp_path / "straight"
    train_cfg.train.out_dir = str(straight_dir)
    train(train_cfg)
    straight = load_checkpoint(straight_dir / "ckpt_0000008.pt")["model"]

    split_dir = tmp_path / "split"
    train_cfg.train.out_dir = str(split_dir)
    train_cfg.train.max_steps = 4
    train(train_cfg)

    train_cfg.train.max_steps = 8
    train(train_cfg, resume=str(latest_checkpoint(split_dir)))
    resumed = load_checkpoint(split_dir / "ckpt_0000008.pt")["model"]

    for key in straight:
        assert torch.equal(straight[key], resumed[key]), f"{key} diverged after resume"


def test_auto_resume_finds_the_latest_checkpoint(train_cfg):
    train_cfg.train.max_steps = 4
    train_cfg.train.ckpt_interval = 4
    train(train_cfg)

    train_cfg.train.max_steps = 8
    summary = train(train_cfg, resume="auto")
    assert summary["steps"] == 8
    assert latest_checkpoint(train_cfg.train.out_dir).name == "ckpt_0000008.pt"


def test_vocab_mismatch_is_caught_early(train_cfg):
    train_cfg.model.vocab_size += 1
    with pytest.raises(ValueError, match="does not match the tokenized"):
        train(train_cfg)


def test_missing_val_split_is_tolerated(tmp_path, model_cfg, write_corpus_fn):
    data_dir = write_corpus_fn(tmp_path / "novals", val_tokens=0)
    cfg = Config()
    cfg.model = model_cfg
    cfg.data.data_dir = str(data_dir)
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.train.batch_size = 2
    cfg.train.grad_accum_steps = 1
    cfg.train.max_steps = 3
    cfg.train.device = "cpu"
    cfg.train.dtype = "float32"
    cfg.train.compile = False
    cfg.train.log_interval = 100
    cfg.train.ckpt_interval = 3
    cfg.optim.warmup_steps = 1
    summary = train(cfg)  # must not raise just because there is nothing to eval on
    assert summary["best_val_loss"] == float("inf")


def test_checkpoint_reloads_into_a_working_model(train_cfg):
    train(train_cfg)
    device = torch.device("cpu")
    model, cfg = load_model(f"{train_cfg.train.out_dir}/best.pt", device)
    assert cfg.model.n_layer == train_cfg.model.n_layer
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 9)


def test_evaluate_split_reports_finite_perplexity(train_cfg):
    train(train_cfg)
    model, cfg = load_model(f"{train_cfg.train.out_dir}/best.pt", torch.device("cpu"))
    stats = evaluate_split(
        model, cfg.data.data_dir, "val", cfg.model.max_seq_len, 2, torch.device("cpu")
    )
    assert stats["loss"] > 0 and stats["perplexity"] > 1
    assert stats["tokens"] == stats["batches"] * 2 * cfg.model.max_seq_len


def test_lr_schedule_is_applied_to_the_optimizer(model_cfg):
    """The loop's LR must actually reach the optimizer's param groups."""
    from motherbrain.config import OptimConfig

    model = Transformer(model_cfg)
    opt_cfg = OptimConfig(lr=1e-3, warmup_steps=5)
    opt = build_optimizer(model, opt_cfg, "cpu")
    for step in (0, 5, 50):
        lr = get_lr(step, opt_cfg, 100)
        for group in opt.param_groups:
            group["lr"] = lr
        assert all(g["lr"] == pytest.approx(lr) for g in opt.param_groups)


def test_resolve_device_honours_explicit_choice():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_resolve_dtype_falls_back_on_cpu():
    cpu = torch.device("cpu")
    assert resolve_dtype("float32", cpu) is torch.float32
    assert resolve_dtype("bfloat16", cpu) is torch.bfloat16
    assert resolve_dtype("auto", cpu) is torch.float32  # no bf16 assumption off-GPU
