# MotherBrain

A from-scratch decoder-only language model with a training stack built to run
real jobs: multi-GPU, mixed precision, resumable, and tested.

Nothing here wraps a pretrained model. The architecture, tokenizer, data
pipeline and training loop are all in this repo, in about 2k lines of Python.

## What's in the box

**Architecture** (`src/motherbrain/model.py`) — a Llama-style transformer:

| Component | Choice | Why |
|---|---|---|
| Normalization | RMSNorm, pre-norm | Cheaper than LayerNorm, stabler than post-norm at depth |
| Positions | Rotary (RoPE) | Relative positions, no learned table, extrapolates better |
| Attention | Grouped-query, via fused SDPA | Shrinks the KV cache at inference; flash kernels when available |
| MLP | SwiGLU | Better loss per parameter than GELU at equal width |
| Biases | None | They cost parameters and buy nothing at this scale |
| Head | Optionally tied to embeddings | Saves `vocab_size x d_model` parameters |

**Training** (`src/motherbrain/train.py`) — DDP via `torchrun`, bf16/fp16
autocast with loss scaling, gradient accumulation, gradient clipping, cosine
decay with warmup, `torch.compile`, periodic eval, and atomic rotating
checkpoints.

**Data** (`src/motherbrain/data.py`) — corpora are pre-tokenized into flat
`.bin` shards read through `np.memmap`, so the dataset never has to fit in RAM.
The loader is deterministic and resumable: restoring a checkpoint replays the
exact remaining sample stream.

## Quick start

```bash
pip install -e ".[tokenizer,data,dev]"

# 1. Train a tokenizer on your corpus
python scripts/train_tokenizer.py --input data/raw/*.txt \
    --vocab-size 32000 --out data/tokenized/tokenizer.json

# 2. Tokenize into memory-mapped shards (+ a held-out val split)
python scripts/prepare_data.py --input data/raw/*.txt \
    --tokenizer data/tokenized/tokenizer.json --out data/tokenized

# 3. Train
python -m motherbrain.train --config configs/small_110m.yaml

# 4. Sample
python -m motherbrain.sample --ckpt runs/small_110m/best.pt --prompt "Once upon a time"
```

Multi-GPU is the same command under `torchrun`:

```bash
torchrun --nproc_per_node=8 -m motherbrain.train --config configs/small_110m.yaml
```

Resume an interrupted run — weights, optimizer state, RNG and data position all
come back:

```bash
python -m motherbrain.train --config configs/small_110m.yaml --resume auto
```

## Configuration

YAML files compose via `inherit:`, and anything can be overridden on the command
line without editing a file:

```bash
python -m motherbrain.train --config configs/small_110m.yaml \
    optim.lr=3e-4 train.batch_size=32 train.compile=false
```

| Config | Params (total) | Context | Reference budget |
|---|---|---|---|
| `configs/debug.yaml` | 0.6M | 128 | seconds on CPU, for smoke tests |
| `configs/small_110m.yaml` | 110M | 1024 | 40k steps ≈ 21B tokens |
| `configs/medium_320m.yaml` | 316M | 2048 | 60k steps ≈ 63B tokens |

Both reference budgets assume 8 GPUs. On a different count, scale
`train.grad_accum_steps` by `8 / num_gpus` to hold the effective batch — and
therefore the tuned learning-rate schedule — fixed.

The budgets deliberately exceed the compute-optimal (Chinchilla) point of
roughly 20 tokens per parameter. Compute-optimal minimizes loss for a fixed
*training* budget; if the model is going to be served, overtraining a smaller
model is usually the better trade.

## Scaling the model

`d_ff` and `n_kv_head` derive themselves from `d_model` and `n_head` unless you
set them, so a new size is normally three numbers:

```yaml
model:
  n_layer: 32
  n_head: 32
  d_model: 4096      # d_ff becomes 11008 automatically
```

Rules of thumb that hold up: keep `d_model / n_layer` near 64–128, keep
`head_dim` (`d_model / n_head`) at 64 or 128, and lower the learning rate as
width grows (`6e-4` at 768 wide, `3e-4` at 1024, `1.5e-4` at 2048).

## Monitoring

Each run writes `metrics.jsonl` (one record per logged step) and `config.json`
to its `out_dir`. Set `train.wandb_project` to mirror metrics to Weights &
Biases.

Two numbers worth watching: **grad_norm**, which should settle to a stable band
after warmup — a sustained climb means the LR is too high — and **val loss**
against train loss, which start diverging when the model begins memorizing.

## Development

```bash
pytest                      # 112 tests, ~4s on CPU
ruff check src tests scripts
```

The suite covers the parts that fail silently rather than loudly: that
attention is actually causal, that KV-cache decoding matches a full forward
pass, that gradient accumulation equals one large batch, that RoPE scores
depend only on relative position, and that a resumed run reproduces an
uninterrupted one bit-for-bit.

## Project layout

```
src/motherbrain/
  config.py       typed configs, YAML inheritance, CLI overrides
  model.py        the transformer, KV cache, sampling
  data.py         memmapped shards, deterministic resumable loader
  tokenizer.py    byte-level BPE
  optim.py        AdamW param groups, LR schedules
  train.py        the training loop
  sample.py       generation CLI
  evaluate.py     loss/perplexity CLI
  checkpoint.py   atomic save/load/rotate
  distributed.py  torchrun helpers
scripts/          tokenizer training, corpus tokenization
configs/          debug / 110M / 320M
tests/
```

## License

MIT — see [LICENSE](LICENSE).
