# MotherBrain

A personal large language model, written from scratch in PyTorch. No
`transformers`, no `tokenizers`, no training framework — the tokenizer, the
model, the data pipeline, the training loop and the sampler are all in this
repo, in about 2,000 lines of readable Python.

It scales from a 2M-parameter model you can train on a laptop in minutes to a
**204-trillion-parameter** sparse configuration, using the same code path.

```
$ motherbrain params --ladder
preset               total      active  active %    bf16 weights
----------------------------------------------------------------
nano                 2.23M       2.23M    100.0%        4.25 MiB
micro               10.49M      10.49M    100.0%       20.01 MiB
mini                51.79M      51.79M    100.0%       98.77 MiB
small              100.09M     100.09M    100.0%      190.91 MiB
medium             315.93M     315.93M    100.0%      602.60 MiB
large              672.08M     672.08M    100.0%        1.25 GiB
xl                   1.18B       1.18B    100.0%        2.20 GiB
moe-small          515.40M     203.97M     39.6%      983.04 MiB
moe-large           50.96B       4.29B      8.4%       94.92 GiB
moe-xl             153.06B      11.83B      7.7%      285.10 GiB
titan                4.83T     111.17B      2.3%        8.78 TiB
motherbrain        204.71T     597.73B      0.3%      372.36 TiB
```

## Quickstart

```bash
pip install -e .

# 1. build a demo corpus (Python's own standard library, ~4.8M characters)
python scripts/prepare_demo.py

# 2. train a BPE tokenizer on it
motherbrain tokenizer 'data/corpus/*.txt' --split-on '<|document|>' \
    --vocab-size 4096 --out data/tokenizer.json

# 3. pack the corpus into train/val token shards
motherbrain data 'data/corpus/*.txt' --split-on '<|document|>' \
    --tokenizer data/tokenizer.json --out-dir data/tokens

# 4. train
motherbrain train --data-dir data/tokens --out-dir runs/demo \
    --vocab-size 4096 --dim 256 --n-layers 6 --n-heads 8 --n-kv-heads 4 \
    --max-seq-len 256 --batch-size 16 --max-steps 600 --lr 1e-3

# 5. generate
motherbrain sample runs/demo/final.pt --tokenizer data/tokenizer.json \
    --prompt 'def parse(self, ' --max-new-tokens 120
```

Steps 1–5 take about six minutes end to end on four CPU cores, with no GPU.
That run takes training loss from 8.36 to 3.54 and validation perplexity from
143 to 70, and the 9.9M-parameter result writes recognisable Python:

```python
def parse(self, ):
    """Results a function and return the stack."""
    raise ValueError("Cannot write for %s" % (self.__self__))

    def __all__(self, other):
        self.__builtin__ = self.__file__
```

It is confabulating, which is exactly what a 9.9M-parameter model trained for
five minutes should do. The point is that the whole pipeline works.

## Architecture

A pre-norm decoder-only transformer, using the choices that current frontier
models have converged on:

| Component | Choice | Why |
|---|---|---|
| Normalisation | RMSNorm | Cheaper than LayerNorm, no centring term needed |
| Position | RoPE, with linear scaling | Relative positions; the scaling knob extends context past training length |
| Attention | Grouped-query (GQA) | The KV cache, not the weights, is what bounds long-context inference |
| Feed-forward | SwiGLU | Better loss per parameter than GELU at matched size |
| Sparsity | Top-k MoE, shared experts | Total parameters grow while per-token compute stays fixed |
| Embeddings | Tied by default | Saves `vocab x dim` parameters at small scale |
| Attention kernel | `F.scaled_dot_product_attention` | Uses FlashAttention where the hardware supports it |

### How the parameter count gets so large

In a dense model, every parameter runs for every token, so total parameters and
per-token compute rise together — that is the wall.

A mixture-of-experts layer replaces one feed-forward network with `n_experts` of
them plus a router that picks `n_experts_per_tok` per token. Total parameters
scale with `n_experts`; FLOPs scale with `n_experts_per_tok`, which stays
constant. The `motherbrain` preset holds 4,096 experts across 124 layers and
activates 8, so **0.29%** of its 204.71T parameters run for any given token.

That is a real architecture with a real parameter count, and it is the same
architecture the `nano` preset uses. It is not a model anyone can train today:
its weights alone are 372 TiB in bf16, AdamW state pushes a training run past
2.9 PiB, and Chinchilla-optimal training would want ~12T tokens. The repo will
size and validate the configuration; it will not conjure the cluster.

```bash
motherbrain params --preset motherbrain   # full report for any preset
```

### Routing

Top-k routing with softmax-normalised weights over the selected experts, plus:

- **Load-balancing loss** (Switch Transformer): without it, the router collapses
  onto a few experts and the rest are dead weight. Its floor is 1.0 at perfectly
  uniform routing.
- **Router z-loss**: keeps router logits from drifting large, which is the usual
  source of MoE training instability.
- **Shared experts** (DeepSeek-style): always-on experts that absorb the general
  patterns every token needs, so routed experts can specialise.
- **Dense early layers**: `moe_first_dense_layers` keeps the first layers dense,
  where routing decisions are least informed.

## Layout

```
motherbrain/
  config.py      ModelConfig / TrainConfig / RunConfig, serialised into checkpoints
  tokenizer.py   byte-level BPE: training, encoding, decoding
  model.py       RMSNorm, RoPE, GQA attention, SwiGLU, MoE, KV cache
  data.py        token packing and memory-mapped batch loading
  train.py       AdamW, LR schedules, grad accumulation, checkpoint/resume
  sample.py      generation: temperature, top-k, top-p, repetition penalty
  scaling.py     analytic parameter counts, memory estimates, the preset ladder
  cli.py         the `motherbrain` command
tests/           122 tests
scripts/         corpus preparation
configs/         example run configs
```

## Design notes

**The tokenizer is lossless.** The pre-tokenizer regex is *total* — re-joining
its pieces reproduces the input byte for byte — so `decode(encode(x)) == x` for
any string, including unseen unicode and control bytes. This is easy to get
wrong: Python's `\w` includes underscore, so the obvious "unicode letter" class
`[^\W\d_]` silently drops every `_` in the corpus. There is a test for it.

**The KV cache is verified, not assumed.** A test asserts that cached
incremental decoding produces bit-identical logits to recomputing the full
sequence. Cache bugs otherwise surface as mysteriously bad generation quality
long after the fact.

**Parameter counts are computed analytically and checked against reality.**
`scaling.py` derives counts from a config without building anything — which is
the only way to size a 204T-parameter model — and `verify_counts` asserts those
formulas match an instantiated `nn.Module` exactly, across dense, tied, untied,
GQA and MoE variants.

**Training is checked by training.** The suite runs real training runs on a real
corpus and asserts the loss actually falls and that a trained model beats the
uniform baseline on held-out data. A transformer with a subtly broken causal
mask or residual stream still runs; it just learns badly.

## Testing

```bash
pip install -e '.[dev]'
pytest
```

122 tests covering tokenizer roundtripping, causal masking (future tokens
provably cannot leak backwards), KV-cache equivalence, MoE routing (only
selected experts affect a token), LR schedules, checkpoint resume, parameter
accounting, and end-to-end convergence.

## License

MIT. See [LICENSE](LICENSE).
