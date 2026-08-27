# MotherBrain

My own personally made Large Language Model.

MotherBrain is a decoder-only transformer written in TensorFlow/Keras that
**grows with every version**. Each release is derived from the one before it by
scaling the architecture up, so the parameter count strictly increases at every
version bump — and that is enforced in code, not just by convention.

```
version       parameters    growth  architecture
0.1.0              4.89M         -  d256 x6L ff704 ctx1024
0.2.0              8.98M     1.84x  d320 x7L ff896 ctx1024
0.2.1              9.41M     1.05x  d320 x7L ff960 ctx1024
0.3.0             15.44M     1.64x  d384 x8L ff1152 ctx1024
1.0.0             69.18M     4.48x  d576 x16L ff1728 ctx2048
```

## Install

```bash
pip install -e .          # tensorflow-cpu + numpy
pip install -e ".[dev]"   # plus pytest
```

## Feeding it information

MotherBrain learns from plain text. Point it at a file or a directory and it
reads every text file underneath (`.txt`, `.md`, `.json`, `.jsonl`, `.csv`,
`.py`, `.rst`).

```bash
# 1. Train the latest release on your own text.
python -m motherbrain train --data path/to/your/notes --epochs 3

# 2. Ask it to continue something.
python -m motherbrain generate "The thing about attention is"
```

`train` writes `checkpoints/motherbrain-<version>.weights.h5` alongside a
`.json` file recording the version and architecture those weights belong to.
Run `train` again and it picks up where it left off; pass `--restart` to begin
from fresh weights. `train.py` and `generate.py` in the repository root are
thin wrappers if you prefer running scripts directly.

Useful training flags: `--seq-len`, `--batch-size`, `--learning-rate`,
`--epochs`, `--validation-split`, `--version`.
Useful generation flags: `--temperature`, `--top-k`, `--top-p`,
`--max-new-tokens`, `--seed`.

Text is tokenized as raw UTF-8 bytes (259 ids: 256 bytes plus `<bos>`,
`<eos>`, `<pad>`), so anything you can save as a file can be fed in without
training a vocabulary first.

### What to expect

A freshly initialised MotherBrain knows nothing — its weights are random and
its output is noise. Everything it knows has to be trained in. The seed release
is ~4.9M parameters, which is *tiny*: on a few kilobytes of text it will
memorise rather than generalise, and on CPU you want megabytes of text and
patience before output starts looking like language. Version numbers describe
capacity; training describes what is actually in there.

## Growing it

```bash
python -m motherbrain history              # every release and its size
python -m motherbrain show 0.3.0           # one release in detail
python -m motherbrain evolve minor --write # grow, and record the new release
python -m motherbrain validate             # check the growth invariant
```

Each bump scales the model differently:

| Bump    | What grows                                             |
| ------- | ------------------------------------------------------ |
| `patch` | the feed-forward width (`d_ff`)                        |
| `minor` | the residual stream (`d_model`), plus one more layer   |
| `major` | width, double the depth, and double the context window |

Widths are rounded **up** onto a 64-element grid, dimensions are never scaled
down, and attention heads grow with the width so `head_dim` stays roughly
fixed. `GrowthPolicy.grow` raises rather than return an architecture that is not
strictly larger than its parent, so a policy edit can't quietly break growth.

## The growth invariant

The release history lives in `motherbrain/lineage.json` as an ordered chain.
`Lineage.validate()` runs on construction, on every `evolve`, and on every load,
and requires that at each step **both** the version and the parameter count
strictly increase. Loading also recomputes each parameter count from its
architecture and rejects a file whose recorded number disagrees — a hand-edited
lineage cannot smuggle in a model that shrinks.

`Architecture.parameter_count` is derived from the dimensions rather than
stored, and `tests/test_model.py` builds real Keras models to confirm the
formula matches `model.count_params()` exactly, including after growth.

## Layout

| Path                         | What it is                                  |
| ---------------------------- | ------------------------------------------- |
| `motherbrain/version.py`     | semantic versions and bumps                 |
| `motherbrain/architecture.py`| dimensions and the parameter-count formula  |
| `motherbrain/growth.py`      | how each bump scales the architecture       |
| `motherbrain/lineage.py`     | the release chain and its growth invariant  |
| `motherbrain/model.py`       | the network: RMSNorm, RoPE, SwiGLU, no biases|
| `motherbrain/tokenizer.py`   | byte-level tokenizer                        |
| `motherbrain/data.py`        | corpus loading and next-token windows       |
| `motherbrain/training.py`    | the training loop                           |
| `motherbrain/generation.py`  | sampling (temperature, top-k, top-p)        |
| `motherbrain/checkpoint.py`  | weights plus the metadata describing them   |
| `data/sample/`               | a small corpus to smoke-test the pipeline   |

Weights are not committed: `motherbrain init --seed N` regenerates identical
random weights, and trained weights come from your own text.

## Tests

```bash
python -m pytest              # 80 tests
python -m pytest -m "not slow" # skip the ones that build Keras models
```
