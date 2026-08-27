# MotherBrain

A language model you own end to end: a sparse mixture-of-experts transformer,
a tokenizer it learns from your own text, a training loop, an incremental
patching system that versions every new thing it learns, and an HTTP server
that any IDE can talk to.

```
mb scale      what a configuration costs, before you build it
mb feed       put information in
mb prepare    learn a vocabulary from it
mb train      train the base model            -> v0
mb patch      learn new information           -> v1, v2, v3 ...
mb versions   the lineage
mb checkout   go back to any earlier version
mb chat       talk to it locally
mb serve      expose it to every IDE you own
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python -m motherbrain.cli feed ./my-notes ./src        # anything textual
python -m motherbrain.cli prepare --vocab-size 4096
python -m motherbrain.cli train --preset micro --steps 400 --batch-size 16 --seq-len 256
python -m motherbrain.cli chat --prompt "hello"
python -m motherbrain.cli serve --host 0.0.0.0
```

## How do I feed it new information?

Four ways, all landing in the same corpus and all producing a new version.

**1. The command line.** Files, whole directories, or literal text:

```bash
python -m motherbrain.cli feed ./docs ./notes.md "The deploy key rotates on Fridays."
python -m motherbrain.cli patch --steps 150      # -> v1
```

**2. Over HTTP, from anywhere:**

```bash
curl -X POST http://your-host:8000/feed \
  -H 'Content-Type: application/json' \
  -d '{"text": "Postgres runs on port 6543 in staging."}'
```

With the server started normally (auto-patch on), that is the whole procedure.
The text is queued, and once feeding goes quiet the model trains a patch and
promotes itself to the next version without being asked.

**3. The web UI.** Open `http://your-host:8000`, use the **Feed** tab.

**4. Git.** Commit a file to `corpus/inbox/` and the CI pipeline does the rest —
see below.

Feeding stores text. *Training* is what absorbs it; with auto-patch on, the
second follows the first by itself.

## Versions

New information never retrains the model from scratch. It trains a **patch**: a
small low-rank delta, learned while the existing weights stay frozen. Applying
a patch mints the next sequential version.

```
v0  base checkpoint          (mb train)
v1  = v0 + patch-0001        the operating notes you fed it
v2  = v1 + patch-0002        the API docs you fed it next
```

```bash
python -m motherbrain.cli versions
python -m motherbrain.cli checkout v1     # any earlier version, exactly
```

Patches are kilobytes to megabytes, so the lineage is cheap to keep forever and
small enough to live in git. Two details make this work rather than merely run:

* **Replay.** Training only on new text makes a model forget the old text. Each
  patch trains on a mixture of the new information and a sample of everything
  before it.
* **A frozen, byte-level vocabulary.** New information can use words the base
  corpus never contained, with no vocabulary surgery and no unknown tokens.

Observed on the run in this repository — 2 documents, 252 tokens, rank 8,
224k trainable parameters, 150 steps, about a minute on 4 CPU cores:

```
v0 -> v1   loss on the new material 6.25 -> 1.26

  prompt: "The mother preset has"
  v0:  " a string of the\n#      "
  v1:  " 1157\ntrillion total parameters and acti"
```

## Use it from your IDE

MotherBrain speaks the two protocols editors already support, so no editor
needs a MotherBrain-specific plugin — you point it at a base URL.

```bash
python -m motherbrain.cli serve --host 0.0.0.0 --port 8000
```

| Protocol | Base URL | Endpoints |
|---|---|---|
| OpenAI | `http://host:8000/v1` | `/models`, `/chat/completions`, `/completions`, `/embeddings` |
| Ollama | `http://host:8000` | `/api/tags`, `/api/chat`, `/api/generate`, `/api/embeddings` |

Streaming works in both framings. The model reports itself as `motherbrain`.
Set `--api-key` and it accepts either `X-API-Key` or `Authorization: Bearer`,
which is what OpenAI-compatible clients send.

**Continue (VS Code, JetBrains)** — `~/.continue/config.json`:

```json
{ "models": [{ "title": "MotherBrain", "provider": "openai",
               "model": "motherbrain", "apiBase": "http://host:8000/v1",
               "apiKey": "your-key" }],
  "tabAutocompleteModel": { "title": "MotherBrain", "provider": "openai",
                            "model": "motherbrain",
                            "apiBase": "http://host:8000/v1" } }
```

**Zed** — `settings.json`:

```json
{ "language_models": { "openai": { "api_url": "http://host:8000/v1",
                                   "available_models": [
                                     { "name": "motherbrain",
                                       "max_tokens": 2048 }] } } }
```

**Cline / Roo / Kilo (VS Code)** — provider *OpenAI Compatible*,
base URL `http://host:8000/v1`, model `motherbrain`.

**Anything that supports Ollama** (JetBrains AI Assistant, Obsidian, many
others) — point the Ollama host at `http://host:8000`; `/api/tags` advertises
`motherbrain:latest`.

**Neovim / Emacs** — any OpenAI-compatible plugin (avante, gp.nvim, gptel):
set the base URL to `http://host:8000/v1`.

**aider, llm, the openai SDK:**

```bash
aider --openai-api-base http://host:8000/v1 --openai-api-key x --model motherbrain
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://host:8000/v1", api_key="your-key")
print(client.chat.completions.create(model="motherbrain",
      messages=[{"role": "user", "content": "hello"}]).choices[0].message.content)
```

Editors cancel in-flight completions on nearly every keystroke, so the server
never holds a lock across a response; abandoned streams cannot wedge it, and
concurrent requests are served in parallel.

## Running it on Termux (Android)

PyTorch publishes no Termux-native wheels, so install into a proot distro,
where the ordinary Linux aarch64 wheels work:

```bash
pkg install proot-distro
proot-distro install debian
proot-distro login debian

apt update && apt install -y python3 python3-pip git
git clone <this-repo> && cd MotherBrain
pip install -r requirements.txt
python -m motherbrain.cli serve --host 0.0.0.0 --port 8000
```

Then reach it at `http://localhost:8000` from the phone's browser, or from
anything on the same network. Training on a phone is realistic only for the
`micro` preset; a phone is a fine place to *serve* and *feed*, and a poor place
to train.

If that is more trouble than it is worth, the simpler answer is that MotherBrain
is already a network service: run `mb serve` on a real machine and use Termux
purely as a client (`curl`, or any OpenAI-compatible app).

## Learning automatically, through CI/CD

`.github/workflows/apply-information.yml` closes the loop. Commit a file to
`corpus/inbox/`, or paste text into the Actions **Run workflow** box, and the
pipeline feeds it, trains the patch, mints the next version, commits the patch
plus `versions.json`, and moves the source file to `corpus/learned/`.

Because patches are small, **the model's version history is the git history** —
one commit per version, each with the information that produced it. Runs are
serialised by a concurrency group, so versions stay strictly sequential.

The base checkpoint is cached rather than committed; the first run trains it.

## Scale

MotherBrain is a mixture-of-experts model, which is what lets the parameter
count grow without per-token compute growing with it. Total parameters scale
with the expert count; only `n_experts_per_token` of them run for any token.

```
$ python -m motherbrain.cli scale --preset mother
  total parameters     1157T  (1,156,907,161,374,720)
  active per token     6.93T  (0.599% of total)
```

| preset | total | active/token | note |
|---|---|---|---|
| `micro` | 5.5M | 5.5M | trains on a laptop in minutes |
| `small` | 32M | 32M | |
| `small-moe` | 170M | 67M | |
| `medium` | 2.4B | 491M | one good GPU |
| `large` | 147B | 16.8B | multi-GPU |
| `titan` | 5.6T | 209B | cluster |
| `leviathan` | 137T | 2.7T | |
| `mother` | 1157T | 6.9T | larger than any model ever trained |

`mb scale` prints the arithmetic honestly, including what it would actually
take:

```
  weights+optimizer need ~16,196,700 GB, so ~202,459 80GB GPUs just to hold it.
  a compute-optimal run is ~138.6T tokens, ~4,001,691,400 GPU-hours.
  on 202,459 GPUs at full utilisation that is ~2.3 years of wall-clock training.
```

So: the architecture and the accounting are real and the counts are exact —
`tests/` verifies the analytic formula against actually-instantiated models.
Training the largest presets is a datacenter procurement problem, not a
software one. Everything up to `medium` trains on hardware you have.

## Architecture

Decoder-only transformer, in the shape current frontier models use:

* RMSNorm, pre-norm residual blocks
* Rotary position embeddings (RoPE)
* Grouped-query attention (`n_kv_heads` ≤ `n_heads`), with a KV cache
* SwiGLU feed-forwards
* Sparse MoE with top-k routing, optional always-on shared experts, and a
  load-balancing plus router z-loss so routing does not collapse
* Byte-level BPE tokenizer trained on your corpus — no unknown tokens, ever
* Weight tying, GPT-2 style depth-scaled initialisation

## Layout

```
motherbrain/
  config.py      shapes, presets, exact parameter accounting, feasibility
  tokenizer.py   byte-level BPE, trained incrementally on your corpus
  model.py       the transformer
  data.py        ingestion, corpus on disk, memory-mapped token streams
  train.py       training loop, checkpoints, resume
  patches.py     LoRA patches, replay, sequential versions
  server.py      HTTP API, auto-patcher, web UI
  api_compat.py  OpenAI and Ollama protocol surfaces
  cli.py         the mb command line
tests/           39 tests
.github/workflows/apply-information.yml   the automatic learning pipeline
```

## Honest limits

* The checkpoint in this repository is a `micro` model: 5.5M parameters, 400
  steps, 4.8MB of Python source. It has learned the shape of the language and
  the facts it was patched with; it is not a chatbot and will not answer
  general questions well. Scale and corpus are the only cure, and both are
  yours to choose.
* Chat formatting uses `<user>`/`<assistant>` markers. A base model trained on
  raw text follows them loosely; feed it transcripts in that shape and it
  learns to.
* `/v1/completions` accepts `suffix` but conditions only on the prefix — real
  fill-in-the-middle needs FIM training, which is not implemented.
