# MotherBrain

A language model you own end to end: a sparse mixture-of-experts transformer,
a tokenizer it learns from your own text, a training loop, an incremental
patching system that versions every new thing it learns, and an HTTP server
that any IDE can talk to.

```
mb console    tell it what to do, interactively
mb status     what is on disk, and what to run next
mb bootstrap  fresh clone -> a loaded model, in one command
mb scale      what a configuration costs, before you build it
mb feed       put information in
mb prepare    learn a vocabulary from it
mb train      train the base model            -> v0
mb patch      learn new information           -> v1, v2, v3 ...
mb versions   the lineage
mb checkout   go back to any earlier version
mb chat       talk to it locally
mb export     write a shareable model file
mb cert       make a TLS certificate
mb serve      expose it to every IDE you own
```

## Install

```bash
scripts/install.sh          # handles the usual install failures
```

or, if your Python is not externally managed:

```bash
pip install -e .
```

Either installs the dependencies **and** the `mb` command, so every example below
works from any directory. Without it you would get `No module named
motherbrain` outside the repository root.

```bash
pip install -e ".[dev]"      # plus pytest and httpx, to run the tests
```

`mb` finds your corpus and checkpoints by walking up from the current directory
for a workspace, the way git finds `.git`, so it works from a subdirectory too.
Override with `--corpus`/`--run`, or the `MB_CORPUS`/`MB_RUN` environment
variables.

### If it will not start

Run the diagnostic and read the first two sections:

```bash
sh scripts/doctor.sh
```

It reports the branch, whether the files exist, the Python version, the
virtual environment, the dependencies, and then tries to start.

**The most common cause is the branch.** `main` holds only a LICENSE and a
README, so a plain `git clone` produces a directory with nothing to run in it:

```bash
git fetch origin
git checkout claude/massive-parameter-llm-mcs613
```

### If `pip install -e .` fails

Three causes account for almost all of it, and none are about this project.

**`error: externally-managed-environment`** — Debian 12+, Ubuntu 23.04+ and
Termux's proot images mark the system Python as managed by the OS (PEP 668), so
pip refuses to install into it. Use a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

`scripts/install.sh` does exactly that, and checks the other two causes before
downloading several hundred megabytes to discover them.

**`No matching distribution found for torch`** — PyTorch publishes wheels only
for certain Python versions and platforms. A very new Python (3.14+) usually
has no wheel yet, and neither does Termux's own Python, which is why the
Android instructions install into a proot distro rather than Termux directly.
Check with `python3 -VV`, and install an older Python if needed:

```bash
apt install python3.12 python3.12-venv
PYTHON=python3.12 scripts/install.sh
```

**`No module named venv`** — `apt install python3-venv`.

**It downloads gigabytes, or runs out of disk.** On Linux x86_64, `pip install
torch` pulls the CUDA runtime — around 2.5GB — even with no GPU present. For a
CPU-only machine, ask for the CPU build instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

That is about a tenth of the size. ARM64 machines, phones included, get a
CPU-only wheel from PyPI anyway and need no flag.

**Running it needs only torch and numpy.** fastapi, uvicorn and pydantic are
imported only by `mb serve`, so a machine that cannot install a web stack can
still run the console:

```bash
pip install torch numpy
python3 -m motherbrain.cli console
```

You do not have to install the package at all. From the repository root,
everything works as a module once the dependencies are present:

```bash
pip install -r requirements.txt      # or: apt install python3-torch python3-numpy
python3 -m motherbrain.cli console
```

That is the same program; `mb` is only a shortcut that works from any
directory.

## Quick start

```bash
mb feed ./my-notes ./src        # anything textual
mb prepare --vocab-size 4096
mb train --preset micro --steps 400 --batch-size 16 --seq-len 256
mb chat --prompt "hello"
mb serve            # 127.0.0.1 by default
```

## The opening menu

```
What would you like to do?

  1  Tell me what to do by text
  2  Tell me what to do by voice
  3  Learn new information
  4  Apply the learned information as a patch (ascend to the next version)

choose [1-4, default 1]
```

The browser console at `/` opens on the same four.

**1 and 2 — tell it what to do**, typed or spoken. Both land at the same
prompt: write text for the model to continue, or run commands like `/make`,
`/run`, `/versions`. Option 2 falls back to text, with the reason, on a machine
that cannot speak or listen.

**3 — Learn new information.** Type, paste, or give a path. This *stores* it.
The model is unchanged, and the console says so rather than leaving the
impression that feeding was enough.

**4 — Apply it as a patch.** This is where text on disk becomes part of the
model: new experts are trained on it, the parameter count grows, and the
version ascends.

```
v1 -> v2
  applied    1 document(s), 20 tokens
  grew       28.32M -> 37.76M (+9,440,264 parameters)
  loss       3.685 -> 0.090 on the new material
  in effect  the model now serving is v2
```

Learning and applying are separate steps because they are separate things.

Applying also **exports the model**, because a grown model otherwise lives only
in `runs/`, which is gitignored and absent from a fresh clone. The export is
what makes an ascent durable:

```
  in effect  the model now serving is v3
  exported   models/motherbrain.pt (88.4 MB)
             commit it to keep v3: git add models/motherbrain.pt && git commit
```

If the export fails it says so and notes that the patch exists only in `runs/`,
rather than leaving the impression it was saved. `--export <path>` chooses a
different destination.

## Text, voice, or teach it something

Both consoles ask how you want to start: **text**, **voice**, or **teach it**
— feed it something before you begin.

Choosing to teach it takes the information first, then offers to learn it
immediately, because the gap between those two is the thing people most often
miss: feeding stores text in the corpus, and a patch is what puts it into the
weights.

```
text, voice, or feed it something? [text] feed
Paste or type what MotherBrain should learn.
A file or directory path works too. Blank line to finish.

  the deploy key rotates on Fridays

added 1 document(s), 33 characters. 1 waiting to be learned.
learn it now? this grows the model [Y/n]
```

Say yes and it grows a version before dropping you at the prompt; say no and it
stays in the corpus until you run `/grow`.

In the browser, voice is real: recognition and synthesis come from the
browser's own Web Speech API, so there is no dependency and no service of ours
in the loop — audio stays between you and the browser, which asks for
microphone permission itself. Press **speak** (or hold ctrl), say a prompt or
an instruction, and replies are read back. Typing keeps working throughout.

The two halves are detected separately, because they are separate: Firefox can
speak but not listen, so it is offered voice-out with dictation disabled rather
than a promise it cannot keep. A browser with neither gets the option greyed
out and told why.

```bash
mb console                 # asks
mb console --mode feed     # straight to feeding
mb console --mode text     # skips the question
mb console --mode voice
```

In the terminal there is no equivalent guarantee — voice needs a microphone, an
audio stack, and software to drive them — so `mb console` detects what is
actually installed (`say`, `espeak-ng`, `pyttsx3`, `SpeechRecognition`) and
falls back to typing with the reason printed:

```
voice is unavailable here: no speech synthesis (install espeak-ng, or
pyttsx3); no speech recognition (pip install SpeechRecognition PyAudio)
using text.
```

None of that is installed as a dependency. Speech is optional, and a model you
can only talk to would be worse than one you can also type at.

## Making it do things

The console does more than print. In the terminal:

```
> /make a script that renames files -> tools/rename.py
────────────────────────────────────────────────────────────
def make_script_renames(
    path: str,
    ...
────────────────────────────────────────────────────────────
written to tools/rename.py
run it? it was written by a 25M model [y/N] y
running tools/rename.py ...
────────────────────────────────────────────────────────────
SyntaxError: unterminated triple-quoted string literal
────────────────────────────────────────────────────────────
exit code 1  (it failed, which is usual for code this model writes)
```

| command | what it does |
|---|---|
| `/make <what> [-> file]` | writes a program, saves it, offers to run it |
| `/run <file>` | runs a python file and shows its output |
| `/write <file> [text]` | writes a file, asking for the text if not given |
| `/find <pattern>` | searches the files here |
| `/sh <command>` | runs a shell command, after confirming |
| `/delete <file>` | deletes a file, after confirming |
| `/ls [dir]` | lists files |
| `/cat <file>` | shows a file |
| `/see <image> [prompt]` | looks at an image |

Choosing option 1 or 2 opens with what you can ask for, rather than a bare
prompt:

```
Tell me what to do. For example:

  make a script that renames files      write code, save it, run it
  write notes.txt                       create a file
  find TODO                             search the files here
  run script.py                         run it and show the output
  list files                            what is here
  sh git status                         any shell command
  <anything else>                       the model continues it
```

Plain English works for all of them — "search for TODO", "remove old.txt",
"list files". Anything the table does not cover is treated as a prompt and
continued by the model.

**Two things are worth being exact about.**

*The model does not choose the action.* Your command does. MotherBrain is a
base model over source code: it cannot follow an instruction, so `/make` uses
it for the one thing it can do — continue text — and everything else is
ordinary Python doing what you asked. A console that claimed the model decided
would be theatre.

*These never work over the network.* `/make`, `/run`, `/ls` and `/cat` write
files and execute code. In your own terminal that is no more than your shell
already allows. Reached over HTTP it would be remote code execution against
whoever is serving the model, so the server refuses them outright rather than
guarding them — there is no configuration of that which is safe to expose.

## Telling it what to do

`mb console`, or the page `mb serve` puts at `/`, is one place to type both
prompts and instructions:

```
> how big are you
  version      v1
  parameters   25.18M total · 15.75M active/token
  shape        8 layers · d_model 384 · 1 experts/layer
  pending      1 document(s) not yet learned

> learn that the deploy key rotates on Fridays
added 44 characters; 1 document(s) waiting. run /grow to learn them.

> grow
v1 -> v2: 25.18M -> 34.62M params, loss 3.402 -> 0.094

> def softmax(x, axis=-1):
    if not isinstance(x, np.ndarray):
        return x.shape[0]
    return x
```

Commands are `/learn`, `/grow`, `/train`, `/versions`, `/version`, `/checkout`,
`/status`, `/scale`, `/export`, `/help`; a fixed set of plain phrasings mean
the same things ("learn that …", "grow yourself", "how big are you", "list
versions", "roll back to 1"). Anything unrecognised is treated as a prompt and
completed by the model.

**The parsing is a lookup table, not the model.** MotherBrain is a base
language model trained on source code: it completes text and cannot follow
instructions. A console that claimed to understand them would be theatre. So
the table is small, literal, and refuses rather than guesses - `/checkout`
without a version is an error, and "learning rates matter" is a prompt, not an
instruction to learn something.

## How do I run it?

Three commands, from a clean clone:

```bash
pip install -e .        # installs deps and the `mb` command
mb bootstrap            # trains a base model if there is not one yet
mb chat                 # talk to it
```

`mb serve` instead of `mb chat` exposes it over HTTP for your IDEs. `mb status`
at any point tells you what state you are in.

A freshly bootstrapped model is barely trained and will emit mostly whitespace
and fragments — `mb chat` frames its output and reports a token count so you
can tell it ran rather than failed, and warns when the step count is too low
to expect anything coherent. `mb train --steps 2000 --resume` is what fixes
that.

## How do I load it?

Run `mb status` — it inspects what is on disk and tells you which situation
you are in and what to run next.

**On a machine that has already trained:**

```bash
mb chat      # loads the current version, interactively
mb serve     # loads it and serves it to your IDEs
```

Both load the *current version*: the base checkpoint with patches 1..N applied.
`mb checkout v1` changes which version that is.

**On a fresh clone, `mb console` and `mb chat` just work.** The clone carries
`models/motherbrain.pt`, and both fall back to it when there is no training
checkpoint — so a clone runs the trained model without training anything.

`mb train`, `mb patch` and `mb grow` do need a checkpoint, because they
continue training rather than only running the model.

**If you want to train your own base,** `checkpoint.pt` holds the
weights and is far too large to commit, so a clone carries the code, the
tokenizer, the manifest and the patches — but no base model. One command fixes
that:

```bash
mb bootstrap        # feed -> prepare -> train -> ready
```

Or download the `motherbrain-base-checkpoint` artifact from a CI run into
`runs/default/`, which gives you the exact base the committed patches belong to.

Loading a lineage against the wrong base is refused rather than silently
applied, and training a new base drops the patches that no longer apply:

```
note: dropped 3 patch(es) trained against the previous base checkpoint:
      v1 (500af452), v2 (859de298), v3 (989d8cdc)
```

## Sharing a trained model

A training checkpoint carries optimizer state, weighs several times what the
weights alone do, and loads through pickle. `mb export` writes the model
instead: fp16 weights with the config and tokenizer embedded as JSON.

```bash
mb export --out models/motherbrain-15m.pt
mb chat --model models/motherbrain-15m.pt      # runs it directly
```

The result is roughly a sixth the size of the checkpoint, self-contained, and
loads under `torch.load(weights_only=True)` — no code executes when you open
one, which is what makes it safe to hand to someone else. This is also how a
trained model survives a machine: checkpoints are gitignored, exports are
small enough to keep.

## How do I feed it new information?

Four ways, all landing in the same corpus and all producing a new version.

**1. The command line.** Files, whole directories, or literal text:

```bash
mb feed ./docs ./notes.md "The deploy key rotates on Fridays."
mb patch --steps 150      # -> v1
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

## Sight

MotherBrain reads images as well as text. The transformer consumes a sequence
of vectors: text becomes vectors by looking words up in a table, and an image
becomes vectors by cutting it into square patches and projecting each one.
After that the model cannot tell them apart — one sequence, two sources.

```bash
mb chat --image photo.png --prompt "the picture shows"
```

or in the console:

```
> /see photo.png what is in it
looking at photo.png (64 patches) ...
```

The encoder is its own small vision transformer rather than extra layers of
the language model, so an image is understood before it is handed over. Its
parameters are counted exactly, like everything else:

```
  sight                48 layers x 4096, 448px in 784 patches (9.754B params)
```

Two properties worth stating:

* **Sight is additive.** With `vision_layers = 0` there is no tower, no extra
  parameters, and the forward pass is exactly what it was. A text-only model is
  untouched by any of this, and a test asserts it.
* **Sight is not free the way experts are.** Only `n_experts_per_token` experts
  run for a token, but *every* parameter of the vision tower runs for every
  image. Growing the experts costs nothing per token; growing the eyes costs
  the full amount.

The `mother` preset now sees at a matching scale — 448px in 784 patches through
a 48-layer tower, 9.75B parameters of sight inside 1157T total. `micro-vision`
is the same idea small enough to train here.

**The honest gap:** the architecture is built and tested, and the committed
model has no vision tower because training one needs image-text pairs, which
this project has none of. `mb chat --image` against a text-only model says so
rather than pretending. Feeding it paired data and training the tower is real
work that has not been done.

## Growing as it learns

By default every patch makes the model **larger**. New experts are appended to
every feed-forward layer, only those are trained, and the parameter count rises
with each version and never falls:

```bash
mb feed "the deploy key rotates on Fridays"
mb patch                       # -> v1, and the model is bigger than it was
```

```
v0 -> v1   patch 4f2a91c0
  learned      1 docs, 48 tokens
  grew         +1 expert(s) per layer, +9,437,184 parameters
  size         15.74M -> 25.17M
```

Two properties make that safe rather than merely impressive:

* **Growth is a no-op at birth.** A new expert's output projection starts at
  zero and its router bias at -1e9, so it cannot be selected and contributes
  nothing. The grown model computes exactly what it did before, bit for bit -
  learning a new fact cannot silently damage what was already known. Training
  is the only thing that makes it diverge.
* **Compute does not grow with it.** Only `n_experts_per_token` experts run for
  any token, so a model that has grown through fifty versions costs the same
  per token as it did at version one. That is precisely why the parameter count
  can be allowed to run away, and it is the same mechanism behind the `mother`
  preset's quadrillion parameters.

A dense model has no experts to append to, so the first patch converts its
feed-forward layers into MoE layers: the existing dense FFN becomes an
always-on shared expert, which preserves its behaviour exactly, and the new
routed experts are added alongside.

`mb patch --mode lora` keeps the old behaviour - a low-rank delta that teaches
the model something new without changing its size - when growth is not wanted.
`--grow N` adds N experts per layer instead of one.

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
mb versions
mb checkout v1     # any earlier version, exactly
```

A LoRA patch is under a megabyte. A growth patch is not: it carries whole new
experts, so against the 15.7M model it is 19MB at half precision and gets
larger as the model does. Patch binaries are therefore kept out of git -
committing one per version is unbounded - while `versions.json`, the record of
what was learned and how much the model grew, is small and does live there.
The distributable artifact is the exported current model in `models/`.

A patch is a delta against *particular* weights, so the manifest records a
fingerprint of the base checkpoint it was built on. Loading a lineage whose
base does not match refuses loudly instead of applying deltas to the wrong
weights, and retraining the base drops the now-meaningless patches and says so.
This matters in practice: the base checkpoint is too large to commit, so a CI
run that rebuilds it from scratch produces different weights than your laptop
did, and without the fingerprint the committed patches would be applied to it
silently.

Three details make this work rather than merely run:

* **Replay.** Training only on new text makes a model forget the old text. Each
  patch trains on a mixture of the new information and a sample of everything
  before it.
* **A frozen, byte-level vocabulary.** New information can use words the base
  corpus never contained, with no vocabulary surgery and no unknown tokens.
* **Base fingerprinting.** Patches are refused unless the base they were
  trained against is the one loaded.

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
mb serve --host 0.0.0.0 --port 8000
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

## Serving over HTTPS

Generate a certificate and serve TLS. Anything fed over a plaintext connection
crosses the network in the clear — including the API key — so use HTTPS for
anything that leaves the machine.

```bash
mb cert --host your-hostname-or-ip
mb serve --host 0.0.0.0 \
  --tls-cert certs/server.crt --tls-key certs/server.key \
  --api-key "$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  --allow-path ./corpus
```

`mb cert` writes a self-signed pair (key mode 600) and prints its SHA-256 so
clients can pin it. A self-signed certificate encrypts the link but does not
prove identity: clients will warn, and `curl` needs `-k` or `--cacert`. For a
public hostname use a real certificate (Let's Encrypt) or put MotherBrain
behind a reverse proxy that terminates TLS.

Point IDEs at `https://host:8443/v1` rather than `http://`.

## Security

There is no such thing as 100% secure, and any component that claims it is
lying to you. What MotherBrain does is close the holes this design actually
has, and fail closed rather than open:

| Control | Behaviour |
|---|---|
| Path ingestion | `/feed` accepts a filesystem path only inside `--allow-path` roots. **Off entirely by default.** |
| Credential files | `.ssh`, `.env`, `.netrc`, `*.key`, `*.pem` and friends are refused even inside an allowed root. |
| Authentication | `--api-key`, compared in constant time, accepted as `X-API-Key` or `Authorization: Bearer`. |
| Public bind | Binding a non-loopback interface **without** a key is refused outright unless you pass `--insecure`. |
| Default bind | `127.0.0.1`. Exposing the server is a deliberate act. |
| Transport | `--tls-cert`/`--tls-key`; a plaintext public bind warns loudly. |
| Rate limiting | Token bucket per client address, `--rate-limit` (default 120/min). |
| Body size | Requests over 8MB rejected with 413; `/feed` text capped separately. |
| Headers | `nosniff`, `DENY` framing, `no-referrer`, `no-store`. |
| CORS | Credentials never accepted cross-origin; restrict with `--allow-origin`. |
| Secrets in git | `certs/`, `*.key`, `*.pem` are gitignored. |

The sharpest edge is the first row, and it is worth being explicit about why.
`/feed` reads files into the corpus; the corpus is training data; training data
can be recovered through generation. An unrestricted path parameter is
therefore not merely a file read — it is a file read whose output can be
extracted later by anyone allowed to prompt the model. That is why path
ingestion is disabled by default and confined to an allowlist when enabled.

What is still on you:

* **Trust the weights you load.** `torch.load` on a base *checkpoint* uses
  Python pickle and can execute code, because checkpoints carry config
  objects. Patches and `mb export` files both load with `weights_only=True`
  and are safe to accept from elsewhere; raw checkpoints are not, so only
  load ones you produced or trust.
* **Anything you feed can come back out.** Do not feed secrets to a model that
  other people may prompt.
* **A self-signed certificate is not identity.** Pin the fingerprint, or use a
  real CA.
* **Auto-patch means untrusted input becomes training data.** With `/feed` open
  to a network, whoever can reach it can steer the model. Keep the API key
  secret, or run with `--no-auto-patch` and patch deliberately.
* Rate limiting is per process and in memory; it is not a substitute for a
  real gateway if you are genuinely exposed to the internet.

## Running it on Windows

Windows 10 and 11 both work; nothing here is version-specific. Install Python
from python.org (tick **Add python.exe to PATH**), then in PowerShell:

```powershell
git clone https://github.com/samus0123/MotherBrain
cd MotherBrain
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

`start.ps1` is the Windows twin of `start.sh`: it fetches the code if missing,
finds a Python (`py -3`, `python`, `python3`), creates `.venv`, installs, and
launches — stopping at whichever step fails and naming it. If it will not
start, `scripts\doctor.ps1` reports the branch, files, Python, virtual
environment and dependencies in one go.

By hand:

```powershell
python -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\mb console
```

`\Scripts\` rather than `/bin/` is the only path difference from Linux.

**Voice works with nothing installed.** Windows ships System.Speech, which
MotherBrain reaches through PowerShell, so option 2 reads replies aloud out of
the box. Dictation still needs `pip install SpeechRecognition PyAudio`; the
browser console at `/` has both through Chrome or Edge and needs neither.

**Use Windows Terminal** rather than the old `cmd.exe` window. The console asks
for UTF-8 so it will not crash on the legacy code page, but box-drawing
characters may render as `?` there.

**PyTorch's default Windows wheel is CPU-only**, about 200MB, so the 2.5GB
CUDA download that Linux x86_64 pulls is not a concern and no `MB_CPU_ONLY`
flag is needed.

**`mb cert` needs openssl**, which Windows does not ship. Either
`winget install ShiningLight.OpenSSL`, or use Git Bash which includes it, or
supply your own certificate. The command says so rather than failing opaquely.
Note also that the private key it writes cannot be locked down with `chmod` on
Windows — its permissions are whatever the folder grants.

**WSL** works too, and there the Linux instructions apply unchanged.

I have no Windows machine, so unlike the Linux path these are not verified end
to end: the encoding fix and the speech detection are covered by tests, and the
PowerShell scripts are written carefully but untested on real Windows.

## Running it on Termux (Android)

PyTorch publishes no Termux-native wheels, so install into a proot distro,
where the ordinary Linux aarch64 wheels work:

```bash
pkg install proot-distro
proot-distro install debian
proot-distro login debian

apt update && apt install -y python3 python3-pip python3-venv git
git clone <this-repo> && cd MotherBrain
scripts/install.sh
.venv/bin/mb console
```

`python3-venv` matters: Debian's proot image marks its Python as externally
managed, so installing without a virtual environment fails with
`externally-managed-environment`. The script creates one.

The clone already contains a trained model, so there is nothing to train
before it will answer:

```bash
.venv/bin/mb chat --model models/motherbrain.pt --prompt "def softmax(x):"
```

For voice, serve it and open the page in Chrome on the phone — Android Chrome
supports the Web Speech API, while the Termux terminal has no speech backends:

```bash
.venv/bin/mb serve --host 0.0.0.0
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
$ mb scale --preset mother
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

`configs/mother.json` is the committed definition of that model. The shape is
not just arithmetic: one attention block at mother's true width instantiates
here as 922,746,880 real parameters and runs a real forward pass, matching the
analytic prediction exactly (`tests/` asserts both). What needs a datacenter is
assembling 160 layers of them alongside 2048 experts apiece.

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

For the practical inverse — the largest model your actual hardware can hold —
ask it directly:

```bash
mb scale --fit-gpus 8              # largest config that fits on 8 x 80GB
mb scale --fit-gpus 1 --gpu-gb 24  # ...or on one 24GB card
```

It grows the expert count until the weights plus optimizer state stop fitting,
steps down to a smaller shape when the requested one cannot fit at all, and
says plainly when nothing fits rather than returning a configuration that does
not.

```
              1 x 24GB   medium-fit-1x24gb        1.321B
              8 x 80GB   large-fit-8x80gb         42.73B
           1024 x 80GB   titan-fit-1024x80gb       5.62T
```

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
  patches.py     patches, replay, sequential versions
  growth.py      adding experts so each version is larger than the last
  server.py      HTTP API, auto-patcher, web UI
  api_compat.py  OpenAI and Ollama protocol surfaces
  security.py    path confinement, auth, rate limiting, exposure checks
  commands.py    parsing what you tell it to do
  voice.py       speech backends for the terminal, and what to say without one
  vision.py      the image encoder, and how a picture becomes tokens
  cli.py         the mb command line
configs/         mother.json, and the model actually being trained
tests/           49 tests
.github/workflows/apply-information.yml   the automatic learning pipeline
```

## The trained model

`models/motherbrain.pt` is a real trained model, committed to this repository.
A clone can run it immediately:

```bash
mb chat --model models/motherbrain.pt --prompt "def softmax(x, axis=-1):"
```

It is v1: a 15.7M-parameter base trained for 4,000 steps (24.6M tokens) on a
53.2M-token corpus of Python source, grown to 25.18M parameters by one growth
patch. 56.8MB of fp16 weights against a 181MB training checkpoint, loaded with
`weights_only=True` so opening it executes no code.

```
step   200   val loss 5.94   perplexity 381
step  1000   val loss 4.48   perplexity  89
step  2000   val loss 3.29   perplexity  27
step  3600   val loss 2.80   perplexity  16.4   <- best
step  4000   val loss 2.82   perplexity  16.7
```

A 23x improvement in validation perplexity, on a held-out split. What it
writes:

```python
    if not isinstance(x, np.ndarray):
        return x.shape[0]
    return x


def _to_int(
```

Syntactically valid, idiomatic Python: a type guard, correct returns, correct
spacing before a new definition. What it does not write is *correct* code, and
at this size it will not. It is a real language model that has learned the
shape of Python from 53M tokens; it is not a coding assistant. Scale and
corpus are the only cure, and `mb train` is how you apply them.

## Honest limits

* The committed model is 25.18M parameters (a 15.7M base plus one growth
  patch), trained on CPU for under one epoch.
  It writes plausible Python structure, not working programs, and it is not a
  chatbot - it will not answer general questions. Scale and corpus are the only
  cure, and both are yours to choose.
* Chat formatting uses `<user>`/`<assistant>` markers. A base model trained on
  raw text follows them loosely; feed it transcripts in that shape and it
  learns to.
* `/v1/completions` accepts `suffix` but conditions only on the prefix — real
  fill-in-the-middle needs FIM training, which is not implemented.
