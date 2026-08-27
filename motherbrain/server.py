"""HTTP API and web UI for MotherBrain.

Bind it to 0.0.0.0 and the model answers from anywhere that can reach the host:

    POST /generate      prompt in, text out (set stream=true for token-by-token)
    POST /feed          add information to the corpus
    POST /train         kick off a training run in the background
    GET  /status        model, corpus and training state
    GET  /health        liveness
    GET  /              a browser UI that uses all of the above
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import torch
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from motherbrain.api_compat import register_compat_routes
from motherbrain.config import human
from motherbrain.security import (RateLimiter, constant_time_eq,
                                  install_middleware, safe_resolve)


# ---- request models -------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(40, ge=0)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=3.0)
    stream: bool = False


class FeedRequest(BaseModel):
    text: str | None = None
    path: str | None = None
    source: str = "api"


class PatchRequest(BaseModel):
    steps: int = Field(100, ge=1, le=100_000)
    rank: int = Field(8, ge=1, le=256)
    replay_ratio: float = Field(0.25, ge=0.0, le=0.95)
    note: str = ""


class CheckoutRequest(BaseModel):
    version: int = Field(0, ge=0)


class TrainRequest(BaseModel):
    steps: int = Field(200, ge=1, le=1_000_000)
    preset: str = "micro"
    batch_size: int = Field(8, ge=1, le=512)
    lr: float = 3e-4
    seq_len: int | None = None
    resume: bool = True
    retokenize: bool = False
    vocab_size: int | None = None


# ---- the app --------------------------------------------------------------


class State:
    """Everything the server holds: the model, the corpus, the training job."""

    def __init__(self, run_dir: str, corpus_dir: str, device: str,
                 auto_patch: bool = True, auto_patch_chars: int = 2000,
                 auto_patch_delay: float = 20.0) -> None:
        self.run_dir = run_dir
        self.corpus_dir = corpus_dir
        self.device_pref = device
        self.model = None
        self.tokenizer = None
        self.device = None
        self.version = 0
        self.meta: dict = {}
        self.lock = threading.Lock()
        self.training: dict[str, Any] = {"active": False}
        # Automatic learning: fed information becomes a patch on its own.
        self.auto_patch = auto_patch
        self.auto_patch_chars = auto_patch_chars
        self.auto_patch_delay = auto_patch_delay
        self.pending_chars = 0
        self.last_feed = 0.0
        self.patching: dict[str, Any] = {"active": False}
        self.load()

    def load(self) -> bool:
        """(Re)load the current version: base checkpoint plus applied patches."""
        from motherbrain.patches import build_version
        from motherbrain.train import pick_device

        try:
            model, tok, version = build_version(self.run_dir, device=self.device_pref)
        except FileNotFoundError:
            return False
        with self.lock:
            self.model, self.tokenizer = model, tok
            self.device = pick_device(self.device_pref)
            self.version = version
            try:
                self.meta = torch.load(Path(self.run_dir) / "checkpoint.pt",
                                       map_location="cpu", weights_only=False)
                self.meta.pop("model", None)
                self.meta.pop("optimizer", None)
            except Exception:
                self.meta = {}
        return True

    def snapshot(self):
        """An atomic (model, tokenizer, device, version) view.

        Generation must never hold a lock for the length of a response: an IDE
        cancels in-flight completions on almost every keystroke, and a stream
        abandoned mid-flight would never release one. `load()` rebinds these
        attributes to new objects rather than mutating them, so a request that
        grabbed the previous version simply finishes against it.
        """
        with self.lock:
            if self.model is None:
                raise HTTPException(
                    503, "no trained model yet - POST /feed some text, then POST /train")
            return self.model, self.tokenizer, self.device, self.version

    @property
    def ready(self) -> bool:
        return self.model is not None

    def require_model(self):
        if not self.ready:
            raise HTTPException(
                503,
                "no trained model yet — POST /feed some text, then POST /train",
            )
        return self.model


def create_app(run_dir: str = "runs/default", corpus_dir: str = "data/corpus",
               device: str = "auto", api_key: str | None = None,
               auto_patch: bool = True, auto_patch_chars: int = 2000,
               auto_patch_delay: float = 20.0,
               allow_paths: list[str] | None = None,
               allow_origins: list[str] | None = None,
               rate_limit: int = 120, max_feed_chars: int = 2_000_000) -> FastAPI:
    state = State(run_dir, corpus_dir, device, auto_patch=auto_patch,
                  auto_patch_chars=auto_patch_chars, auto_patch_delay=auto_patch_delay)
    app = FastAPI(title="MotherBrain", version="0.1.0",
                  description="A trainable language model, reachable over HTTP.")
    # Editors and their webviews call from arbitrary origins. Credentials are
    # never accepted cross-origin, so the API key cannot ride along implicitly.
    app.add_middleware(
        CORSMiddleware, allow_origins=allow_origins or ["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    install_middleware(app, RateLimiter(rate_limit) if rate_limit else None)

    # Paths that /feed may read. Empty means path ingestion is refused entirely.
    feed_roots = [Path(p).expanduser().resolve() for p in (allow_paths or [])]

    def auth(x_api_key: str | None = Header(default=None),
             authorization: str | None = Header(default=None)) -> None:
        """Optional shared-secret gate. Set MB_API_KEY before exposing publicly.

        Accepts either `X-API-Key: <key>` or the `Authorization: Bearer <key>`
        header that OpenAI-compatible IDE clients send, so the same key works
        whichever protocol the editor speaks.
        """
        if not api_key:
            return
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        if not (constant_time_eq(api_key, x_api_key)
                or constant_time_eq(api_key, bearer)):
            raise HTTPException(401, "invalid or missing API key")

    # ---- introspection ----------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model_loaded": state.ready}

    @app.get("/status")
    def status(_: None = Depends(auth)) -> dict:
        from motherbrain.data import Corpus
        from motherbrain.patches import PatchStore

        corpus = Corpus(state.corpus_dir)
        out: dict[str, Any] = {
            "model_loaded": state.ready,
            "corpus": {
                "documents": corpus.n_documents,
                "chars": corpus.n_chars,
                "tokens": corpus.n_tokens,
                "path": str(corpus.root),
            },
            "training": state.training,
            "patching": state.patching,
            "version": state.version,
            "pending_documents": max(
                0, Corpus(state.corpus_dir).n_documents - PatchStore(state.run_dir).consumed_docs()),
        }
        if state.ready:
            cfg = state.model.cfg
            out["model"] = {
                "version": state.version,
                "preset": cfg.name,
                "total_params": state.model.n_params(),
                "total_params_human": human(state.model.n_params()),
                "active_params": cfg.n_active_params,
                "active_params_human": human(cfg.n_active_params),
                "layers": cfg.n_layers,
                "d_model": cfg.d_model,
                "experts": cfg.n_experts,
                "context": cfg.max_seq_len,
                "vocab_size": cfg.vocab_size,
                "trained_steps": state.meta.get("step", 0),
                "device": str(state.device),
            }
            hist = state.meta.get("history") or []
            if hist:
                out["model"]["latest_eval"] = hist[-1]
        return out

    # ---- generation -------------------------------------------------------

    def _sample(req: GenerateRequest):
        import torch
        from motherbrain.tokenizer import EOS_ID

        model, tok, device, _ = state.snapshot()
        ids = torch.tensor([tok.encode(req.prompt, bos=True)], device=device)
        if ids.shape[1] >= model.cfg.max_seq_len:
            ids = ids[:, -(model.cfg.max_seq_len - 1):]
        return model.generate(
            ids, max_new_tokens=req.max_tokens, temperature=req.temperature,
            top_k=req.top_k or None, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty, eos_id=EOS_ID,
        )

    @app.post("/generate")
    def generate(req: GenerateRequest, _: None = Depends(auth)):
        if state.training.get("active"):
            raise HTTPException(409, "a training run is in progress; try again shortly")

        if not req.stream:
            t0 = time.time()
            ids = list(_sample(req))
            text = state.tokenizer.decode(ids)
            return {
                "text": text,
                "tokens": len(ids),
                "elapsed_sec": round(time.time() - t0, 3),
                "tokens_per_sec": round(len(ids) / max(time.time() - t0, 1e-9), 1),
            }

        def event_stream():
            # Server-sent events, one JSON object per token.
            for token in _sample(req):
                piece = state.tokenizer.decode([token])
                yield f"data: {json.dumps({'token': piece})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- feeding ----------------------------------------------------------

    @app.post("/feed")
    def feed(req: FeedRequest, _: None = Depends(auth)) -> dict:
        """Add information to the corpus. It takes effect at the next /train."""
        from motherbrain.data import Corpus

        corpus = Corpus(state.corpus_dir)
        added_files = added_chars = 0
        if req.text:
            if len(req.text) > max_feed_chars:
                raise HTTPException(
                    413, f"text exceeds the {max_feed_chars:,} character limit")
            added_chars += corpus.add_text(req.text, source=req.source)
            added_files += 1
        if req.path:
            # Confined to an explicit allowlist: without this, /feed is an
            # arbitrary-file-read primitive, and whatever it reads can be
            # extracted again through generation.
            target = safe_resolve(req.path, feed_roots)
            f, c = corpus.add_path(target)
            added_files += f
            added_chars += c
        if not added_files:
            raise HTTPException(400, "send `text`, `path`, or both")
        corpus.write_meta()
        state.pending_chars += added_chars
        state.last_feed = time.time()
        auto = state.auto_patch and state.ready
        return {
            "added_documents": added_files,
            "added_chars": added_chars,
            "corpus_documents": corpus.n_documents,
            "corpus_chars": corpus.n_chars,
            "version": state.version,
            "auto_patch": auto,
            "note": (
                "a patch will train automatically and become the next version"
                if auto else
                "POST /train for a base model, or POST /patch to learn this now"
            ),
        }

    def run_patch(steps: int = 100, rank: int = 8, note: str = "",
                  replay: float = 0.25) -> None:
        """Train pending information into the next version. Runs on a thread."""
        from motherbrain.patches import PatchConfig, create_patch

        state.patching = {"active": True, "step": 0, "total": steps,
                          "started_at": time.time()}
        try:
            cfg = PatchConfig(rank=rank, steps=steps, replay_ratio=replay)
            version = create_patch(
                state.run_dir, state.corpus_dir, cfg, note=note,
                device=state.device_pref,
                progress_cb=lambda info: state.patching.update(info),
            )
            if version is None:
                state.patching = {"active": False, "note": "nothing new to learn"}
                return
            state.load()  # serve the new version from here on
            state.pending_chars = 0
            state.patching = {
                "active": False, "version": version.version,
                "parent": version.parent, "patch_id": version.patch_id,
                "documents": version.n_documents, "tokens": version.n_tokens,
                "loss_before": version.loss_before, "loss_after": version.loss_after,
                "finished_at": time.time(),
            }
        except Exception as exc:
            state.patching = {"active": False, "error": f"{type(exc).__name__}: {exc}"}

    def auto_patch_worker() -> None:
        """Watch for fed information and learn it once the feeding settles.

        Waiting for a quiet gap means a burst of twenty files becomes one
        version rather than twenty, while a single paste still lands promptly.
        """
        while True:
            time.sleep(2.0)
            if not state.auto_patch or state.patching.get("active") \
                    or state.training.get("active") or not state.ready:
                continue
            if state.pending_chars <= 0:
                continue
            quiet = time.time() - state.last_feed
            enough = state.pending_chars >= state.auto_patch_chars
            if quiet >= state.auto_patch_delay or (enough and quiet >= 3.0):
                run_patch(note="auto")

    if auto_patch:
        threading.Thread(target=auto_patch_worker, daemon=True).start()

    # ---- versions ---------------------------------------------------------

    @app.get("/versions")
    def versions(_: None = Depends(auth)) -> dict:
        from dataclasses import asdict

        from motherbrain.patches import PatchStore

        store = PatchStore(state.run_dir)
        return {
            "current": store.current,
            "head": store.head,
            "versions": [asdict(v) for v in store.versions()],
        }

    @app.post("/patch")
    def patch_now(req: PatchRequest, _: None = Depends(auth)) -> dict:
        """Force a patch immediately instead of waiting for the auto-patcher."""
        from motherbrain.data import Corpus
        from motherbrain.patches import PatchStore

        if state.patching.get("active"):
            raise HTTPException(409, "a patch is already training")
        if not state.ready:
            raise HTTPException(503, "no base model yet — POST /train first")
        pending = Corpus(state.corpus_dir).n_documents - PatchStore(state.run_dir).consumed_docs()
        if pending <= 0:
            return {"started": False, "note": "nothing new to learn",
                    "version": state.version}
        threading.Thread(
            target=run_patch,
            kwargs={"steps": req.steps, "rank": req.rank, "note": req.note,
                    "replay": req.replay_ratio},
            daemon=True,
        ).start()
        return {"started": True, "from_version": state.version,
                "pending_documents": pending, "note": "poll GET /status"}

    @app.post("/checkout")
    def checkout(req: CheckoutRequest, _: None = Depends(auth)) -> dict:
        from motherbrain.patches import PatchStore

        store = PatchStore(state.run_dir)
        try:
            store.set_current(req.version)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        state.load()
        return {"version": state.version}

    # ---- training ---------------------------------------------------------

    @app.post("/train")
    def train_endpoint(req: TrainRequest, _: None = Depends(auth)) -> dict:
        from motherbrain.config import PRESETS, ModelConfig
        from motherbrain.data import Corpus
        from motherbrain.train import TrainConfig, train

        if state.training.get("active"):
            raise HTTPException(409, "a training run is already in progress")
        corpus = Corpus(state.corpus_dir)
        if corpus.n_documents == 0:
            raise HTTPException(400, "corpus is empty — POST /feed first")
        if req.preset not in PRESETS:
            raise HTTPException(400, f"unknown preset {req.preset!r}")

        def run() -> None:
            state.training = {"active": True, "step": 0, "total": req.steps,
                              "loss": None, "started_at": time.time()}
            try:
                if req.retokenize or corpus.n_tokens == 0:
                    vocab = req.vocab_size or PRESETS[req.preset].vocab_size
                    corpus.prepare(vocab_size=vocab, verbose=False)
                cfg = ModelConfig.from_dict(PRESETS[req.preset].to_dict())
                cfg.vocab_size = corpus.load_tokenizer().vocab_size
                if req.seq_len:
                    cfg.max_seq_len = req.seq_len
                tc = TrainConfig(steps=req.steps, batch_size=req.batch_size,
                                 lr=req.lr, seq_len=req.seq_len,
                                 device=state.device_pref)

                def progress(info: dict) -> None:
                    state.training.update(info)

                summary = train(state.corpus_dir, state.run_dir, cfg, tc,
                                resume=req.resume, progress_cb=progress)
                state.training = {"active": False, "step": req.steps,
                                  "total": req.steps,
                                  "final_loss": summary["final_loss"],
                                  "finished_at": time.time()}
                state.load()
            except Exception as exc:  # surfaced through /status
                state.training = {"active": False, "error": f"{type(exc).__name__}: {exc}"}

        threading.Thread(target=run, daemon=True).start()
        return {"started": True, "steps": req.steps, "preset": req.preset,
                "note": "poll GET /status for progress"}

    @app.post("/reload")
    def reload_model(_: None = Depends(auth)) -> dict:
        return {"reloaded": state.load(), "model_loaded": state.ready}

    # ---- browser UI -------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI_HTML

    # Everything an IDE needs: /v1/* (OpenAI) and /api/* (Ollama).
    register_compat_routes(app, state, auth)
    return app


UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MotherBrain</title>
<style>
  :root { color-scheme: dark; --bg:#0b0d10; --panel:#14181d; --line:#242b33;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { margin:0; font-size:19px; letter-spacing:.3px; }
  #stats { color:var(--muted); font-size:13px; }
  main { max-width:900px; margin:0 auto; padding:24px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:14px; }
  textarea, input, select { background:var(--panel); color:var(--text);
      border:1px solid var(--line); border-radius:8px; padding:10px 12px;
      font:inherit; width:100%; }
  textarea { min-height:90px; resize:vertical; }
  button { background:var(--accent); color:#04121f; border:0; border-radius:8px;
           padding:10px 18px; font-weight:600; cursor:pointer; }
  button.ghost { background:transparent; color:var(--accent);
                 border:1px solid var(--line); }
  button:disabled { opacity:.5; cursor:default; }
  #out { background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:16px; min-height:150px; white-space:pre-wrap; font-family:
         ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13.5px; }
  label { font-size:13px; color:var(--muted); display:flex; gap:6px;
          align-items:center; }
  label input { width:80px; }
  .tabs { display:flex; gap:6px; margin-bottom:16px; }
  .tabs button { background:transparent; border:1px solid var(--line);
                 color:var(--muted); font-weight:500; }
  .tabs button.on { color:var(--text); border-color:var(--accent); }
  section { display:none; } section.on { display:block; }
  .note { color:var(--muted); font-size:13px; margin-top:10px; }
</style>
</head>
<body>
<header>
  <h1>MotherBrain</h1>
  <span id="stats">connecting…</span>
</header>
<main>
  <div class="tabs">
    <button class="on" data-tab="gen">Generate</button>
    <button data-tab="feed">Feed</button>
    <button data-tab="train">Train</button>
  </div>

  <section id="gen" class="on">
    <div class="row"><textarea id="prompt" placeholder="Write a prompt…"></textarea></div>
    <div class="row">
      <button id="go">Generate</button>
      <label>tokens <input id="maxtok" type="number" value="200" min="1" max="4096"></label>
      <label>temp <input id="temp" type="number" value="0.8" step="0.1" min="0" max="5"></label>
      <label>top-k <input id="topk" type="number" value="40" min="0"></label>
    </div>
    <div id="out"></div>
  </section>

  <section id="feed">
    <div class="row"><textarea id="feedtext"
        placeholder="Paste anything you want the model to learn from…"></textarea></div>
    <div class="row"><input id="feedpath" placeholder="…or a path on the server (file or directory)"></div>
    <div class="row"><button id="feedgo">Add to corpus</button></div>
    <div class="note">Feeding stores text. Training is what makes the model absorb it.</div>
    <div id="feedout" class="note"></div>
  </section>

  <section id="train">
    <div class="row">
      <label>steps <input id="steps" type="number" value="200" min="1"></label>
      <label>batch <input id="batch" type="number" value="8" min="1"></label>
      <label>preset <input id="preset" value="micro"></label>
      <label><input id="retok" type="checkbox" style="width:auto"> rebuild vocabulary</label>
    </div>
    <div class="row"><button id="traingo">Start training</button>
      <button class="ghost" id="refresh">Refresh status</button></div>
    <div id="trainout" class="note"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get('key');
const headers = () => KEY ? {'Content-Type':'application/json','X-API-Key':KEY}
                          : {'Content-Type':'application/json'};

document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); $(b.dataset.tab).classList.add('on');
});

async function status() {
  try {
    const r = await fetch('/status', {headers: headers()});
    const s = await r.json();
    if (!s.model_loaded) {
      $('stats').textContent =
        `no model yet · corpus ${s.corpus.documents} docs, ${s.corpus.chars.toLocaleString()} chars`;
    } else {
      const m = s.model;
      $('stats').textContent =
        `${m.total_params_human} params (${m.active_params_human} active/token) · ` +
        `${m.layers}L · ${m.experts||0} experts · step ${m.trained_steps} · ${m.device}`;
    }
    if (s.training && s.training.active) {
      $('trainout').textContent =
        `training: step ${s.training.step||0}/${s.training.total} ` +
        `loss ${(s.training.loss||0).toFixed(4)}`;
      setTimeout(status, 1500);
    } else if (s.training && s.training.error) {
      $('trainout').textContent = 'error: ' + s.training.error;
    }
  } catch (e) { $('stats').textContent = 'offline'; }
}

$('go').onclick = async () => {
  const btn = $('go'); btn.disabled = true; $('out').textContent = '';
  const body = JSON.stringify({
    prompt: $('prompt').value, max_tokens: +$('maxtok').value,
    temperature: +$('temp').value, top_k: +$('topk').value, stream: true });
  try {
    const r = await fetch('/generate', {method:'POST', headers: headers(), body});
    if (!r.ok) { $('out').textContent = 'error: ' + (await r.text()); btn.disabled=false; return; }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split('\\n\\n'); buf = parts.pop();
      for (const p of parts) {
        const line = p.replace(/^data: /, '');
        if (line === '[DONE]') continue;
        try { $('out').textContent += JSON.parse(line).token; } catch {}
      }
    }
  } catch (e) { $('out').textContent = 'error: ' + e; }
  btn.disabled = false;
};

$('feedgo').onclick = async () => {
  const body = JSON.stringify({text: $('feedtext').value || null,
                               path: $('feedpath').value || null, source: 'web'});
  const r = await fetch('/feed', {method:'POST', headers: headers(), body});
  const j = await r.json();
  $('feedout').textContent = r.ok
    ? `added ${j.added_documents} docs (${j.added_chars.toLocaleString()} chars); ` +
      `corpus now ${j.corpus_documents} docs`
    : 'error: ' + (j.detail || JSON.stringify(j));
  if (r.ok) { $('feedtext').value=''; $('feedpath').value=''; }
  status();
};

$('traingo').onclick = async () => {
  const body = JSON.stringify({steps:+$('steps').value, batch_size:+$('batch').value,
                               preset:$('preset').value, retokenize:$('retok').checked});
  const r = await fetch('/train', {method:'POST', headers: headers(), body});
  const j = await r.json();
  $('trainout').textContent = r.ok ? 'training started…' : 'error: ' + (j.detail||'');
  status();
};
$('refresh').onclick = status;
status();
</script>
</body>
</html>
"""
