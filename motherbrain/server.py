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


class CommandRequest(BaseModel):
    text: str = ""
    max_tokens: int = Field(120, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(40, ge=0)


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
                  replay: float = 0.25, mode: str = "grow",
                  grow_experts: int = 1) -> None:
        """Train pending information into the next version. Runs on a thread."""
        from motherbrain.patches import PatchConfig, create_patch

        state.patching = {"active": True, "step": 0, "total": steps,
                          "started_at": time.time()}
        try:
            cfg = PatchConfig(mode=mode, grow_experts=grow_experts, rank=rank,
                              steps=steps, replay_ratio=replay)
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
                "mode": version.mode,
                "params_before": version.params_before,
                "params_after": version.params_after,
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

    # ---- console ----------------------------------------------------------

    @app.post("/command")
    def command(req: CommandRequest, _: None = Depends(auth)) -> dict:
        """Execute one console instruction.

        Parsing is a fixed table (motherbrain/commands.py), not the model. The
        model is a base language model over source code: it completes prompts,
        it does not follow instructions, and this endpoint does not pretend
        otherwise. Unrecognised input is treated as a prompt.
        """
        from dataclasses import asdict

        from motherbrain.commands import HELP, LOCAL_ONLY, parse
        from motherbrain.config import PRESETS, ModelConfig
        from motherbrain.data import Corpus
        from motherbrain.patches import PatchStore

        cmd = parse(req.text)
        corpus = Corpus(state.corpus_dir)
        store = PatchStore(state.run_dir, create=False)

        if cmd.name in ("noop",):
            return {"kind": "noop", "text": ""}

        if cmd.name == "help":
            return {"kind": "help", "text": HELP}

        if cmd.name == "error":
            return {"kind": "error", "text": cmd.args["message"]}

        if cmd.name in LOCAL_ONLY:
            # /make, /run, /ls and /cat write files and execute code. In a
            # terminal that is no more than the shell already allows. Over
            # HTTP it is remote code execution against whoever is serving the
            # model, so it is refused here rather than guarded - there is no
            # configuration of this that is safe to expose.
            return {"kind": "error",
                    "text": f"/{cmd.name} runs code and touches files, so it "
                            f"works only in the local terminal (mb console), "
                            f"never over the network."}

        if cmd.name == "unknown":
            return {"kind": "error",
                    "text": f"unknown command /{cmd.args['command']} - try /help"}

        if cmd.name == "version":
            return {"kind": "info", "text": f"v{state.version}",
                    "data": {"version": state.version}}

        if cmd.name == "status":
            return {"kind": "status", "data": status(None)}

        if cmd.name == "versions":
            return {"kind": "versions",
                    "data": {"current": store.current,
                             "versions": [asdict(v) for v in store.versions()]}}

        if cmd.name == "checkout":
            try:
                store.set_current(cmd.args["version"])
            except ValueError as exc:
                return {"kind": "error", "text": str(exc)}
            state.load()
            return {"kind": "info", "text": f"now serving v{state.version}"}

        if cmd.name == "scale":
            name = cmd.args.get("preset", "mother")
            if name not in PRESETS:
                return {"kind": "error",
                        "text": f"unknown preset {name}; try {', '.join(PRESETS)}"}
            return {"kind": "info", "text": PRESETS[name].summary()}

        if cmd.name == "learn":
            n = corpus.add_text(cmd.text, source="console")
            corpus.write_meta()
            state.pending_chars += n
            state.last_feed = time.time()
            pending = corpus.n_documents - store.consumed_docs()
            return {"kind": "learned",
                    "text": f"added {n:,} characters; {pending} document(s) "
                            f"waiting to be learned"
                            + (" (auto-patch will do it shortly)"
                               if state.auto_patch and state.ready else
                               " - run /grow to learn them now"),
                    "data": {"pending": pending}}

        if cmd.name == "grow":
            if state.patching.get("active"):
                return {"kind": "error", "text": "already learning; try again shortly"}
            if not state.ready:
                return {"kind": "error", "text": "no model yet - train a base first"}
            pending = corpus.n_documents - store.consumed_docs()
            if pending <= 0:
                return {"kind": "info", "text": "nothing new to learn"}
            threading.Thread(
                target=run_patch,
                kwargs={"mode": "grow", "grow_experts": cmd.args.get("experts", 1),
                        "steps": 100, "note": "console"},
                daemon=True).start()
            return {"kind": "started",
                    "text": f"growing by {cmd.args.get('experts', 1)} expert(s) "
                            f"per layer on {pending} document(s); poll /status"}

        if cmd.name == "train":
            if state.training.get("active"):
                return {"kind": "error", "text": "already training"}
            return {"kind": "started",
                    "text": train_endpoint(
                        TrainRequest(steps=cmd.args.get("steps", 200)), None)["note"]}

        if cmd.name == "export":
            return {"kind": "error",
                    "text": "export runs from the command line: "
                            "mb export --out <path>"}

        # Anything unrecognised is a prompt, which is the one thing the model does.
        ids = list(_sample(GenerateRequest(
            prompt=cmd.text, max_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k,
            repetition_penalty=1.1)))
        _, tok, _, version = state.snapshot()
        return {"kind": "generated", "text": tok.decode(ids),
                "data": {"tokens": len(ids), "version": version}}

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
  :root { color-scheme: dark; --bg:#0b0d10; --panel:#12161b; --line:#242b33;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950;
          --warn:#d29922; --err:#f85149; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin:0; background:var(--bg); color:var(--text); display:flex;
         flex-direction:column;
         font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:baseline; flex-wrap:wrap;
           background:var(--panel); }
  h1 { margin:0; font-size:15px; letter-spacing:.5px; font-weight:600; }
  #stat { color:var(--muted); font-size:12.5px; }
  #stat b { color:var(--text); font-weight:600; }
  main { flex:1; overflow-y:auto; padding:18px; }
  #log { max-width:980px; margin:0 auto; }
  .entry { margin-bottom:14px; white-space:pre-wrap; word-break:break-word; }
  .you { color:var(--accent); }
  .you::before { content:"> "; opacity:.6; }
  .out { color:var(--text); }
  .muted { color:var(--muted); }
  .err { color:var(--err); }
  .ok { color:var(--ok); }
  .warn { color:var(--warn); }
  table { border-collapse:collapse; margin:4px 0; font-size:13px; }
  td { padding:2px 14px 2px 0; vertical-align:top; }
  td.k { color:var(--muted); }
  footer { border-top:1px solid var(--line); background:var(--panel);
           padding:10px 18px; }
  form { max-width:980px; margin:0 auto; display:flex; gap:10px;
         align-items:center; }
  .prompt { color:var(--accent); opacity:.7; }
  input { flex:1; background:transparent; border:0; color:var(--text);
          font:inherit; outline:none; padding:6px 0; }
  button { background:var(--accent); color:#04121f; border:0; border-radius:6px;
           padding:7px 16px; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  .hint { max-width:980px; margin:6px auto 0; color:var(--muted);
          font-size:12px; }
  #ask { position:fixed; inset:0; background:rgba(11,13,16,.97); display:flex;
         flex-direction:column; align-items:center; justify-content:center;
         gap:22px; z-index:10; padding:24px; text-align:center; }
  #ask h2 { margin:0; font-size:17px; font-weight:600; }
  #ask .why { color:var(--muted); font-size:12.5px; max-width:460px;
              line-height:1.7; }
  .choices { display:flex; gap:14px; flex-wrap:wrap; justify-content:center; }
  .choice { background:var(--panel); border:1px solid var(--line);
            border-radius:10px; padding:18px 28px; cursor:pointer; color:var(--text);
            font:inherit; min-width:170px; text-align:left; }
  .choice:hover:not(:disabled) { border-color:var(--accent); }
  .choice:disabled { opacity:.45; cursor:not-allowed; }
  .choice b { display:block; font-size:15px; margin-bottom:4px; }
  .choice span { color:var(--muted); font-size:12px; }
  #mic { background:transparent; border:1px solid var(--line); color:var(--muted);
         border-radius:6px; padding:7px 12px; cursor:pointer; }
  #mic.on { border-color:var(--err); color:var(--err); }
  #mic.hidden { display:none; }
  #programpanel { display:flex; flex-direction:column; gap:10px;
                  width:min(560px,92vw); text-align:left; }
  #programpanel[hidden] { display:none; }
  #programpanel input { background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:8px; padding:10px 12px;
    font:inherit; font-size:13px; width:100%; }
  #feedpanel { display:flex; flex-direction:column; gap:10px; width:min(560px,92vw);
               text-align:left; }
  #feedpanel[hidden] { display:none; }
  #feedpanel textarea, #feedpanel input { background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:8px; padding:10px 12px;
    font:inherit; font-size:13px; width:100%; resize:vertical; }
  .feedrow { display:flex; gap:10px; flex-wrap:wrap; }
  .feedrow[hidden] { display:none; }
  .feedrow .choice { min-width:0; padding:10px 16px; font-size:13px; }
  .choice.go { border-color:var(--accent); color:var(--accent); }
</style>
</head>
<body>
<div id="ask">
  <h2>What would you like to do?</h2>
  <div class="choices">
    <button class="choice" data-mode="program">
      <b>Write a program</b><span>describe it, by text</span></button>
    <button class="choice" id="voice-choice" data-mode="program-voice">
      <b>Write a program</b><span id="voice-note">describe it, by voice</span></button>
    <button class="choice" data-mode="feed">
      <b>Teach it</b><span>add information it can learn</span></button>
    <button class="choice" data-mode="text">
      <b>Console</b><span>prompts and commands, free-form</span></button>
  </div>
  <div class="why" id="voice-why"></div>

  <div id="programpanel" hidden>
    <input id="progwant" placeholder="what should the program do?">
    <div class="why">e.g. read a csv file and print the column averages</div>
    <div class="feedrow">
      <button class="choice go" id="progwrite">write it</button>
      <button class="choice" id="progback">back</button>
    </div>
    <div class="why" id="progout"></div>
  </div>

  <div id="feedpanel" hidden>
    <textarea id="feedtext" rows="6"
      placeholder="Paste what MotherBrain should learn."></textarea>
    <input id="feedpath" placeholder="…or a path on the server (needs --allow-path)">
    <div class="feedrow">
      <button class="choice go" id="feedadd">add to corpus</button>
      <button class="choice" id="feedback">back</button>
    </div>
    <div class="why" id="feedout"></div>
    <div class="feedrow" id="growrow" hidden>
      <button class="choice go" id="feedgrow">learn it now (grows the model)</button>
      <button class="choice" id="feedlater">continue to console</button>
    </div>
  </div>
</div>
<header>
  <h1>MotherBrain</h1>
  <span id="stat">connecting…</span>
  <span id="modelabel" class="muted" style="font-size:12px"></span>
</header>
<main><div id="log"></div></main>
<footer>
  <form id="f">
    <span class="prompt">&gt;</span>
    <input id="in" autocomplete="off" autofocus
           placeholder="type a prompt, or an instruction like: learn that … / grow / status">
    <button id="mic" class="hidden" type="button" title="hold to speak">speak</button>
    <button id="go">send</button>
  </form>
  <div class="hint">/help for commands · ↑ ↓ for history · plain English works
    for the same things</div>
</footer>
<script>
const $ = id => document.getElementById(id);
const KEY = new URLSearchParams(location.search).get('key');
const H = () => KEY ? {'Content-Type':'application/json','X-API-Key':KEY}
                    : {'Content-Type':'application/json'};
const hist = []; let hpos = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
function add(html, cls) {
  const d = document.createElement('div');
  d.className = 'entry ' + (cls || '');
  d.innerHTML = html;
  $('log').appendChild(d);
  document.querySelector('main').scrollTop = 1e9;
  return d;
}
function rows(pairs) {
  return '<table>' + pairs.map(([k, v]) =>
    `<tr><td class="k">${esc(k)}</td><td>${esc(v)}</td></tr>`).join('') + '</table>';
}
function human(n) {
  if (n == null) return '—';
  for (const [lim, s] of [[1e12,'T'],[1e9,'B'],[1e6,'M'],[1e3,'K']])
    if (Math.abs(n) >= lim) return (n/lim).toPrecision(4).replace(/\.?0+$/,'') + s;
  return String(n);
}

async function status(quiet) {
  try {
    const s = await (await fetch('/status', {headers: H()})).json();
    const m = s.model;
    if (!m) {
      $('stat').innerHTML = `no model yet · corpus <b>${s.corpus.documents}</b> docs`;
    } else {
      $('stat').innerHTML =
        `<b>v${m.version}</b> · <b>${m.total_params_human}</b> params` +
        (m.experts ? ` · ${m.experts} experts/layer` : '') +
        ` · ${m.active_params_human} active/token · ${m.device}` +
        (s.pending_documents ? ` · <span style="color:var(--warn)">${s.pending_documents} pending</span>` : '');
    }
    const busy = (s.patching && s.patching.active) || (s.training && s.training.active);
    if (busy) {
      const p = s.patching.active ? s.patching : s.training;
      $('stat').innerHTML += ` · <span style="color:var(--warn)">working: step ${p.step||0}/${p.total||'?'}</span>`;
      setTimeout(() => status(true), 1500);
    } else if (window._wasBusy) {
      window._wasBusy = false;
      if (s.patching && s.patching.version) {
        const p = s.patching;
        add(rows([
          ['learned', `v${p.parent} → v${p.version}`],
          ['grew', p.params_after ? `${human(p.params_before)} → ${human(p.params_after)} parameters` : '—'],
          ['loss', `${p.loss_before} → ${p.loss_after}`],
        ]), 'ok');
      }
    }
    window._wasBusy = busy || window._wasBusy;
    return s;
  } catch (e) { $('stat').textContent = 'offline'; }
}

function render(r) {
  if (r.kind === 'generated') {
    add(esc(r.text), 'out');
    speak(r.text);
    if (r.data) add(`${r.data.tokens} tokens · v${r.data.version}`, 'muted');
  } else if (r.kind === 'status') {
    const d = r.data, m = d.model;
    if (m) speak(`version ${m.version}, ${m.total_params_human} parameters`);
    add(rows(m ? [
      ['version', 'v' + m.version],
      ['parameters', `${m.total_params_human} total · ${m.active_params_human} active/token`],
      ['shape', `${m.layers} layers · d_model ${m.d_model} · ${m.experts || 0} experts/layer`],
      ['context', `${m.context} tokens · vocab ${m.vocab_size}`],
      ['trained', `${m.trained_steps} steps`],
      ['corpus', `${d.corpus.documents} docs · ${d.corpus.tokens.toLocaleString()} tokens`],
      ['pending', `${d.pending_documents} document(s) not yet learned`],
    ] : [['model', 'none trained yet'], ['corpus', `${d.corpus.documents} docs`]]), 'out');
  } else if (r.kind === 'versions') {
    const vs = r.data.versions;
    if (!vs.length) { add('v0 base — no patches yet', 'muted'); return; }
    add('v0  base checkpoint' + (r.data.current === 0 ? '   ← current' : ''), 'out');
    vs.forEach(v => {
      const grew = v.params_after ? `  ${human(v.params_before)}→${human(v.params_after)}` : '';
      add(`v${v.version}  ${v.patch_id}  ${v.n_documents} doc(s)` +
          `  loss ${v.loss_before}→${v.loss_after}${grew}` +
          (v.version === r.data.current ? '   ← current' : ''), 'out');
    });
  } else if (r.kind === 'error') {
    add(esc(r.text), 'err');
    speak(r.text);
  } else if (r.kind === 'started' || r.kind === 'learned') {
    add(esc(r.text), 'ok');
    speak(r.text);
  } else if (r.text) {
    add(esc(r.text), r.kind === 'help' ? 'muted' : 'out');
    if (r.kind !== 'help') speak(r.text);
  }
}

$('f').onsubmit = async ev => {
  ev.preventDefault();
  const text = $('in').value.trim();
  if (!text) return;
  hist.push(text); hpos = hist.length;
  add(esc(text), 'you');
  $('in').value = '';
  $('go').disabled = true;
  try {
    const r = await fetch('/command', {method:'POST', headers:H(),
                                       body: JSON.stringify({text, max_tokens:120})});
    if (!r.ok) { add('error: ' + await r.text(), 'err'); }
    else render(await r.json());
  } catch (e) { add('error: ' + e, 'err'); }
  $('go').disabled = false;
  $('in').focus();
  status(true);
};

$('in').onkeydown = e => {
  if (e.key === 'ArrowUp' && hpos > 0) { $('in').value = hist[--hpos]; e.preventDefault(); }
  if (e.key === 'ArrowDown') {
    hpos = Math.min(hpos + 1, hist.length);
    $('in').value = hist[hpos] || '';
    e.preventDefault();
  }
};

// ---- text or voice -------------------------------------------------------
// Speech uses the browser's own Web Speech API: recognition and synthesis are
// built into Chrome, Edge and Safari, so this needs no dependency and no
// service of ours. Firefox has synthesis but not recognition, which is why the
// two halves are detected separately rather than as one "voice" capability.
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const TTS = window.speechSynthesis;
let mode = 'text', recog = null, listening = false;

(function offerVoice() {
  const btn = $('voice-choice'), note = $('voice-note'), why = $('voice-why');
  if (!SR && !TTS) {
    btn.disabled = true;
    note.textContent = 'not supported by this browser';
    why.textContent = 'This browser exposes neither speech recognition nor '
      + 'speech synthesis. Chrome, Edge and Safari support both.';
  } else if (!SR) {
    note.textContent = 'hear replies (no dictation here)';
    why.textContent = 'This browser can speak but not listen, so voice mode '
      + 'reads replies aloud while you still type. Chrome, Edge and Safari '
      + 'support dictation.';
  } else {
    why.textContent = 'Voice uses your browser\u2019s built-in speech, so audio '
      + 'stays between you and the browser. It will ask for microphone '
      + 'permission. You can switch back to typing at any time.';
  }
})();

function setMode(chosen) {
  mode = chosen;
  $('ask').style.display = 'none';
  $('modelabel').textContent = chosen === 'voice' ? '· voice' : '';
  if (chosen === 'voice' && SR) {
    $('mic').classList.remove('hidden');
    recog = new SR();
    recog.lang = 'en-US';
    recog.interimResults = false;
    recog.maxAlternatives = 1;
    recog.onresult = e => {
      const said = e.results[0][0].transcript;
      $('in').value = said;
      $('f').requestSubmit();
    };
    recog.onerror = e => {
      add('microphone: ' + e.error, 'err');
      stopListening();
    };
    recog.onend = stopListening;
  }
  add(chosen === 'voice'
      ? 'Voice mode. Press speak (or ctrl) and say a prompt or an instruction. '
        + 'Replies are read aloud. Typing still works.'
      : 'Text mode. Type /help for commands, or just start writing and the '
        + 'model will continue it.', 'muted');
  status();
  $('in').focus();
}

function startListening() {
  if (!recog || listening) return;
  listening = true;
  $('mic').classList.add('on');
  $('mic').textContent = 'listening';
  try { recog.start(); } catch (e) { stopListening(); }
}
function stopListening() {
  listening = false;
  $('mic').classList.remove('on');
  $('mic').textContent = 'speak';
}
function speak(text) {
  if (mode !== 'voice' || !TTS || !text) return;
  TTS.cancel();
  // A long completion is not worth reciting in full.
  const u = new SpeechSynthesisUtterance(String(text).slice(0, 600));
  u.rate = 1.05;
  TTS.speak(u);
}

// ---- teaching it something before you start ------------------------------
// Feeding only stores text; a patch is what puts it into the weights. The
// offer to grow immediately is here because the gap between those two is the
// thing people most often miss.
$('feedback').onclick = () => hidePanels();

$('feedadd').onclick = async () => {
  const text = $('feedtext').value.trim();
  const path = $('feedpath').value.trim();
  if (!text && !path) { $('feedout').textContent = 'nothing to add'; return; }
  $('feedadd').disabled = true;
  $('feedout').textContent = 'adding…';
  try {
    const r = await fetch('/feed', {method:'POST', headers:H(),
      body: JSON.stringify({text: text || null, path: path || null, source:'startup'})});
    const j = await r.json();
    if (!r.ok) {
      $('feedout').textContent = 'error: ' + (j.detail || JSON.stringify(j));
    } else {
      $('feedout').textContent =
        `added ${j.added_documents} document(s), ${j.added_chars.toLocaleString()} `
        + `characters. Feeding stores it; learning is what puts it in the weights.`;
      $('feedtext').value = ''; $('feedpath').value = '';
      $('growrow').hidden = false;
    }
  } catch (e) { $('feedout').textContent = 'error: ' + e; }
  $('feedadd').disabled = false;
};

$('feedgrow').onclick = async () => {
  $('feedgrow').disabled = true;
  $('feedout').textContent = 'growing…';
  try {
    const r = await fetch('/command', {method:'POST', headers:H(),
                                       body: JSON.stringify({text:'/grow'})});
    const j = await r.json();
    setMode('text');
    add(j.text || 'growing', j.kind === 'error' ? 'err' : 'ok');
    window._wasBusy = true;          // so the result is reported when it lands
  } catch (e) { $('feedout').textContent = 'error: ' + e; }
};

$('feedlater').onclick = () => setMode('text');

// ---- writing a program ---------------------------------------------------
// The description becomes a docstring and its own words become the function
// name, because a base model cannot be told what to write - it continues from
// context, and words in the signature pull the body towards the same subject.
const Q = '"'.repeat(3);

function showPanel(which) {
  document.querySelector('.choices').hidden = true;
  $('voice-why').hidden = true;
  $(which).hidden = false;
  $(which === 'feedpanel' ? 'feedtext' : 'progwant').focus();
}
function hidePanels() {
  $('feedpanel').hidden = true;
  $('programpanel').hidden = true;
  $('voice-why').hidden = false;
  document.querySelector('.choices').hidden = false;
}
$('progback').onclick = hidePanels;

$('progwrite').onclick = async () => {
  const want = $('progwant').value.trim();
  if (!want) { $('progout').textContent = 'describe it first'; return; }
  const stop = new Set(['a','an','the','that','to','and','of','for','in','it',
                        'with','program']);
  const words = (want.toLowerCase().match(/[a-z]+/g) || []).filter(w => !stop.has(w));
  const slug = words.slice(0, 4).join('_') || 'main';
  const opener = `def ${slug}(`;

  setMode(pendingVoice ? 'voice' : 'text');
  add(esc(want), 'you');
  $('progwrite').disabled = false;
  add('writing…', 'muted');
  try {
    const r = await fetch('/command', {method:'POST', headers:H(),
      // Q is built rather than written literally: a triple quote would end the
      // Python string that carries this page.
      body: JSON.stringify({text: Q + want + Q + '\n\n\n' + opener,
                            max_tokens: 160, temperature: 0.6})});
    const j = await r.json();
    add(opener + (j.text || ''), 'out');
    add(`from a ${$('stat').textContent.match(/[\d.]+M|[\d.]+B/)?.[0] || 'small'} `
        + 'model: plausible Python, not working Python. A model this size '
        + 'reproduces the shape of code and cannot be told what to write. '
        + 'Read it before running it.', 'muted');
    speak('program written');
  } catch (e) { add('error: ' + e, 'err'); }
};

let pendingVoice = false;
document.querySelectorAll('.choice[data-mode]').forEach(b => {
  b.onclick = () => {
    const m = b.dataset.mode;
    if (m === 'feed') return showPanel('feedpanel');
    if (m === 'program' || m === 'program-voice') {
      pendingVoice = (m === 'program-voice');
      return showPanel('programpanel');
    }
    setMode(m);
  };
});
$('mic').onclick = () => listening ? recog.stop() : startListening();
document.addEventListener('keydown', e => {
  if (e.key === 'Control' && mode === 'voice' && !listening) startListening();
});
</script>
</body>
</html>
"""
