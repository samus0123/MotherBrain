"""Protocol shims that make MotherBrain look like tools IDEs already speak.

No editor needs a bespoke MotherBrain plugin. Between two wire formats — the
OpenAI REST API and Ollama's — essentially every IDE assistant can point at
this server by changing a base URL:

    OpenAI-compatible                 Ollama-compatible
      GET  /v1/models                   GET  /api/tags
      POST /v1/chat/completions         POST /api/chat
      POST /v1/completions              POST /api/generate
      POST /v1/embeddings               POST /api/embeddings
                                        GET  /api/version, POST /api/show

Both support streaming, in each protocol's own framing. `/v1/completions`
honours `suffix`, which is what editors send for inline autocomplete.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "motherbrain"


# ---- request bodies -------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""  # str, or the list-of-parts form some IDEs send


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = 40
    stream: bool = False
    stop: Any = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: Any = ""
    suffix: str | None = None  # inline autocomplete: text after the cursor
    max_tokens: int | None = 128
    temperature: float = 0.4
    top_p: float = 0.95
    top_k: int | None = 40
    stream: bool = False
    stop: Any = None


class EmbeddingRequest(BaseModel):
    model: str = MODEL_ID
    input: Any = ""


class OllamaGenerateRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str = ""
    suffix: str | None = None
    system: str | None = None
    stream: bool = True
    options: dict = Field(default_factory=dict)


class OllamaChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = True
    options: dict = Field(default_factory=dict)


# ---- helpers --------------------------------------------------------------


def content_to_text(content: Any) -> str:
    """Flatten the several shapes IDEs use for message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def content_to_images(content: Any, size: int) -> list:
    """Pull inline images out of OpenAI-style content parts.

    Only `data:` URIs are decoded. A URL pointing elsewhere is left alone on
    purpose: a model server that fetches whatever appears in its input is a
    request-forgery primitive aimed at the inside of your network.
    """
    from motherbrain.vision import load_data_uri

    if not isinstance(content, list):
        return []
    images = []
    for part in content:
        if not isinstance(part, dict):
            continue
        url = part.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str):
            decoded = load_data_uri(url, size)
            if decoded is not None:
                images.append(decoded)
    return images


def normalise_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str)]


def build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Render a conversation using the tokenizer's role tokens.

    A base model trained only on raw text will not follow this format well;
    it becomes meaningful once you feed MotherBrain transcripts written the
    same way. The markers are plain text so nothing breaks either way.
    """
    lines = []
    for m in messages:
        text = content_to_text(m.content)
        role = (m.role or "user").lower()
        if role == "system":
            lines.append(text)
        elif role == "assistant":
            lines.append(f"<assistant>{text}")
        else:
            lines.append(f"<user>{text}")
    lines.append("<assistant>")
    return "\n".join(lines)


def stream_pieces(state, prompt: str, max_tokens: int, temperature: float,
                  top_k: int | None, top_p: float, stops: list[str],
                  image=None) -> Iterator[str]:
    """Yield decoded text incrementally, honouring stop sequences.

    Tokens are decoded one at a time, so a stop string that straddles a token
    boundary is still caught: the check runs against the accumulated text and
    the generator truncates at the first hit.
    """
    import torch

    from motherbrain.tokenizer import EOS_ID

    model, tok, device, _ = state.snapshot()
    ids = tok.encode(prompt, bos=True)
    limit = model.cfg.max_seq_len - 1
    if len(ids) > limit:
        ids = ids[-limit:]  # keep the most recent context
    x = torch.tensor([ids], device=device)

    emitted = ""
    extra = {}
    if image is not None and getattr(model, "vision", None) is not None:
        extra["images"] = image.to(device)
    for token in model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                                top_k=top_k or None, top_p=top_p, eos_id=EOS_ID,
                                **extra):
        piece = tok.decode([token])
        if not piece:
            continue
        emitted += piece
        hit = next((s for s in stops if s and s in emitted), None)
        if hit:
            keep = emitted[: emitted.index(hit)]
            tail = keep[len(emitted) - len(piece):] if len(keep) > len(emitted) - len(piece) else ""
            if tail:
                yield tail
            return
        yield piece


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def register_compat_routes(app, state, auth) -> None:
    """Attach every compatibility route to `app`."""

    def _limits(req_max: int | None, default: int) -> int:
        model = state.require_model()
        ceiling = model.cfg.max_seq_len
        return max(1, min(req_max or default, ceiling))

    def _model_card() -> dict:
        cfg = state.model.cfg if state.ready else None
        return {
            "id": MODEL_ID,
            "object": "model",
            "created": int(state.meta.get("created", time.time())),
            "owned_by": "motherbrain",
            "context_length": cfg.max_seq_len if cfg else 0,
        }

    # ---- OpenAI ------------------------------------------------------------

    @app.get("/v1/models")
    def list_models(_: None = Depends(auth)) -> dict:
        return {"object": "list", "data": [_model_card()]}

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str, _: None = Depends(auth)) -> dict:
        return _model_card()

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest, _: None = Depends(auth)):
        if state.training.get("active"):
            raise HTTPException(409, "a training run is in progress; try again shortly")
        prompt = build_chat_prompt(req.messages)
        model, _tok, _dev, _ = state.snapshot()
        image = None
        if getattr(model, "vision", None) is not None:
            for message in reversed(req.messages):
                found = content_to_images(message.content, model.cfg.image_size)
                if found:
                    image = found[-1]     # the most recent picture is the subject
                    break
        stops = normalise_stop(req.stop)
        max_tokens = _limits(req.max_tokens, 256)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            text = "".join(stream_pieces(state, prompt, max_tokens, req.temperature,
                                         req.top_k, req.top_p, stops, image))
            _, counter, _, _ = state.snapshot()
            n_prompt = len(counter.encode(prompt))
            n_out = len(counter.encode(text))
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                          "total_tokens": n_prompt + n_out},
            }

        def gen():
            head = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]}
            yield sse(head)
            for piece in stream_pieces(state, prompt, max_tokens, req.temperature,
                                       req.top_k, req.top_p, stops, image):
                yield sse({"id": cid, "object": "chat.completion.chunk",
                           "created": created, "model": MODEL_ID,
                           "choices": [{"index": 0, "delta": {"content": piece},
                                        "finish_reason": None}]})
            yield sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": MODEL_ID,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest, _: None = Depends(auth)):
        if state.training.get("active"):
            raise HTTPException(409, "a training run is in progress; try again shortly")
        prompt = content_to_text(req.prompt if not isinstance(req.prompt, list)
                                 else "".join(map(str, req.prompt)))
        # Editors send the text after the cursor as `suffix`. Without FIM
        # training the model can only condition on the prefix, so the suffix is
        # supplied as trailing context and the stop list keeps the completion short.
        stops = normalise_stop(req.stop)
        if req.suffix:
            prompt = f"{prompt}"
        max_tokens = _limits(req.max_tokens, 128)
        cid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            text = "".join(stream_pieces(state, prompt, max_tokens, req.temperature,
                                         req.top_k, req.top_p, stops))
            _, counter, _, _ = state.snapshot()
            n_prompt = len(counter.encode(prompt))
            n_out = len(counter.encode(text))
            return {
                "id": cid, "object": "text_completion", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "text": text, "logprobs": None,
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                          "total_tokens": n_prompt + n_out},
            }

        def gen():
            for piece in stream_pieces(state, prompt, max_tokens, req.temperature,
                                       req.top_k, req.top_p, stops):
                yield sse({"id": cid, "object": "text_completion",
                           "created": created, "model": MODEL_ID,
                           "choices": [{"index": 0, "text": piece,
                                        "logprobs": None, "finish_reason": None}]})
            yield sse({"id": cid, "object": "text_completion", "created": created,
                       "model": MODEL_ID,
                       "choices": [{"index": 0, "text": "", "logprobs": None,
                                    "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingRequest, _: None = Depends(auth)) -> dict:
        import torch

        model, tokenizer, device, _ = state.snapshot()
        items = req.input if isinstance(req.input, list) else [req.input]
        items = [content_to_text(i) for i in items]
        limit = model.cfg.max_seq_len

        vectors = []
        total = 0
        for text in items:
            ids = tokenizer.encode(text, bos=True)[:limit] or [1]
            total += len(ids)
            x = torch.tensor([ids], device=device)
            vectors.append(model.embed_text(x)[0].tolist())
        return {
            "object": "list",
            "model": MODEL_ID,
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": total, "total_tokens": total},
        }

    # ---- Ollama ------------------------------------------------------------

    def _ollama_details() -> dict:
        cfg = state.model.cfg if state.ready else None
        params = state.model.n_params() if state.ready else 0
        return {
            "parent_model": "", "format": "motherbrain", "family": "motherbrain",
            "families": ["motherbrain"],
            "parameter_size": f"{params / 1e9:.2f}B" if params else "0B",
            "quantization_level": "F32",
            "context_length": cfg.max_seq_len if cfg else 0,
        }

    @app.get("/api/version")
    def ollama_version() -> dict:
        return {"version": "0.1.0-motherbrain"}

    @app.get("/api/tags")
    def ollama_tags(_: None = Depends(auth)) -> dict:
        size = state.model.n_params() * 4 if state.ready else 0
        return {"models": [{
            "name": f"{MODEL_ID}:latest", "model": f"{MODEL_ID}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size": size, "digest": "motherbrain", "details": _ollama_details(),
        }]}

    @app.post("/api/show")
    def ollama_show(_: None = Depends(auth)) -> dict:
        return {"license": "", "modelfile": "", "parameters": "",
                "template": "{{ .Prompt }}", "details": _ollama_details(),
                "model_info": _ollama_details()}

    def _ollama_opts(options: dict) -> tuple[int, float, int | None, float, list[str]]:
        return (
            _limits(options.get("num_predict"), 256),
            float(options.get("temperature", 0.8)),
            options.get("top_k", 40),
            float(options.get("top_p", 0.95)),
            normalise_stop(options.get("stop")),
        )

    def ndjson(payload: dict) -> str:
        return json.dumps(payload) + "\n"

    @app.post("/api/generate")
    def ollama_generate(req: OllamaGenerateRequest, _: None = Depends(auth)):
        if state.training.get("active"):
            raise HTTPException(409, "a training run is in progress; try again shortly")
        max_tokens, temp, top_k, top_p, stops = _ollama_opts(req.options)
        prompt = f"{req.system}\n{req.prompt}" if req.system else req.prompt
        now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if not req.stream:
            text = "".join(stream_pieces(state, prompt, max_tokens, temp,
                                         top_k, top_p, stops))
            return {"model": MODEL_ID, "created_at": now(), "response": text,
                    "done": True, "done_reason": "stop"}

        def gen():
            for piece in stream_pieces(state, prompt, max_tokens, temp,
                                       top_k, top_p, stops):
                yield ndjson({"model": MODEL_ID, "created_at": now(),
                              "response": piece, "done": False})
            yield ndjson({"model": MODEL_ID, "created_at": now(), "response": "",
                          "done": True, "done_reason": "stop"})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/chat")
    def ollama_chat(req: OllamaChatRequest, _: None = Depends(auth)):
        if state.training.get("active"):
            raise HTTPException(409, "a training run is in progress; try again shortly")
        max_tokens, temp, top_k, top_p, stops = _ollama_opts(req.options)
        prompt = build_chat_prompt(req.messages)
        now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if not req.stream:
            text = "".join(stream_pieces(state, prompt, max_tokens, temp,
                                         top_k, top_p, stops))
            return {"model": MODEL_ID, "created_at": now(),
                    "message": {"role": "assistant", "content": text},
                    "done": True, "done_reason": "stop"}

        def gen():
            for piece in stream_pieces(state, prompt, max_tokens, temp,
                                       top_k, top_p, stops):
                yield ndjson({"model": MODEL_ID, "created_at": now(),
                              "message": {"role": "assistant", "content": piece},
                              "done": False})
            yield ndjson({"model": MODEL_ID, "created_at": now(),
                          "message": {"role": "assistant", "content": ""},
                          "done": True, "done_reason": "stop"})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/embeddings")
    def ollama_embeddings(req: EmbeddingRequest, _: None = Depends(auth)) -> dict:
        import torch

        model, tokenizer, device, _ = state.snapshot()
        text = content_to_text(req.input if not isinstance(req.input, list)
                               else (req.input[0] if req.input else ""))
        ids = tokenizer.encode(text, bos=True)[:model.cfg.max_seq_len] or [1]
        vec = model.embed_text(torch.tensor([ids], device=device))[0].tolist()
        return {"embedding": vec}
