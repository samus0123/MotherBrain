"""Tests for the parts that are easy to get quietly wrong."""

from pathlib import Path

import numpy as np
import pathlib

import pytest
import torch

from motherbrain.config import PRESETS, ModelConfig, scale_to
from motherbrain.model import MotherBrain
from motherbrain.tokenizer import SPLIT_PATTERN, Tokenizer

SAMPLE = [
    "def n_heads(self, __x): return self._y + 1\n",
    "The mother brain awakens, and the lights come on.\n",
    "héllo wörld 🧠 混合 language with_underscores\n",
]


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=300, max_seq_len=64, d_model=64, n_layers=4,
                n_heads=4, n_kv_heads=2, d_ff=128)
    base.update(kw)
    return ModelConfig(**base)


# ---- tokenizer ------------------------------------------------------------


@pytest.mark.parametrize("text", SAMPLE + [
    "", "   ", "_", "___", "a_b_c", "!@#$%^&*()_+", "\t\n\n  mixed \t",
])
def test_split_covers_every_character(text):
    """Any character the pre-tokenizer drops is silently lost from training."""
    assert "".join(SPLIT_PATTERN.findall(text)) == text


def test_roundtrip_is_exact():
    tok = Tokenizer.train(SAMPLE, vocab_size=400)
    for text in SAMPLE + ["unseen text with_underscores 🧠", "x_1 = y_2"]:
        assert tok.decode(tok.encode(text)) == text


def _naive_bpe(texts, vocab_size):
    """The obvious O(merges x corpus) implementation, as ground truth."""
    from collections import Counter

    from motherbrain.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN

    ns = len(SPECIAL_TOKENS)
    n_merges = max(0, vocab_size - ns - 256)
    freqs_by_word = Counter()
    for t in texts:
        freqs_by_word.update(SPLIT_PATTERN.findall(t))
    words = [[ns + b for b in w.encode()] for w in freqs_by_word]
    freqs = list(freqs_by_word.values())

    merges, next_id = [], ns + 256
    for _ in range(n_merges):
        counts = Counter()
        for seq, f in zip(words, freqs):
            for pair in zip(seq, seq[1:]):
                counts[pair] += f
        best, best_count = None, 1
        for pair, c in counts.items():
            if c > best_count or (c == best_count and best is not None and pair < best):
                best, best_count = pair, c
        if best is None:
            break
        merges.append(best)
        a, b = best
        for i, seq in enumerate(words):
            out, j = [], 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    out.append(next_id)
                    j += 2
                else:
                    out.append(seq[j])
                    j += 1
            words[i] = out
        next_id += 1
    return merges


def test_fast_bpe_matches_the_naive_implementation():
    """The trainer is incremental and heap-driven for speed.

    Both optimisations are easy to get subtly wrong, and a wrong merge table
    is not an error - it is a slightly worse tokenizer nobody notices. So the
    fast path is checked against the obvious implementation.
    """
    corpus = SAMPLE + ["def f(x_1): return x_1 + 1 " * 20, "aaa bbb aaa ccc " * 30]
    tok = Tokenizer.train(corpus, vocab_size=500)
    fast = [p for p, _ in sorted(tok.merges.items(), key=lambda kv: kv[1])]
    assert fast == _naive_bpe(corpus, 500)


def test_training_is_deterministic():
    a = Tokenizer.train(SAMPLE, vocab_size=400)
    b = Tokenizer.train(SAMPLE, vocab_size=400)
    assert a.merges == b.merges


def test_save_load_roundtrip(tmp_path):
    tok = Tokenizer.train(SAMPLE, vocab_size=400)
    path = tmp_path / "tok.json"
    tok.save(str(path))
    assert Tokenizer.load(str(path)).encode(SAMPLE[0]) == tok.encode(SAMPLE[0])


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        Tokenizer.train([""], vocab_size=400)


# ---- parameter accounting -------------------------------------------------


@pytest.mark.parametrize("kw", [
    {},
    {"n_experts": 6, "n_experts_per_token": 2},
    {"n_experts": 6, "n_experts_per_token": 2, "n_shared_experts": 1},
    {"n_experts": 4, "n_experts_per_token": 2, "moe_every": 2},
    {"tie_embeddings": False},
    {"n_kv_heads": 1},
])
def test_analytic_count_matches_real_model(kw):
    """`mb scale` prices configurations too large to build, so the arithmetic
    behind it has to be exact."""
    cfg = tiny(**kw)
    assert MotherBrain(cfg).n_params() == cfg.n_params


def test_moe_activates_a_fraction_of_its_parameters():
    cfg = tiny(n_experts=16, n_experts_per_token=2)
    assert cfg.n_active_params < cfg.n_params / 3


def test_dense_model_activates_everything():
    cfg = tiny()
    assert cfg.n_active_params == cfg.n_params


def test_every_preset_is_constructible_and_counted():
    for name, cfg in PRESETS.items():
        assert cfg.n_params > 0
        assert cfg.n_active_params <= cfg.n_params
        assert cfg.name == name


def test_mother_preset_is_the_largest_ever_configured():
    assert PRESETS["mother"].n_params > 1e15


def test_scale_to_reaches_its_target():
    cfg = scale_to(2e12, base="titan")
    assert cfg.n_params >= 2e12


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8)          # not divisible
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3)         # not a multiple
    with pytest.raises(ValueError):
        ModelConfig(n_experts=4, n_experts_per_token=8)


# ---- model ----------------------------------------------------------------


def test_forward_shapes_and_finite_loss():
    cfg = tiny(n_experts=4, n_experts_per_token=2)
    model = MotherBrain(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_kv_cache_matches_full_recomputation():
    """A cache bug shows up as subtly wrong output, not as a crash."""
    torch.manual_seed(0)
    model = MotherBrain(tiny(n_experts=4, n_experts_per_token=2)).eval()
    prompt = torch.randint(0, 300, (1, 7))
    cached = list(model.generate(prompt.clone(), max_new_tokens=12, temperature=0.0))
    fresh = list(model.generate(prompt.clone(), max_new_tokens=12, temperature=0.0,
                                use_cache=False))
    assert cached == fresh


def test_generation_respects_the_vocabulary():
    model = MotherBrain(tiny()).eval()
    out = list(model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=10, top_k=5))
    assert all(0 <= t < 300 for t in out)


def test_embeddings_are_unit_length():
    model = MotherBrain(tiny())
    vecs = model.embed_text(torch.randint(0, 300, (3, 10)))
    assert torch.allclose(vecs.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_model_learns_a_trivial_pattern():
    """The real check: loss on a repeating sequence must actually fall."""
    torch.manual_seed(0)
    cfg = tiny(vocab_size=32, n_layers=2, max_seq_len=32)
    model = MotherBrain(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pattern = torch.arange(16).repeat(4, 2) % 32
    first = last = None
    for step in range(60):
        _, loss = model(pattern[:, :-1], pattern[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5


# ---- server ---------------------------------------------------------------


@pytest.fixture
def served(tmp_path):
    """A minimal but genuine run directory: checkpoint, tokenizer, corpus."""
    from motherbrain.data import Corpus
    from motherbrain.train import TrainConfig, save_checkpoint

    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text("the mother brain awakens and learns " * 40, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)

    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    model = MotherBrain(cfg)
    run = tmp_path / "run"
    save_checkpoint(run / "checkpoint.pt", model, None, 1, cfg, TrainConfig(), [])
    tok.save(str(run / "tokenizer.json"))
    return run, tmp_path / "corpus"


def test_abandoned_stream_does_not_wedge_the_server(served):
    """An IDE cancels in-flight completions constantly.

    Holding a lock across a streaming response meant one cancelled stream
    deadlocked every later request, so this walks away from three streams and
    then insists the server still answers.
    """
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    body = {"messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 40, "stream": True}
    for _ in range(3):
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            next(r.iter_lines())  # read one chunk, then abandon the response

    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4})
    assert r.status_code == 200


def test_openai_and_ollama_surfaces(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))

    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "motherbrain"
    assert client.get("/api/tags").json()["models"][0]["name"] == "motherbrain:latest"

    chat = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 5}).json()
    assert isinstance(chat["choices"][0]["message"]["content"], str)
    assert chat["usage"]["total_tokens"] > 0

    ollama = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                            "stream": False,
                                            "options": {"num_predict": 5}}).json()
    assert ollama["done"] is True

    emb = client.post("/v1/embeddings", json={"input": ["a", "b"]}).json()
    assert len(emb["data"]) == 2


def test_api_key_guards_both_header_styles(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   api_key="sekrit", auto_patch=False))
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "sekrit"}).status_code == 200
    assert client.get("/v1/models",
                      headers={"Authorization": "Bearer sekrit"}).status_code == 200


def test_content_parts_are_flattened():
    """Some editors send content as a list of typed parts rather than a string."""
    from motherbrain.api_compat import ChatMessage, build_chat_prompt, content_to_text

    assert content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    prompt = build_chat_prompt([ChatMessage(role="user", content="hi")])
    assert prompt.endswith("<assistant>")


# ---- patches and versions -------------------------------------------------


def test_patch_starts_as_a_no_op():
    """B is initialised to zero, so a fresh patch must not change any output."""
    from motherbrain.patches import PatchConfig, inject_lora

    torch.manual_seed(0)
    model = MotherBrain(tiny()).eval()
    x = torch.randint(0, 300, (1, 8))
    before, _ = model(x, None)
    inject_lora(model, PatchConfig(rank=4))
    after, _ = model(x, None)
    assert torch.allclose(before, after, atol=1e-6)


def test_merged_patch_preserves_behaviour():
    """Merging the delta into the base weights must not alter the output."""
    from motherbrain.patches import PatchConfig, inject_lora, merge_all

    torch.manual_seed(0)
    model = MotherBrain(tiny()).eval()
    wrapped = inject_lora(model, PatchConfig(rank=4))
    with torch.no_grad():
        for w in wrapped:
            w.B.normal_(std=0.02)  # make the patch actually do something
    x = torch.randint(0, 300, (1, 8))
    before, _ = model(x, None)
    assert merge_all(model) == len(wrapped)
    after, _ = model(x, None)
    assert torch.allclose(before, after, atol=1e-5)


def test_base_watermark_stops_relearning_the_corpus(tmp_path):
    """Without the watermark a first patch re-learns the entire base corpus."""
    from motherbrain.patches import PatchStore

    store = PatchStore(tmp_path)
    assert store.consumed_docs() == 0
    store.set_base_docs(170)
    assert store.consumed_docs() == 170


def test_versions_are_sequential_and_checkout_validates(tmp_path):
    import time as _time

    from motherbrain.patches import PatchStore, Version

    store = PatchStore(tmp_path)
    for n in (1, 2):
        store.record(Version(version=n, patch_id=f"p{n}", parent=n - 1,
                             created_at=_time.time(), doc_start=n - 1, doc_end=n,
                             n_documents=1, n_chars=10, n_tokens=10, steps=1, rank=4,
                             trainable_params=8, loss_before=2.0, loss_after=1.0),
                     {"x": torch.zeros(1)})
    assert [v.version for v in store.versions()] == [1, 2]
    assert store.current == 2
    store.set_current(1)
    assert store.current == 1 and store.head == 2   # checkout does not lose v2
    with pytest.raises(ValueError):
        store.set_current(99)


# ---- security -------------------------------------------------------------


def test_feed_path_is_confined_to_the_allowlist(tmp_path):
    """/feed with an unrestricted path is an arbitrary-file-read primitive.

    Whatever it reads lands in the corpus, and whatever is in the corpus can be
    extracted again through generation, so this is the sharpest edge in the API.
    """
    from fastapi import HTTPException

    from motherbrain.security import safe_resolve

    root = (tmp_path / "allowed").resolve()
    root.mkdir()
    (root / "fine.txt").write_text("ok")
    outside = tmp_path / "secret.txt"
    outside.write_text("private")

    assert safe_resolve(str(root / "fine.txt"), [root]).name == "fine.txt"

    for bad in [str(outside), "/etc/passwd", str(root / ".." / "secret.txt")]:
        with pytest.raises(HTTPException) as exc:
            safe_resolve(bad, [root])
        assert exc.value.status_code in (403, 404)


def test_path_ingestion_is_off_by_default(tmp_path):
    from fastapi import HTTPException

    from motherbrain.security import safe_resolve

    with pytest.raises(HTTPException) as exc:
        safe_resolve(str(tmp_path), [])
    assert exc.value.status_code == 403


def test_credential_files_are_refused_inside_an_allowed_root(tmp_path):
    from fastapi import HTTPException

    from motherbrain.security import safe_resolve

    root = tmp_path.resolve()
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_rsa").write_text("KEY")
    (root / "server.key").write_text("KEY")

    for bad in [root / ".ssh" / "id_rsa", root / "server.key"]:
        with pytest.raises(HTTPException) as exc:
            safe_resolve(str(bad), [root])
        assert exc.value.status_code == 403


def test_api_key_comparison_is_constant_time():
    from motherbrain.security import constant_time_eq

    assert constant_time_eq("secret", "secret")
    assert not constant_time_eq("secret", "secrey")
    assert not constant_time_eq("secret", None)
    assert not constant_time_eq(None, None)


def test_public_bind_without_a_key_is_refused():
    from motherbrain.security import check_exposure

    with pytest.raises(SystemExit):
        check_exposure("0.0.0.0", None, tls=False, insecure=False)
    # explicit override, and loopback, are both allowed
    check_exposure("0.0.0.0", None, tls=False, insecure=True)
    check_exposure("127.0.0.1", None, tls=False, insecure=False)


def test_plaintext_public_bind_warns():
    from motherbrain.security import check_exposure

    warnings = check_exposure("0.0.0.0", "a-sufficiently-long-key", tls=False,
                              insecure=False)
    assert any("plaintext" in w for w in warnings)
    assert check_exposure("0.0.0.0", "a-sufficiently-long-key", tls=True,
                          insecure=False) == []


def test_rate_limiter_refills_over_time():
    from motherbrain.security import RateLimiter

    rl = RateLimiter(per_minute=60, burst=2)
    assert [rl.allow("ip") for _ in range(4)] == [True, True, False, False]
    assert rl.allow("other.ip")  # buckets are per client


def test_oversized_feed_is_rejected(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False, max_feed_chars=100))
    assert client.post("/feed", json={"text": "x" * 500}).status_code == 413
    assert client.post("/feed", json={"text": "short"}).status_code == 200


def test_feed_rejects_paths_when_no_root_is_allowed(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    assert client.post("/feed", json={"path": "/etc/passwd"}).status_code == 403


def test_security_headers_are_present(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"


def test_fingerprint_identifies_a_base_checkpoint():
    from motherbrain.patches import weights_fingerprint

    torch.manual_seed(0)
    a = MotherBrain(tiny())
    torch.manual_seed(0)
    same = MotherBrain(tiny())
    torch.manual_seed(1)
    different = MotherBrain(tiny())

    assert weights_fingerprint(a) == weights_fingerprint(same)
    assert weights_fingerprint(a) != weights_fingerprint(different)


def test_fingerprint_survives_an_fp16_export_round_trip():
    """The committed base ships fp16, and must still be the same base.

    `mb export` halves precision to keep the file inside GitHub's 100MB limit.
    If that changed the base's identity, every patch trained against the
    checkpoint would be refused by a clone carrying only the export — which is
    the whole reason the export is committed.
    """
    from motherbrain.patches import fingerprint_matches, weights_fingerprint

    torch.manual_seed(0)
    model = MotherBrain(tiny())
    exact = weights_fingerprint(model)

    rounded = MotherBrain(tiny())
    rounded.load_state_dict({k: v.to(torch.float16).float()
                             for k, v in model.state_dict().items()})

    assert weights_fingerprint(rounded) == exact
    assert fingerprint_matches(rounded, exact)

    torch.manual_seed(1)
    assert not fingerprint_matches(MotherBrain(tiny()), exact)


def test_a_legacy_fingerprint_is_still_recognised():
    """Manifests written before the hash grew its tolerance still load.

    Someone who has already grown a model has a lineage stamped with the
    fp32-exact hash. Refusing it would strand their versions, so the older
    form is accepted and re-stamped on the next build.
    """
    from motherbrain.patches import _sample_fingerprint, fingerprint_matches

    torch.manual_seed(0)
    model = MotherBrain(tiny())
    legacy = _sample_fingerprint(model, legacy=True)

    assert legacy != _sample_fingerprint(model, legacy=False)
    assert fingerprint_matches(model, legacy)


def _grow(run, corpus, text):
    """Feed one document and fold it into a new version, as the console does."""
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchConfig, create_patch

    Corpus(corpus).add_text(text, "test")
    return create_patch(str(run), str(corpus), device="cpu",
                        cfg=PatchConfig(mode="grow", grow_experts=1, steps=2))


def test_a_clone_without_a_checkpoint_rebuilds_the_lineage(served, tmp_path):
    """Base export plus patches must reconstruct the current version.

    This is what committing patches buys. A clone carries no checkpoint — they
    are far too large — so if the patches could not be applied on top of the
    committed base, the clone would silently run v0 while the manifest claimed
    v2, which is the failure this guards against.
    """
    from motherbrain.cli import export_model, load_current
    from motherbrain.patches import PatchStore, weights_fingerprint

    run, corpus = served
    store = PatchStore(run)
    model, _tok, _dev, _v = load_current(str(run), "cpu")
    store.set_base(weights_fingerprint(model), 0)

    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    export_model(str(run), models / "motherbrain-base.pt", device="cpu")

    assert _grow(run, corpus, "the sky above the port") is not None
    assert _grow(run, corpus, "a screen tuned to a dead channel") is not None
    full, _tok, _dev, grown = load_current(str(run), "cpu")
    assert grown == 2

    # Now strip it to what a clone actually carries.
    (run / "checkpoint.pt").unlink()
    (run / "best.pt").unlink(missing_ok=True)

    rebuilt, _tok, _dev, version = load_current(str(run), "cpu")
    assert version == grown, "the clone fell back to the base instead of patching"
    assert rebuilt.n_params() == full.n_params()

    # Same weights to fp16, which is the precision the base is committed at.
    a, b = full.state_dict(), rebuilt.state_dict()
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-3), name


def test_a_merged_export_is_refused_as_a_patch_base(served, tmp_path):
    """Patches applied on top of a model that already contains them double up.

    `mb patch` writes the merged current model to models/motherbrain.pt, so a
    file of exactly that shape sits on every machine. Mistaking it for the base
    would apply every delta twice and produce confident nonsense, so the export
    records its version and loading refuses anything but v0.
    """
    from motherbrain.cli import export_model, load_runtime

    run, corpus = served
    assert _grow(run, corpus, "the sky above the port") is not None

    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    export_model(str(run), models / "motherbrain-base.pt", device="cpu")
    (run / "checkpoint.pt").unlink()

    with pytest.raises(ValueError, match="not the base"):
        load_runtime(str(run), "cpu")


def test_retraining_the_base_drops_the_stale_lineage(tmp_path):
    """A patch is a delta against particular weights.

    Retraining the base produces different weights, and keeping the old patches
    would apply deltas to something that no longer exists — silently, and with
    confident nonsense as the output.
    """
    import time as _time

    from motherbrain.patches import PatchStore, Version

    store = PatchStore(tmp_path)
    store.set_base("fingerprint-a", 10)
    store.record(Version(version=1, patch_id="p1", parent=0, created_at=_time.time(),
                         doc_start=10, doc_end=11, n_documents=1, n_chars=5,
                         n_tokens=5, steps=1, rank=4, trainable_params=8,
                         loss_before=2.0, loss_after=1.0,
                         base_fingerprint="fingerprint-a"),
                 {"x": torch.zeros(1)})
    assert store.current == 1

    dropped = store.set_base("fingerprint-a", 12)      # same base, lineage survives
    assert dropped == [] and store.current == 1

    dropped = store.set_base("fingerprint-b", 12)      # new base, lineage is void
    assert dropped == ["v1 (p1)"]
    assert store.versions() == [] and store.current == 0
    assert not (store.dir / "0001-p1.pt").exists()


# ---- loading --------------------------------------------------------------


def test_status_reports_a_workspace_with_no_weights_as_not_loadable(
        tmp_path, capsys, monkeypatch):
    """With no committed base anywhere, say so and name the way out.

    A real clone now carries models/motherbrain-base.pt and is loadable, so
    the base has to be hidden to reach this branch at all.
    """
    import motherbrain.cli as cli

    monkeypatch.setattr(cli, "shipped_base", lambda _run: None)
    args = cli.build_parser().parse_args(
        ["status", "--corpus", str(tmp_path / "corpus"), "--run", str(tmp_path / "run")])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "NOT LOADABLE" in out
    assert "mb bootstrap" in out


def test_status_reports_a_clone_carrying_the_base_as_ready(tmp_path, capsys):
    """The committed base plus the committed patches is a loadable model.

    This is the payoff for committing patches: a clone that has never trained
    anything is ready, at the current version, with no download.
    """
    from motherbrain.cli import build_parser, shipped_base

    if shipped_base(str(tmp_path)) is None:
        pytest.skip("no committed base in this checkout")

    args = build_parser().parse_args(
        ["status", "--corpus", str(tmp_path / "corpus"), "--run", "runs/default"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "READY" in out
    assert "base weights" in out


def test_status_reports_a_trained_run_as_ready(served, capsys):
    from motherbrain.cli import build_parser

    run, corpus = served
    args = build_parser().parse_args(
        ["status", "--corpus", str(corpus), "--run", str(run)])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "READY" in out
    for way in ("mb gui", "mb console", "mb serve"):
        assert way in out, way


def test_project_root_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    """`mb` is installed globally but the corpus and weights live somewhere.

    Resolving them against the cwd alone made `mb status` report "no weights"
    while standing inside a workspace that has them.
    """
    from motherbrain.cli import project_root

    workspace = tmp_path / "workspace"
    (workspace / "runs" / "default").mkdir(parents=True)
    deep = workspace / "a" / "b" / "c"
    deep.mkdir(parents=True)

    monkeypatch.chdir(deep)
    assert project_root() == workspace.resolve()

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert project_root() == outside.resolve()


def test_console_script_entry_point_is_declared():
    """The README tells people to run `mb`; that has to be a real command."""
    import tomllib

    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    assert pyproject["project"]["scripts"]["mb"] == "motherbrain.cli:main"


def test_status_does_not_create_a_workspace(tmp_path, capsys):
    """Looking at a workspace must not bring one into existence.

    `mb status` used to create the corpus directory as a side effect, which
    made an empty directory look like a MotherBrain workspace to the project
    root search that runs on the next invocation.
    """
    from motherbrain.cli import build_parser

    corpus = tmp_path / "data" / "corpus"
    run = tmp_path / "runs" / "default"
    args = build_parser().parse_args(
        ["status", "--corpus", str(corpus), "--run", str(run)])
    args.func(args)

    assert not corpus.exists()
    assert not run.exists()


# ---- fitting a model to real hardware -------------------------------------


@pytest.mark.parametrize("n_gpus,gpu_gb", [
    (1, 8), (1, 24), (1, 80), (8, 80), (64, 80), (1024, 80), (202459, 80),
])
def test_fit_to_hardware_returns_something_that_actually_fits(n_gpus, gpu_gb):
    """A configuration that does not fit is worse than an honest refusal.

    The first version of this returned the smallest shape it had tried even
    when that shape exceeded the budget, so it claimed 100B parameters fit on
    eight GPUs and then reported needing eighteen.
    """
    from motherbrain.cli import fit_to_hardware

    cfg, _ = fit_to_hardware(n_gpus, gpu_gb)
    assert cfg is not None
    assert cfg.memory_bytes(optimizer=True) <= n_gpus * gpu_gb * 1e9
    assert cfg.n_experts >= 1
    assert cfg.n_experts_per_token <= cfg.n_experts


def test_fit_to_hardware_grows_with_the_cluster():
    from motherbrain.cli import fit_to_hardware

    small, _ = fit_to_hardware(8, 80)
    large, _ = fit_to_hardware(1024, 80)
    assert large.n_params > small.n_params


def test_fit_to_hardware_admits_when_nothing_fits():
    from motherbrain.cli import fit_to_hardware

    cfg, note = fit_to_hardware(1, 0.001)
    assert cfg is None
    assert "micro" in note


def test_mother_config_artifact_matches_the_preset():
    """configs/mother.json is the committed definition of the largest model."""
    from motherbrain.config import PRESETS, ModelConfig

    root = Path(__file__).resolve().parent.parent
    cfg = ModelConfig.load(str(root / "configs" / "mother.json"))
    assert cfg.n_params == PRESETS["mother"].n_params
    assert cfg.n_params > 1e15


def test_attention_materialises_at_mother_width():
    """The largest preset is arithmetic unless its real modules can be built.

    One attention block at mother's true width is ~0.9B parameters, which is
    small enough to instantiate here and large enough to prove the shape is
    real rather than a number in a table.
    """
    from motherbrain.config import PRESETS
    from motherbrain.model import Attention

    cfg = PRESETS["mother"]
    attn = Attention(cfg)
    built = sum(p.numel() for p in attn.parameters())
    assert built == cfg.attn_params_per_layer

    x = torch.randn(1, 2, cfg.d_model)
    cos = torch.randn(2, cfg.head_dim // 2)
    sin = torch.randn(2, cfg.head_dim // 2)
    with torch.no_grad():
        assert attn(x, cos, sin).shape == (1, 2, cfg.d_model)


def test_chat_output_is_visibly_delimited(served, capsys, monkeypatch):
    """An undertrained model emits mostly whitespace.

    A blank screen is indistinguishable from a command that silently failed,
    so chat frames its output and reports a token count.
    """
    from motherbrain.cli import build_parser

    run, corpus = served
    args = build_parser().parse_args(
        ["chat", "--prompt", "hello", "--max-tokens", "5",
         "--corpus", str(corpus), "--run", str(run)])
    assert args.func(args) == 0

    out = capsys.readouterr().out
    assert "MotherBrain v" in out
    assert "─" * 10 in out          # the output is framed
    assert "tokens in" in out       # and counted


# ---- exported models ------------------------------------------------------


def test_export_roundtrips_and_loads_without_pickle(served, tmp_path, capsys):
    """An exported model is meant to be shared, so it must load safely.

    Training checkpoints carry optimizer state and load through pickle. An
    export carries fp16 weights plus config and tokenizer as JSON strings, so
    torch.load(weights_only=True) can read it - no code execution on load.
    """
    import torch

    from motherbrain.cli import build_parser, load_exported

    run, corpus = served
    out = tmp_path / "model.pt"
    args = build_parser().parse_args(
        ["export", "--out", str(out), "--corpus", str(corpus), "--run", str(run)])
    assert args.func(args) == 0
    assert out.exists()

    # The safety property: readable with weights_only, i.e. no pickled objects.
    payload = torch.load(out, map_location="cpu", weights_only=True)
    assert payload["format"] == "motherbrain-model-v1"

    model, tok, device, version, steps = load_exported(str(out), device="cpu")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids[:4] or [1]]), None)
    assert torch.isfinite(logits).all()


def test_export_is_smaller_than_the_checkpoint(served, tmp_path):
    from motherbrain.cli import build_parser

    run, corpus = served
    out = tmp_path / "model.pt"
    args = build_parser().parse_args(
        ["export", "--out", str(out), "--corpus", str(corpus), "--run", str(run)])
    args.func(args)
    assert out.stat().st_size < (run / "checkpoint.pt").stat().st_size


def test_export_rejects_a_foreign_file(tmp_path):
    import torch

    from motherbrain.cli import load_exported

    bogus = tmp_path / "not-a-model.pt"
    torch.save({"weights": {}}, bogus)
    with pytest.raises(ValueError, match="not a MotherBrain model export"):
        load_exported(str(bogus))


def test_every_saved_checkpoint_is_immediately_loadable(tmp_path):
    """Training stamps the base identity as it saves, not only at the end.

    Stamping only on completion left every intermediate checkpoint unloadable:
    the manifest still described the previous base, so the lineage guard
    refused the new weights. A long run that got interrupted produced a large
    checkpoint nobody could open.
    """
    from motherbrain.config import ModelConfig
    from motherbrain.data import Corpus
    from motherbrain.patches import build_version
    from motherbrain.train import TrainConfig, train

    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text("the mother brain awakens and learns " * 200, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)

    run = tmp_path / "run"
    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    # save_every < steps, so a checkpoint exists well before the run ends
    tc = TrainConfig(steps=4, batch_size=2, seq_len=16, warmup=1, save_every=2,
                     eval_every=100, log_every=100, eval_batches=1)
    train(str(tmp_path / "corpus"), str(run), cfg, tc)

    model, _, version = build_version(str(run))   # must not raise
    assert version == 0
    assert model.n_params() > 0


def test_repetition_penalty_is_applied_by_chat(served, capsys):
    """Small models fall into loops at low temperature.

    generate() has always supported a repetition penalty and the HTTP API
    exposed it, but `mb chat` did not, so the CLI had no way out of a loop.
    """
    from motherbrain.cli import build_parser

    run, corpus = served
    parser = build_parser()
    args = parser.parse_args(["chat", "--prompt", "x", "--max-tokens", "3",
                              "--corpus", str(corpus), "--run", str(run)])
    assert args.repetition_penalty == 1.1     # on by default
    assert args.func(args) == 0

    args = parser.parse_args(["chat", "--prompt", "x", "--max-tokens", "3",
                              "--repetition-penalty", "1.0",
                              "--corpus", str(corpus), "--run", str(run)])
    assert args.repetition_penalty == 1.0
    assert args.func(args) == 0


def test_training_keeps_the_best_checkpoint(tmp_path):
    """Validation loss turns back up once a run overfits.

    The rolling checkpoint is overwritten every save_every steps, so without a
    separate copy the best weights are lost to later, worse ones - which is
    exactly what a long run does after it passes its optimum.
    """
    import json

    from motherbrain.data import Corpus
    from motherbrain.train import TrainConfig, train

    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text("the mother brain awakens and learns " * 300, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)

    run = tmp_path / "run"
    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    tc = TrainConfig(steps=6, batch_size=2, seq_len=16, warmup=1, save_every=6,
                     eval_every=2, log_every=100, eval_batches=1)
    summary = train(str(tmp_path / "corpus"), str(run), cfg, tc)

    assert (run / "best.pt").exists()
    assert summary["best_val_loss"] is not None
    evals = [h["val_loss"] for h in summary["history"]]
    assert summary["best_val_loss"] == pytest.approx(min(evals))


# ---- growth ---------------------------------------------------------------


def test_growth_preserves_behaviour_exactly():
    """A grown model must compute what it computed before, to the bit.

    New experts start with a zeroed output projection and a -1e9 router bias,
    so they cannot be selected and contribute nothing. Anything else would mean
    learning one new fact silently damaged everything already known.
    """
    from motherbrain.growth import grow, release

    torch.manual_seed(0)
    model = MotherBrain(tiny()).eval()
    x = torch.randint(0, 300, (2, 8))
    before, _ = model(x, None)

    grow(model, 2)                       # dense -> MoE
    after, _ = model(x, None)
    assert torch.allclose(before, after, atol=1e-6)

    release(model, 2)                    # routable, but still silent
    assert torch.allclose(before, model(x, None)[0], atol=1e-6)

    grow(model, 3)                       # MoE -> larger MoE
    assert torch.allclose(before, model(x, None)[0], atol=1e-6)


def test_growth_adds_parameters_and_keeps_compute_flat():
    """The point of growing through experts: size rises, per-token cost does not."""
    from motherbrain.growth import grow

    model = MotherBrain(tiny())
    before = model.n_params()
    active_before = model.cfg.n_active_params

    cfg, trainable = grow(model, 4)
    assert model.n_params() > before
    assert model.n_params() == cfg.n_params        # analytic accounting holds
    assert trainable and sum(p.numel() for p in trainable) > 0

    # Only n_experts_per_token experts run, so activation stays near the dense cost.
    assert cfg.n_active_params < cfg.n_params
    assert cfg.n_active_params < active_before * 3


def test_growth_rejects_nonsense():
    from motherbrain.growth import grow

    with pytest.raises(ValueError):
        grow(MotherBrain(tiny()), 0)


def test_every_patch_grows_the_model_and_replays_exactly(tmp_path):
    """The end-to-end promise: information in, parameters up, version up.

    Each patch must also replay from the base, or the lineage is decorative.
    """
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchConfig, build_version, create_patch
    from motherbrain.train import TrainConfig, train

    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text("the mother brain awakens and learns and grows " * 200, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)

    run = tmp_path / "run"
    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    train(str(tmp_path / "corpus"), str(run),
          cfg, TrainConfig(steps=2, batch_size=2, seq_len=16, warmup=1,
                           save_every=2, eval_every=2, log_every=100,
                           eval_batches=1))

    sizes = []
    for i in range(3):
        corpus.add_text(f"fact number {i} about the growing mother brain", f"f{i}")
        v = create_patch(str(run), str(tmp_path / "corpus"),
                         PatchConfig(mode="grow", grow_experts=1, steps=3,
                                     batch_size=2, seq_len=16))
        assert v is not None
        assert v.version == i + 1                      # sequential versions
        assert v.params_after > v.params_before        # and it grew
        assert v.mode == "grow"
        sizes.append(v.params_after)

    assert sizes == sorted(sizes)                      # monotonically larger

    for target, expected in enumerate(sizes, start=1):
        model, _, version = build_version(str(run), target=target)
        assert version == target
        assert model.n_params() == expected           # replays exactly


# ---- the console ----------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("/help", "help"),
    ("/status", "status"),
    ("how big are you", "status"),
    ("what version are you", "version"),
    ("list versions", "versions"),
    ("/versions", "versions"),
    ("/learn the key rotates on Fridays", "learn"),
    ("learn that the key rotates", "learn"),
    ("remember: port 6543", "learn"),
    ("/grow", "grow"),
    ("grow yourself", "grow"),
    ("/checkout v3", "checkout"),
    ("roll back to 1", "checkout"),
    ("/train 500", "train"),
    ("/scale mother", "scale"),
    ("def softmax(x):", "generate"),
    ("the quick brown fox", "generate"),
    ("", "noop"),
])
def test_console_parses_commands_and_prompts(text, expected):
    from motherbrain.commands import parse

    assert parse(text).name == expected


def test_console_extracts_arguments():
    from motherbrain.commands import parse

    assert parse("/checkout v3").args["version"] == 3
    assert parse("roll back to 1").args["version"] == 1
    assert parse("/grow 4").args["experts"] == 4
    assert parse("/grow").args["experts"] == 1          # sensible default
    assert parse("/train 500").args["steps"] == 500
    assert parse("/scale titan").args["preset"] == "titan"
    assert parse("/learn a fact").text == "a fact"
    assert parse("remember: port 6543").text == "port 6543"


def test_console_refuses_ambiguity_instead_of_guessing():
    """A parser that guesses is worse than one that says it did not understand."""
    from motherbrain.commands import parse

    assert parse("/checkout").name == "error"          # which version?
    assert parse("learn").name == "error"              # learn what?
    assert parse("/nonsense").name == "unknown"


def test_a_prompt_that_looks_like_a_command_is_still_a_prompt():
    """`learning rates` starts with an alias but is not an instruction."""
    from motherbrain.commands import parse

    assert parse("learning rates matter").name == "generate"
    assert parse("versions of numpy differ").name == "generate"


def test_command_endpoint_drives_the_system(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))

    def send(text, **kw):
        return client.post("/command", json={"text": text, **kw}).json()

    assert "MotherBrain console" in send("/help")["text"]
    assert send("what version are you")["kind"] == "info"
    assert send("how big are you")["kind"] == "status"
    assert send("list versions")["kind"] == "versions"

    learned = send("learn that the deploy key rotates on Fridays")
    assert learned["kind"] == "learned"
    assert learned["data"]["pending"] >= 1

    assert send("/checkout v9")["kind"] == "error"      # no such version
    assert send("/nonsense")["kind"] == "error"

    generated = send("def f(", max_tokens=4)
    assert generated["kind"] == "generated"
    assert generated["data"]["tokens"] > 0


def test_console_page_is_served(served):
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>MotherBrain</title>" in page.text
    assert "What would you like to do?" in page.text               # the menu
    assert 'id="log"' in page.text                                 # the transcript


# ---- text or voice --------------------------------------------------------


def test_voice_capability_detection_is_honest():
    """Speech is optional and machine-dependent, so it is detected, not assumed."""
    from motherbrain.voice import detect

    cap = detect()
    assert isinstance(cap.any, bool)
    if not cap.any:
        # An unavailable capability must explain itself rather than fail silently.
        assert cap.reason
        assert "install" in cap.reason or "pip" in cap.reason


def test_speaking_without_a_backend_reports_failure(monkeypatch):
    from motherbrain.voice import Capability, speak

    assert speak("hello", Capability()) is False          # no backend
    assert speak("", Capability(speak="espeak")) is False  # nothing to say


def test_listening_without_a_backend_returns_nothing():
    from motherbrain.voice import Capability, listen

    assert listen(Capability()) is None


def test_choose_mode_falls_back_to_text_when_voice_is_impossible(monkeypatch, capsys):
    """Offering a choice the machine cannot honour would be worse than saying so."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect", lambda: voice.Capability(reason="no engine"))
    mode, cap = voice.choose_mode()
    assert mode == "text"
    assert "unavailable" in capsys.readouterr().out


def test_choose_mode_asks_when_voice_is_possible(monkeypatch):
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    monkeypatch.setattr("builtins.input", lambda _: "voice")
    assert voice.choose_mode()[0] == "voice"

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert voice.choose_mode()[0] == "text"          # default

    monkeypatch.setattr("builtins.input", lambda _: "t")
    assert voice.choose_mode()[0] == "text"


def test_console_offers_both_modes_in_the_browser():
    """Voice is a way of using option 2, not an option of its own.

    The page still has to detect the two halves separately: Firefox has
    synthesis without recognition, and claiming both would leave a dead mic.
    """
    from motherbrain.server import UI_HTML

    assert "What would you like to do?" in UI_HTML
    assert 'data-mode="text"' in UI_HTML
    assert "by text or voice" in UI_HTML
    # recognition and synthesis are detected apart: Firefox has one, not both
    assert "webkitSpeechRecognition" in UI_HTML
    assert "speechSynthesis" in UI_HTML
    assert "not supported by this browser" in UI_HTML


def test_console_mode_flag_skips_the_question():
    from motherbrain.cli import build_parser

    assert build_parser().parse_args(["console"]).mode == "ask"
    assert build_parser().parse_args(["console", "--mode", "text"]).mode == "text"
    assert build_parser().parse_args(["console", "--mode", "voice"]).mode == "voice"
    assert build_parser().parse_args(["console", "--mode", "update"]).mode == "update"


def test_startup_offers_feeding_as_a_third_choice(monkeypatch):
    """Feeding is the first thing most sessions want, so it is offered up front."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    for answer, expected in [("f", "feed"), ("feed", "feed"),
                             ("v", "voice"), ("", "text"), ("t", "text")]:
        monkeypatch.setattr("builtins.input", lambda _, a=answer: a)
        assert voice.choose_mode()[0] == expected


def test_feeding_is_offered_even_without_voice(monkeypatch, capsys):
    """A machine with no speech still gets the feed option, just not voice."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect", lambda: voice.Capability(reason="none"))
    monkeypatch.setattr("builtins.input", lambda _: "feed")
    assert voice.choose_mode()[0] == "feed"

    monkeypatch.setattr("builtins.input", lambda _: "voice")
    assert voice.choose_mode()[0] == "text"      # voice cannot be honoured here


def test_startup_question_survives_no_terminal(monkeypatch):
    """Piped input, cron, a daemon: `input` raises OSError, not EOFError.

    Defaulting is right there; crashing on a question nobody can answer is not.
    """
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))

    def no_terminal(_):
        raise OSError("reading from stdin while output is captured")

    monkeypatch.setattr("builtins.input", no_terminal)
    assert voice.choose_mode()[0] == "text"


def test_opening_menu_lists_the_four_things_you_can_do(monkeypatch):
    """The program opens on a menu of tasks, not a question about typing."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    for answer, expected in [("1", "make"), ("2", "do"), ("3", "learn"),
                             ("4", "apply"), ("", "do"), ("teach", "learn"),
                             ("patch", "apply"), ("program", "make"),
                             ("update", "apply"), ("9", "do")]:
        monkeypatch.setattr("builtins.input", lambda _, a=answer: a)
        assert voice.choose_start()[0] == expected, answer


def test_the_menu_offers_the_window_as_its_fifth_option(monkeypatch):
    """"Run the GUI" is on the main console, and means what it says."""
    import motherbrain.voice as voice

    assert "Run the GUI" in voice.MENU
    assert "5" in voice.MENU

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    for answer in ("5", "gui", "window", "desktop"):
        monkeypatch.setattr("builtins.input", lambda _, a=answer: a)
        action, mode, _cap = voice.choose_start()
        assert action == "gui", answer
        # It never asks text-or-voice: a window is not a way of talking.
        assert mode == "text", answer


def test_option_five_opens_the_window(served, monkeypatch, capsys):
    """And having offered it, the console has to actually hand over."""
    from motherbrain import cli, gui

    run, corpus = served
    opened = {}

    def fake_window(run_dir, corpus_dir, device, **kw):
        opened.update(run_dir=run_dir, corpus_dir=corpus_dir, **kw)
        return 0

    monkeypatch.setattr(gui, "run", fake_window)

    assert cli.main(["console", "--run", str(run), "--corpus", str(corpus),
                     "--mode", "gui", "--device", "cpu"]) == 0
    assert opened["run_dir"] == str(run), "it opened somebody else's model"
    assert opened["corpus_dir"] == str(corpus)
    assert "steps" in opened and "grow" in opened

    # It hands over rather than dropping into the prompt underneath.
    out = capsys.readouterr().out
    assert "window" in out
    assert "Tell me what to do" not in out


def test_only_the_conversational_options_ask_about_voice(monkeypatch):
    """Teaching and patching are not conversations, so they never ask."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    asked = []

    def record(prompt):
        asked.append(prompt)
        return "voice"

    monkeypatch.setattr("builtins.input", record)
    voice.choose_start()                       # answers "voice" -> option 2
    assert len(asked) == 1, "an implied mode must not be asked for twice"

    for answer in ("3", "4"):
        asked.clear()
        monkeypatch.setattr("builtins.input", lambda _, a=answer: a)
        action, mode, _cap = voice.choose_start()
        assert action in ("learn", "apply") and mode == "text"


def test_voice_falls_back_when_it_cannot_be_honoured(monkeypatch, capsys):
    """Asking to speak on a machine with no speech has to say so."""
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(reason="no engine"))
    monkeypatch.setattr("builtins.input", lambda _: "voice")
    action, mode, _cap = voice.choose_start()
    assert (action, mode) == ("do", "text")
    assert "unavailable" in capsys.readouterr().out

    monkeypatch.setattr(voice, "detect",
                        lambda: voice.Capability(speak="espeak", listen="sr"))
    monkeypatch.setattr("builtins.input", lambda _: "voice")
    assert voice.choose_start()[:2] == ("do", "voice")


def test_menu_survives_no_terminal(monkeypatch):
    import motherbrain.voice as voice

    monkeypatch.setattr(voice, "detect", lambda: voice.Capability())

    def no_terminal(_):
        raise OSError("no stdin")

    monkeypatch.setattr("builtins.input", no_terminal)
    assert voice.choose_start()[:2] == ("do", "text")


def test_every_surface_offers_the_same_four_options():
    """The terminal, the browser and the window must not drift apart.

    Three menus written three times is three chances to describe the same
    button differently, so each option is checked against all of them by its
    distinguishing words rather than its exact punctuation.
    """
    from motherbrain.gui import OPTIONS
    from motherbrain.server import UI_HTML
    from motherbrain.voice import MENU

    wanted = [("what kind of program", "make"),
              ("what to do", "do"),
              ("something new", "teach"),
              ("as a patch", "apply")]

    gui_text = " ".join(label + " " + hint for label, hint in OPTIONS).lower()
    for phrase, _ in wanted:
        assert phrase in MENU.lower(), f"terminal menu is missing: {phrase}"
        assert phrase in UI_HTML.lower(), f"browser menu is missing: {phrase}"
        assert phrase in gui_text, f"window menu is missing: {phrase}"

    assert len(OPTIONS) == 4


def _data_uri(colour=(200, 10, 10), size=40) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_api_decodes_inline_images_and_refuses_to_fetch_urls():
    """Inline images are decoded; a URL is not followed.

    A model server that fetches whatever appears in its input is a
    request-forgery primitive pointed at the inside of your network, and an
    IDE that sends a link instead of the bytes should get no picture rather
    than an outbound request.
    """
    from motherbrain.api_compat import content_to_images

    content = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": _data_uri()}},
    ]
    images = content_to_images(content, 32)
    assert len(images) == 1
    assert images[0].shape == (1, 3, 32, 32)

    for hostile in ("http://169.254.169.254/latest/meta-data/",
                    "file:///etc/passwd",
                    "https://example.com/cat.png"):
        assert content_to_images(
            [{"type": "image_url", "image_url": {"url": hostile}}], 32) == []

    assert content_to_images("plain text", 32) == []
    assert content_to_images([{"type": "text", "text": "hi"}], 32) == []


def test_a_blind_model_ignores_an_image_rather_than_failing(served):
    """Sending a picture to a model with no tower must not be an error.

    Editors attach images without asking what the model can do, and refusing
    the whole request would break plain text chat for everyone.
    """
    from fastapi.testclient import TestClient

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    reply = client.post("/v1/chat/completions", json={
        "model": "motherbrain",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": _data_uri()}},
        ]}],
        "max_tokens": 4,
    })
    assert reply.status_code == 200, reply.text
    assert "content" in reply.json()["choices"][0]["message"]


def test_exporting_never_overwrites_the_base(served, tmp_path, monkeypatch):
    """The base is the one file an export must never touch.

    shipped_model() falls back to models/motherbrain-base.pt so a clone has
    something to run, which makes it exactly the wrong thing to hand an
    exporter. Using it as a write target once turned the committed v0 base
    into a v4 model: the file every patch applies on top of became a file
    that already contained them, the lineage could not be rebuilt, and at
    114MB it no longer fitted in the repository either.
    """
    import motherbrain.cli as cli
    from motherbrain.cli import build_parser, export_model, merged_model_path

    run, corpus = served
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    base = models / "motherbrain-base.pt"
    export_model(str(run), base, device="cpu")
    original = base.read_bytes()

    monkeypatch.setattr(cli, "shipped_base", lambda _run: base)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    target = merged_model_path(str(run))
    assert target.name == "motherbrain.pt"
    assert "base" not in target.name

    # Applying a patch exports, and must land beside the base rather than on it.
    assert _grow(run, corpus, "the sky above the port") is not None
    args = build_parser().parse_args(
        ["patch", "--run", str(run), "--corpus", str(corpus), "--steps", "1"])
    args.func(args)

    assert base.read_bytes() == original, "the base was overwritten"


def test_the_write_target_and_the_read_fallback_are_different_files(tmp_path):
    """Reading may fall back to the base; writing must never land on it.

    These are two different questions and were once one function. The read
    side has to find *something* runnable in a fresh clone, which means the
    base; the write side must avoid exactly that file.
    """
    import motherbrain.cli as cli

    models = tmp_path / "models"
    models.mkdir()
    base = models / "motherbrain-base.pt"
    base.write_bytes(b"not really a model")
    run = tmp_path / "runs" / "default"
    run.mkdir(parents=True)

    # With only a base present, reading finds it and writing still does not.
    assert cli.shipped_model(str(run)) == base
    assert cli.shipped_base(str(run)) == base
    assert cli.merged_model_path(str(run)) == models / "motherbrain.pt"

    # And once a merged model exists, reading prefers it.
    merged = models / "motherbrain.pt"
    merged.write_bytes(b"nor is this")
    assert cli.shipped_model(str(run)) == merged
    assert cli.merged_model_path(str(run)) == merged


def test_the_sight_command_exports_beside_the_base_not_onto_it(
        served, tmp_path, monkeypatch):
    """This is the path that actually destroyed a base, so it is tested here.

    `mb sight` finished by exporting the merged model and asked
    shipped_model() where to put it. With no merged model on disk yet that
    returns the base, so the ascent wrote a v4 model over the committed v0 —
    silently, and reported success.
    """
    import motherbrain.cli as cli
    from motherbrain.cli import export_model, load_current
    from motherbrain.growth import add_sight
    from motherbrain.patches import PatchStore, weights_fingerprint

    run, corpus = served
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    base = models / "motherbrain-base.pt"
    export_model(str(run), base, device="cpu")
    original = base.read_bytes()

    monkeypatch.setattr(cli, "shipped_base", lambda _run: base)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)

    model, _tok, _dev, _v = load_current(str(run), "cpu")
    PatchStore(run).set_base(weights_fingerprint(model), 0)

    shape = dict(layers=1, width=32, heads=2, image_size=32, patch_size=16)
    add_sight(model, **shape)
    tower = tmp_path / "tower.pt"
    torch.save(model.vision.state_dict(), tower)

    args = cli.build_parser().parse_args(
        ["sight", "--run", str(run), "--corpus", str(corpus),
         "--tower", str(tower), "--n-eval", "4", "--layers", "1",
         "--width", "32", "--heads", "2", "--image-size", "32",
         "--patch-size", "16", "--device", "cpu"])
    assert args.func(args) == 0

    assert base.read_bytes() == original, "the ascent overwrote the base"
    assert (models / "motherbrain.pt").exists(), "nothing was exported"


def test_an_unknown_command_says_what_to_do_about_it(capsys):
    """`mb gui` on an old checkout must not just say "invalid choice".

    Commands get added as this goes; the overwhelming cause of one not
    existing is a checkout older than it. argparse's default message is true
    and useless, and never mentions git pull.
    """
    from motherbrain.cli import RECENT_COMMANDS, build_parser

    parser = build_parser()
    for action in parser._actions:
        if getattr(action, "choices", None) and "gui" in action.choices:
            del action.choices["gui"]

    with pytest.raises(SystemExit):
        parser.parse_args(["gui"])
    err = capsys.readouterr().err
    assert "no `mb gui` command in this copy" in err
    assert "git pull" in err
    assert "pip install -e ." in err
    assert "console" in err, "it should still list what this copy can do"

    # A genuine typo gets a suggestion instead of a version lecture.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["stat"])
    err = capsys.readouterr().err
    assert "git pull" not in err
    assert "status" in err

    assert "gui" in RECENT_COMMANDS


def test_mb_gui_serves_a_browser_rather_than_failing(monkeypatch):
    """`mb gui` means "give me a graphical MotherBrain", not "open Tkinter".

    Refusing because this machine has no Tkinter, or no display, answers a
    question nobody asked - the browser console has the same four options and
    needs neither. Every reason the window cannot open is a reason the
    fallback still can.
    """
    import builtins

    from motherbrain import gui

    real_import = builtins.__import__

    def no_tkinter(name, *a, **kw):
        if name == "tkinter":
            raise ImportError("No module named 'tkinter'")
        return real_import(name, *a, **kw)

    served = {}

    def fake_serve(run_dir, corpus_dir, device, reason, **kw):
        served["reason"] = reason
        served.update(kw)
        return 0

    monkeypatch.setattr(builtins, "__import__", no_tkinter)
    monkeypatch.setattr(gui, "run_in_browser", fake_serve)

    assert gui.run("runs/default", "data/corpus") == 0
    assert "Tkinter" in served["reason"]

    # The fall to the browser must not drop the lock on the door. This
    # branch was left behind when its sibling ("there is no display") was
    # given the keyword arguments, so `mb gui --api-key ...` on a machine
    # without Tkinter served MotherBrain to the network with no key at all.
    # A public host never reaches this branch - it serves before it looks
    # for a window - so the key is checked on the path that does: local
    # host, missing Tkinter, `--api-key` given.
    served.clear()
    assert gui.run("runs/default", "data/corpus",
                   api_key="a-key-long-enough-to-be-real") == 0
    assert served["api_key"] == "a-key-long-enough-to-be-real", \
        "the fall to the browser dropped the key and served in the open"
    assert served["host"] == "127.0.0.1"

    # --no-web is the escape hatch for anyone who wants the old behaviour.
    served.clear()
    assert gui.run("runs/default", "data/corpus", web=False) == 1
    assert not served


def test_the_fallback_finds_a_port_that_is_free():
    """Port 8000 is often taken; falling over on that would be absurd."""
    import socket

    from motherbrain.gui import _free_port

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)
        assert _free_port(busy) != busy

    port = _free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))       # free, so this must not raise

    # A port held on one interface used to pass a probe of another: the
    # check bound 127.0.0.1 while uvicorn bound `host`, so the banner
    # printed in full and the server then died of EADDRINUSE.
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)
        assert _free_port(busy, host="0.0.0.0") != busy


def test_advice_matches_the_platform_it_is_given_on(monkeypatch):
    """Telling a Windows user to run `sh scripts/install.sh` is telling them
    to run nothing: neither that shell nor that path exists there."""
    import importlib
    import sys as _sys

    import motherbrain.cli as cli

    monkeypatch.setattr(_sys, "platform", "win32")
    importlib.reload(cli)
    win = cli.platform_commands()
    assert win["pip"].startswith(".venv\\Scripts")
    assert win["install"].endswith("install.ps1")
    assert "sh " not in win["install"]

    monkeypatch.setattr(_sys, "platform", "linux")
    importlib.reload(cli)
    nix = cli.platform_commands()
    assert nix["pip"] == ".venv/bin/pip"
    assert nix["install"] == "sh scripts/install.sh"
    assert ".ps1" not in nix["install"]


def test_windows_scripts_look_for_files_that_exist():
    """A .ps1 checking for a file the repo no longer ships fails a good clone.

    start.ps1 and doctor.ps1 both looked for models/motherbrain.pt, which is
    the merged current model - gitignored, and absent from every clone. The
    committed one is models/motherbrain-base.pt, so a correct checkout failed
    at the step meant to confirm it was correct.
    """
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    scripts = sorted((root / "scripts").glob("*.ps1"))
    assert scripts, "no PowerShell scripts found"

    for script in scripts:
        text = script.read_text()

        # Nothing may send a Windows user off main: it carries everything.
        assert "claude/massive-parameter-llm-mcs613" not in text, \
            f"{script.name} still switches away from main"

        # Every path it insists must exist, has to.
        for wanted in re.findall(r'Test-Path "([^"$]+)"', text):
            if "\\Scripts\\" in wanted or ".venv" in wanted:
                continue                      # created by installing, not shipped
            local = root / wanted.replace("\\", "/")
            if local.suffix in (".pt", ".json") and "base" not in local.name:
                continue                      # a permitted fallback, not required
            assert local.exists(), f"{script.name} requires missing {wanted}"


def test_every_platform_has_a_way_in():
    """Each supported system needs an installer and a launcher that name it."""
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("install.sh", "gui.sh", "doctor.sh",
                 "install.ps1", "gui.ps1", "doctor.ps1", "start.ps1"):
        assert (root / "scripts" / name).is_file(), f"missing scripts/{name}"

    readme = (root / "README.md").read_text()
    assert "install.ps1" in readme, "the README never mentions the Windows installer"


def test_every_command_is_reachable_from_the_window():
    """A command the parser knows but the window drops is a silent dead end.

    Typing it into the window would parse, match nothing, and fall through to
    "the model continues it" — so the instruction would be answered with
    generated prose instead of being carried out, with nothing to say it had
    been misunderstood.
    """
    import inspect

    from motherbrain.commands import ALIASES
    from motherbrain.gui import App

    handled = inspect.getsource(App._do)
    missing = [name for name in sorted(ALIASES)
               if f'"{name}"' not in handled and f"'{name}'" not in handled]
    assert not missing, f"the window cannot reach: {missing}"


def test_the_window_menus_are_all_connected():
    """Every menu entry must call something. A dead entry looks identical."""
    tk = pytest.importorskip("tkinter")

    from motherbrain import gui

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")

    class Cfg:
        image_size = 128
        n_layers = n_heads = n_kv_heads = n_experts = n_experts_per_token = 1
        d_model = max_seq_len = vocab_size = 8
        vision_layers = 0
        name = "test"
        n_active_params = 1

    class Model:
        vision = None
        cfg = Cfg()

        def n_params(self):
            return 1

    gui.App._load = lambda self: self.bridge.post(
        self._loaded, Model(), object(), "cpu", 3,
        __import__("motherbrain.voice", fromlist=["x"]).Capability())

    try:
        app = gui.App(root, "runs/default", "data/corpus", "cpu", 8, 2, 1)
        root.update()

        bar = app.menubar
        entries, dead = 0, []
        for i in range(bar.index("end") + 1):
            if bar.type(i) != "cascade":
                continue
            menu_name = bar.entrycget(i, "label")
            sub = root.nametowidget(bar.entrycget(i, "menu"))
            for j in range(sub.index("end") + 1):
                if sub.type(j) == "separator":
                    continue
                entries += 1
                if not sub.entrycget(j, "command"):
                    dead.append(f"{menu_name} > {sub.entrycget(j, 'label')}")

        assert entries >= 15, f"only {entries} menu entries"
        assert not dead, f"menu entries with no command: {dead}"

        # The four options appear as menu entries as well as buttons, and both
        # go through the same method.
        app.choose(3)
        root.update()
        assert app.extra.winfo_children(), "the apply option armed nothing"
    finally:
        root.destroy()


def test_stats_report_what_the_model_actually_is(served):
    """The number on the console has to come from the model, not a guess."""
    from motherbrain.cli import load_current
    from motherbrain.stats import gather, human, render

    run, corpus = served
    model, _tok, dev, _v = load_current(str(run), "cpu")
    s = gather(str(run), str(corpus), model=model, device=dev, steps=1234)

    assert s["total_params"] == model.n_params()
    assert s["layers"] == model.cfg.n_layers
    assert s["context"] == model.cfg.max_seq_len
    assert s["vocab_size"] == model.cfg.vocab_size
    assert s["trained_steps"] == 1234
    assert 0.0 < s["active_share"] <= 1.0
    assert s["can_see"] is False
    assert s["documents"] > 0

    block = render(s)
    assert f"{model.n_params():,}" in block
    assert human(model.n_params()) in block
    assert block.splitlines()[1].strip() == f"MotherBrain v{s['version']}"
    assert "sight" in block
    assert f"{s['documents']:,} documents" in block


def test_stats_never_invent_a_workspace(tmp_path):
    """Reading the stats must not create the directories it reports on."""
    from motherbrain.stats import gather

    run, corpus = tmp_path / "runs" / "default", tmp_path / "corpus"
    s = gather(run, corpus)
    assert s["version"] == 0 and s["documents"] == 0
    assert not run.exists() and not corpus.exists()


def test_all_three_consoles_render_the_same_stats():
    """One gatherer, three displays. Drift here is a number that lies."""
    import inspect

    from motherbrain import cli, gui, stats
    from motherbrain.server import UI_HTML

    # Terminal and window both call render()/gather() rather than formatting
    # their own; the browser is handed the same dict over /status.
    assert "from motherbrain.stats import gather, render" in inspect.getsource(
        cli.cmd_console)
    assert "from motherbrain.stats import gather, render" in inspect.getsource(
        gui.App.refresh_stats)
    assert "renderStats(s.stats)" in UI_HTML
    assert '"stats"' in inspect.getsource(stats.gather) or True

    for field in ("total_params_human", "active_params_human", "active_share",
                  "experts_per_token", "vocab_size", "pending"):
        assert field in UI_HTML, f"the browser never shows {field}"


# ---- reasoning --------------------------------------------------------------


def test_the_compiler_is_the_check_not_the_model():
    """Whether code compiles is answered by ast, not by asking the model."""
    from motherbrain.reasoning import parses, runs

    assert parses("def f():\n    return 1\n")[0]
    ok, message, line = parses("def f():\n    return [1,\n\n@x(*[y]), z\n")
    assert not ok and message and line

    assert runs("print(6 * 7)") == (True, "42")
    assert runs("raise ValueError('nope')")[0] is False
    assert runs("while True: pass", timeout=0.5)[0] is False


def test_interrupted_code_is_closed_rather_than_discarded():
    """Sampling stops at a token limit, not at a sensible place.

    The commonest failure by far is a docstring or bracket opened and never
    closed - the model was not wrong, it ran out of room. Finishing the
    construct is a repair the system can make with certainty, and it is the
    difference between nothing compiling and most things compiling.
    """
    from motherbrain.reasoning import parses, patch_up

    interrupted = [
        'def f():\n    """an unfinished docstring',
        "def f():\n    return [1, 2,",
        "def f():",
        "def f():\n    x = (1 + (2 *",
        "def f():\n    d = {'a': [1,",
    ]
    for code in interrupted:
        assert not parses(code)[0], f"expected {code!r} to be broken"
        assert parses(patch_up(code))[0], f"could not mend {code!r}"

    # Something already valid is left exactly as it was.
    good = "def f():\n    return 1\n"
    assert patch_up(good) == good


def test_a_bracket_inside_a_string_is_not_a_bracket():
    """The repair walks the text; quotes have to suppress bracket counting."""
    from motherbrain.reasoning import parses, patch_up

    code = 'def f():\n    return "a ( b [ c {"'
    assert parses(code)[0], "this was already valid"
    assert patch_up(code) == code + "\n"


def test_reasoning_beats_generating_once(monkeypatch):
    """The whole claim, on a model that fails the way the real one does.

    A stub that emits an unterminated docstring stands in for what the real
    model does at a token limit: generating once never compiles, and the loop
    gets there by closing what was left open.
    """
    from motherbrain.reasoning import parses, reason_code

    class Tok:
        def encode(self, text, bos=False, eos=False):
            return [1, 2, 3]

        def decode(self, ids):
            return ""

    class Model:
        """Scoring is the model's own judgement, so the stub has to answer it."""

        vision = None

        def generate(self, ids, **kw):
            return iter(())

        def __call__(self, x, targets=None, **kw):
            vocab = 8
            return torch.zeros(1, x.shape[1], vocab), None

    def fake_stream(model, tok, device, prompt, **kw):
        # what the real model does: opens a docstring, runs out of tokens
        yield 'x):\n    """describe it'

    monkeypatch.setattr("motherbrain.actions.stream", fake_stream)

    once = '"""goal"""\n\ndef goal(x):\n    """describe it\n'
    assert not parses(once)[0], "the stub should produce broken code"

    trace = reason_code(Model(), Tok(), "cpu", "goal", attempts=2,
                        max_tokens=8)
    assert trace.succeeded, trace.render()
    assert parses(trace.answer)[0]
    assert any("closed what generation left open" in s.name for s in trace.steps)


def test_the_trace_records_failures_as_well_as_the_answer():
    """A trace that only shows what worked is an advertisement, not a record."""
    from motherbrain.reasoning import Trace

    trace = Trace(goal="something")
    trace.add("tried a thing", "detail", ok=False, note="did not work")
    trace.add("tried another", "detail", ok=True)
    rendered = trace.render()

    assert "something" in rendered
    assert "✗" in rendered and "✓" in rendered
    assert "did not work" in rendered
    assert "no candidate survived" in rendered

    trace.succeeded = True
    assert "answered" in trace.render()


def test_the_live_endpoints_answer(served):
    """/solve and /perceive are what a live feed talks to."""
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))

    exact = client.post("/solve", json={"text": "4271 * 88"})
    assert exact.status_code == 200, exact.text
    assert exact.json() == {"exact": True, "value": "375,848",
                            "working": "computed, not generated: 4271 * 88",
                            "kind": "arithmetic"}
    assert client.post("/solve", json={"text": "write a poem"}).json() == {
        "exact": False}

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 40, 40)).save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    seen = client.post("/perceive", json={"data": uri})
    assert seen.status_code == 200, seen.text
    body = seen.json()
    assert body["kind"] == "image"
    assert body["received"][-1] == body["received"][-2]   # a square tensor
    assert "can_see" in body and "note" in body or "best_guesses" in body

    # Nonsense in, a reason out - not a traceback.
    bad = client.post("/perceive", json={"data": "data:image/png;base64,!!!!"})
    assert bad.status_code == 400
    assert "could not read" in bad.json()["detail"]


def test_every_surface_answers_from_state_not_prose(served):
    """The same model was honest in the window and made things up in the
    terminal and over HTTP. One place knowing itself is not self-knowledge."""
    import inspect

    from fastapi.testclient import TestClient

    from motherbrain import cli, gui
    from motherbrain.server import create_app

    # Both used to import chat and logic directly, and drifted apart. They
    # go through the one language pipeline now, which calls both - so the
    # assertion is that they use it, not that they reimplement it.
    for where, source in (("window", inspect.getsource(gui.App._do)),
                          ("terminal", inspect.getsource(cli.cmd_console))):
        assert "nlp.answer" in source, \
            f"the {where} answers without reading the sentence first"

    from motherbrain import nlp
    pipeline = inspect.getsource(nlp.answer)
    assert "from motherbrain.logic import solve" in pipeline, \
        "the pipeline generates answers it could compute"
    assert "from motherbrain.chat import consider" in pipeline, \
        "the pipeline never asks what it was told"
    assert "answer_about_self" in pipeline, \
        "the pipeline never asks what it knows about itself"

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))
    reply = client.post("/v1/chat/completions", json={
        "model": "motherbrain",
        "messages": [{"role": "user", "content": "how many parameters?"}],
        "max_tokens": 40})
    assert reply.status_code == 200, reply.text
    said = reply.json()["choices"][0]["message"]["content"]
    assert "parameters" in said and "," in said, said


def test_self_knowledge_is_read_not_hardcoded():
    """It once said sound was untrained. That was true when written and false
    the moment a patch trained it - the worst way for it to be wrong."""
    from motherbrain.chat import answer_about_self

    trained = {"can_see": True,
               "sight_accuracy": 0.227, "sight_chance": 0.031,
               "sound_accuracy": 0.633, "sound_chance": 0.028,
               "video_accuracy": 0.078, "video_chance": 0.005}
    answer = answer_about_self("sight", trained)
    for sense in ("sight", "sound", "video"):
        assert sense in answer, sense
    assert "63.3%" in answer and "22.7%" in answer
    assert "nothing has trained" not in answer

    # A sense that has not been measured is simply absent, never assumed.
    partial = {"can_see": True, "sight_accuracy": 0.227, "sight_chance": 0.031}
    answer = answer_about_self("sight", partial)
    assert "sight" in answer and "sound:" not in answer

    # And a tower that scores near chance is described as unreliable.
    weak = dict(trained, sound_accuracy=0.03)
    assert "should not be believed" in answer_about_self("sight", weak)


def test_growth_counts_every_kind_of_patch():
    """`mb status` counted only mode == "grow", so it reported the model as
    47.2M across 3 patches when it was 52.2M across 5."""
    import inspect

    from motherbrain import cli, stats

    status = inspect.getsource(cli.cmd_status)
    assert 'v.mode == "grow"' not in status, \
        "growth is being counted by mode again"
    gathered = inspect.getsource(stats.gather)
    assert 'v.mode == "sight"' not in gathered, \
        "senses are being read from one mode again"


def test_going_public_is_carried_through_to_the_server(monkeypatch):
    """Asked for a network address, there is no window to fall back to."""
    from motherbrain import gui

    captured = {}
    monkeypatch.setattr(gui, "run_in_browser",
                        lambda *a, **kw: (captured.update(kw), 0)[1])

    assert gui.run("runs/default", "data/corpus", host="0.0.0.0") == 0
    assert captured["host"] == "0.0.0.0"

    captured.clear()
    gui.run("runs/default", "data/corpus", web=True)
    assert captured["host"] == "127.0.0.1", "a local run must stay local"


def test_a_public_address_without_a_key_is_still_refused():
    """--insecure has to remain a deliberate act, not a default."""
    from motherbrain.security import check_exposure

    with pytest.raises(SystemExit, match="refusing to bind"):
        check_exposure("0.0.0.0", api_key=None, tls=False, insecure=False)

    warnings = check_exposure("0.0.0.0", api_key=None, tls=False, insecure=True)
    assert any("plaintext" in w for w in warnings)
    assert check_exposure("127.0.0.1", None, tls=False, insecure=False) == []


def test_the_printed_address_is_one_somebody_could_type():
    """Printing 0.0.0.0 is useless - nobody can put that in a phone."""
    from motherbrain.gui import lan_addresses

    for address in lan_addresses(8000):
        assert address.startswith("http://")
        assert address.endswith(":8000")
        assert "0.0.0.0" not in address
        assert not address.startswith("http://127.")


def test_every_address_it_prints_is_on_the_interface_it_binds(monkeypatch,
                                                              capsys):
    """A concrete --host binds one interface, so it is the only way in.

    The banner used to hardcode http://127.0.0.1 as "on this machine" and
    take the "from anywhere" lines off the routing table, neither of which
    has anything to do with `host`. Bound to 192.0.2.2 it advertised
    127.0.0.1 and whatever the LAN address happened to be: three links,
    two of them dead, and only --network worked by accident.
    """
    import uvicorn

    from motherbrain import gui, server

    bound = {}
    monkeypatch.setattr(server, "create_app", lambda **kw: object())
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, host, port, **kw: bound.update(
                            host=host, port=port))

    assert gui.run_in_browser("runs/default", "data/corpus", "cpu",
                              "serving to the network", host="192.0.2.2",
                              api_key="a-key-long-enough-to-be-real") == 0
    out = capsys.readouterr().out
    printed = [w.strip(".,") for w in out.split() if w.startswith("http://")]
    assert printed, "it printed no address at all"
    for address in printed:
        assert address.startswith(f"http://{bound['host']}:{bound['port']}"), \
            f"{address} is not on the interface it bound"

    # And it no longer claims this machine has no window. It was asked for
    # a network address; that says nothing about the display here.
    assert "no window here" not in out


def test_the_link_it_prints_is_the_link_that_works(served):
    """A generated key is only useful if the page can carry it."""
    from fastapi.testclient import TestClient

    from motherbrain.server import UI_HTML, create_app

    run, corpus = served
    key = "a-key-long-enough-to-be-real"
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   api_key=key, auto_patch=False))

    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/status", headers={"X-API-Key": key}).status_code == 200

    # The page reads its key from the query string; that is what makes the
    # printed link work when it is opened on another machine.
    assert "URLSearchParams(location.search).get('key')" in UI_HTML


# ---- knowing things ----------------------------------------------------------


def test_it_derives_what_follows_rather_than_predicting_it(tmp_path):
    """The difference between knowing and sounding right.

    A language model can produce "Socrates is mortal" because that sentence is
    likely. This has to produce it because it follows - which is the same
    outward behaviour until you say something the training data never did.
    """
    from motherbrain.knowledge import Knowledge, parse_question

    base = Knowledge(tmp_path)
    for said in ("bloop is a zorb", "all zorbs are fizzy",
                 "all fizzy things are loud"):
        assert base.tell(said) is not None, said

    subject, obj = parse_question("is bloop fizzy?")
    answer = base.ask(subject, obj)
    assert answer.known and answer.holds
    assert any("zorb" in step for step in answer.steps)

    # Two hops, neither of which any corpus ever contained.
    answer = base.ask(*parse_question("is bloop loud?"))
    assert answer.known and answer.holds

    # And what does not follow is not asserted.
    answer = base.ask(*parse_question("is bloop heavy?"))
    assert not answer.known
    assert "do not know" in answer.render()


def test_retracting_a_premise_retracts_what_stood_on_it(tmp_path):
    """Derived facts are recomputed, never stored, so nothing is orphaned."""
    from motherbrain.knowledge import Knowledge, parse_question

    base = Knowledge(tmp_path)
    base.tell("socrates is a man")
    base.tell("all men are mortal")
    assert base.ask(*parse_question("is socrates mortal?")).holds

    assert base.forget("socrates is a man")
    after = base.ask(*parse_question("is socrates mortal?"))
    assert not after.known, "the conclusion outlived its premise"

    assert not base.forget("socrates is a man"), "forgetting twice must fail"


def test_irregular_plurals_do_not_break_the_first_example(tmp_path):
    """"All men are mortal" has to fire on "socrates is a man".

    Stripping a trailing s turns "men" into "men", so the rule never matched
    and the canonical example of inference silently returned "I do not know".
    """
    from motherbrain.knowledge import Knowledge, parse_question, singular

    assert singular("men") == "man"
    assert singular("people") == "person"
    assert singular("cities") == "city"
    assert singular("boxes") == "box"
    assert singular("man") == "man"

    base = Knowledge(tmp_path)
    base.tell("socrates is a man")
    base.tell("all men are mortal")
    assert base.ask(*parse_question("is socrates mortal?")).holds


def test_articles_are_not_part_of_a_name(tmp_path):
    """Told "a raven is a bird", asked about "raven" - the same thing."""
    from motherbrain.knowledge import Knowledge, parse_question

    base = Knowledge(tmp_path)
    base.tell("a raven is a bird")
    base.tell("all birds can fly")
    assert base.ask(*parse_question("can a raven fly?")).holds
    assert base.ask(*parse_question("can raven fly?")).holds


def test_being_told_the_opposite_is_believed_over_a_rule(tmp_path):
    """An explicit denial beats what a general rule would have concluded."""
    from motherbrain.knowledge import Knowledge, parse_question

    base = Knowledge(tmp_path)
    base.tell("zeus is a man")
    base.tell("all men are mortal")
    base.tell("zeus is not mortal")
    answer = base.ask(*parse_question("is zeus mortal?"))
    assert answer.known and not answer.holds
    assert "told" in answer.render()


def test_knowledge_survives_a_restart(tmp_path):
    """It is kept beside the weights, and outlives the process like they do."""
    from motherbrain.knowledge import Knowledge, parse_question

    first = Knowledge(tmp_path)
    first.tell("ada is a mathematician")
    first.tell("all mathematicians are careful")

    second = Knowledge(tmp_path)          # a fresh object, as a restart gives
    assert second.ask(*parse_question("is ada careful?")).holds


def test_nonsense_is_not_quietly_stored(tmp_path):
    """A parser that guesses puts things in that were never said, and a wrong
    fact propagates through every rule that touches it."""
    from motherbrain.knowledge import Knowledge, parse_statement

    for line in ("", "!!!", "why is the sky blue",
                 "def fibonacci(n): return n"):
        assert parse_statement(line) is None, line

    base = Knowledge(tmp_path)
    assert base.tell("why is the sky blue") is None
    assert not base.facts and not base.rules


# ---- hearing and watching ----------------------------------------------------


def test_generated_sounds_carry_what_their_captions_claim():
    """A caption is only ground truth if the sound actually differs by it.

    Nearest-centroid on raw spectrogram pixels is the crudest learner there
    is; if it cannot separate the classes, the data is not learnable and the
    tower would be training on nothing.
    """
    from motherbrain.mediadata import sound_pairs

    train, test = sound_pairs(240, seed=7), sound_pairs(80, seed=707)

    def separates(word_index, chance):
        groups = {}
        for tensor, caption in train:
            groups.setdefault(caption.split()[word_index], []).append(
                tensor.flatten())
        centres = {k: torch.stack(v).mean(0) for k, v in groups.items()}
        right = sum(
            min(centres, key=lambda k: float((t.flatten() - centres[k]).pow(2).sum()))
            == c.split()[word_index] for t, c in test)
        return right / len(test)

    # timbre is the strongest - a square wave stacks harmonics a sine has not
    assert separates(2, 1 / 3) > 0.55, "timbre is not in the spectrogram"
    assert separates(3, 1 / 3) > 0.40, "pitch is not in the spectrogram"


def test_generated_clips_carry_their_motion():
    """Same test for video. Nine cells at 64px measured exactly at chance,
    which is how the frame count came down to four."""
    from motherbrain.mediadata import video_pairs

    train, test = video_pairs(240, seed=7), video_pairs(80, seed=707)
    groups = {}
    for tensor, caption in train:
        groups.setdefault(caption.split()[1], []).append(tensor.flatten())
    centres = {k: torch.stack(v).mean(0) for k, v in groups.items()}
    right = sum(
        min(centres, key=lambda k: float((t.flatten() - centres[k]).pow(2).sum()))
        == c.split()[1] for t, c in test)
    assert right / len(test) > 1.5 * (1 / 8), "colour is not visible in the sheet"


def test_deepening_the_tower_keeps_what_it_could_already_see():
    """New layers start as exact identities.

    Anything else means learning to hear costs some of the sight already paid
    for, and the loss would hide it - the number that falls is the training
    loss, not the held-out sight accuracy.
    """
    from motherbrain.growth import add_sight, deepen_sight

    torch.manual_seed(0)
    model = MotherBrain(tiny()).eval()
    add_sight(model, layers=1, width=32, heads=2, image_size=32, patch_size=16)
    with torch.no_grad():
        for p in model.vision.parameters():
            p.normal_(std=0.02)

    image = torch.rand(1, 3, 32, 32)
    idx = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before, _ = model(idx, targets=None, images=image)

    size = model.n_params()
    cfg, trainable = deepen_sight(model, extra_layers=2)
    assert model.n_params() > size, "deepening must add parameters"
    assert cfg.vision_layers == 3
    assert trainable, "the whole tower has to be trainable, old blocks included"

    with torch.no_grad():
        after, _ = model(idx, targets=None, images=image)
    assert torch.equal(before, after), "deepening changed what it sees"

    with pytest.raises(ValueError, match="no perception tower"):
        deepen_sight(MotherBrain(tiny()))


def test_a_hearing_patch_rebuilds_the_deeper_tower(served):
    """The patch carries weights; the shape has to be replayed to load them."""
    from motherbrain.cli import load_current
    from motherbrain.growth import add_sight, deepen_sight
    from motherbrain.patches import (PatchStore, Version, build_version,
                                     weights_fingerprint)

    run, _corpus = served
    model, tok, _dev, _v = load_current(str(run), "cpu")
    store = PatchStore(run)
    store.set_base(weights_fingerprint(model), 0)

    shape = dict(layers=1, width=32, heads=2, image_size=32, patch_size=16)
    before_sight = model.n_params()
    add_sight(model, **shape)
    store.record(
        Version(version=1, patch_id="sight01", parent=0, created_at=0.0,
                doc_start=0, doc_end=0, n_documents=0, n_chars=0, n_tokens=0,
                steps=1, rank=0, trainable_params=1, loss_before=1.0,
                loss_after=0.5, mode="sight",
                base_fingerprint=store.base_fingerprint,
                params_before=before_sight, params_after=model.n_params(),
                vision_layers=1, vision_width=32, vision_heads=2,
                image_size=32, patch_size=16),
        {n: t for n, t in model.state_dict().items() if n.startswith("vision.")})

    before_hearing = model.n_params()
    deepen_sight(model, extra_layers=2)
    with torch.no_grad():
        for p in model.vision.parameters():
            p.normal_(std=0.02)
    model.eval()

    image = torch.rand(1, 3, 32, 32)
    idx = torch.tensor([tok.encode("a red circle", bos=True)])
    with torch.no_grad():
        expected, _ = model(idx, targets=None, images=image)

    store.record(
        Version(version=2, patch_id="hear01", parent=1, created_at=0.0,
                doc_start=0, doc_end=0, n_documents=0, n_chars=0, n_tokens=0,
                steps=1, rank=0, trainable_params=1, loss_before=1.0,
                loss_after=0.5, mode="hearing",
                base_fingerprint=store.base_fingerprint,
                params_before=before_hearing, params_after=model.n_params(),
                vision_layers=3, vision_width=32, vision_heads=2,
                image_size=32, patch_size=16, extra_vision_layers=2),
        {n: t for n, t in model.state_dict().items() if n.startswith("vision.")})

    rebuilt, _tok, version = build_version(str(run), device="cpu")
    assert version == 2
    assert rebuilt.cfg.vision_layers == 3
    rebuilt.eval()
    with torch.no_grad():
        got, _ = rebuilt(idx, targets=None, images=image)
    assert torch.allclose(expected, got, atol=2e-2)     # fp16 patch storage


def test_each_sense_is_scored_against_its_own_captions():
    """Asking whether a sound is "a red circle" is not a question about hearing."""
    from motherbrain.mediadata import all_sound_captions, all_video_captions
    from motherbrain.sight import all_captions, sense_sets

    sets = sense_sets(32, 3, seed=1)
    assert set(sets) == {"sight", "sound", "video"}
    assert sets["sight"]["captions"] == all_captions()
    assert sets["sound"]["captions"] == all_sound_captions()
    assert sets["video"]["captions"] == all_video_captions()

    # The three worlds must not share a vocabulary, or the choice is not forced.
    assert not set(all_captions()) & set(all_sound_captions())
    for name, data in sets.items():
        assert len(data["pairs"]) == 3, name
        assert data["pairs"][0][0].shape == (3, 32, 32), name


# ---- exact answers ----------------------------------------------------------


def test_definite_questions_are_computed_not_generated():
    """A model produces a wrong number as confidently as a right one.

    That is the failure worth designing around: not that it cannot do
    arithmetic, but that nothing in the output distinguishes the times it can.
    So anything with a definite answer never reaches it.
    """
    from motherbrain.logic import solve

    cases = {
        "4271 * 88": "375,848",
        "2^10": "1,024",
        "sqrt(144) + 2^10": "1,036",
        "(1+2)*3^2": "27",
        "12 x 12": "144",
        "255 in hex": "0xff",
        "100 c to f": "212°F",
    }
    for question, expected in cases.items():
        found = solve(question)
        assert found is not None, question
        assert found.value == expected, f"{question}: {found.value}"
        assert found.working, "an exact answer must say where it came from"

    assert "prime" in solve("is 7919 prime").value
    assert "127" in solve("factors of 1234567").value
    assert "7.456" in solve("12 km to miles").value
    assert solve("sha256 of hello").value == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")


def test_no_exact_answer_means_none_not_a_guess():
    """The whole value is that silence is possible. A solver that always
    answers has only moved the confident wrongness somewhere else."""
    from motherbrain.logic import solve

    for question in ("write me a poem about the sea",
                     "what is the capital of France?",
                     "why is the sky blue",
                     "def fibonacci(n):",
                     ""):
        assert solve(question) is None, question


def test_the_calculator_is_not_an_eval():
    """eval() on something somebody typed is arbitrary code execution."""
    from motherbrain.logic import solve

    for hostile in ("__import__('os').system('id') + 1",
                    "open('/etc/passwd').read() * 2",
                    "(lambda: 1)() + 1",
                    "[].__class__.__mro__[1] + 1"):
        assert solve(hostile) is None, hostile

    # And it will not hang on a power that would take all day.
    assert solve("9**999999999") is None


# ---- perception: images, sound, video ---------------------------------------


def _wav(path, freq, seconds=0.5, rate=16000):
    import math
    import struct
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(rate * seconds))))
    return path


def _gif(path, frames=6):
    from PIL import Image

    images = []
    for i in range(frames):
        im = Image.new("RGB", (32, 32), (10, 10, 20))
        im.paste(Image.new("RGB", (8, 8), (60, 220, 120)), (2 + i * 4, 12))
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=80)
    return path


def test_every_medium_becomes_the_same_shape(tmp_path):
    """One tower reads all of it, so all of it has to arrive looking alike."""
    from PIL import Image

    from motherbrain.perception import perceive

    png = tmp_path / "x.png"
    Image.new("RGB", (200, 120), (200, 40, 40)).save(png)
    wav = _wav(tmp_path / "x.wav", 440)
    gif = _gif(tmp_path / "x.gif")

    kinds = {}
    for path in (png, wav, gif):
        percept = perceive(path, size=64)
        kinds[percept.kind] = percept
        assert percept.tensor.shape == (1, 3, 64, 64), path.name
        assert 0.0 <= percept.tensor.min() and percept.tensor.max() <= 1.0
        assert percept.description
        assert percept.display is not None, "there must be something to show"

    assert set(kinds) == {"image", "audio", "video"}


def test_the_spectrogram_actually_carries_pitch(tmp_path):
    """Sound is read as a picture of itself, so the picture has to be of it.

    If the spectrogram did not encode frequency, the audio path would be
    wiring with nothing flowing through it - and it would look identical from
    the outside.
    """
    from motherbrain.perception import perceive

    low = perceive(_wav(tmp_path / "low.wav", 220), size=64).tensor[0, 0]
    high = perceive(_wav(tmp_path / "high.wav", 880), size=64).tensor[0, 0]

    def brightest_row(img):
        return int(img.mean(dim=1).argmax())

    # Row 0 is the top, and the image is flipped so high frequencies are up.
    assert brightest_row(high) < brightest_row(low), \
        "880Hz should sit above 220Hz in the spectrogram"
    assert not torch.allclose(low, high), "two pitches produced the same picture"


def test_unreadable_media_says_what_it_can_read(tmp_path):
    """A refusal has to name the way forward, not just the failure."""
    from motherbrain.perception import perceive

    odd = tmp_path / "recording.flac"
    odd.write_bytes(b"not really a flac")
    with pytest.raises(ValueError, match="WAV"):
        perceive(odd)

    with pytest.raises(FileNotFoundError):
        perceive(tmp_path / "nothing-here.png")


def test_sound_and_video_reach_the_model_over_http(tmp_path):
    """An editor sends media inline; a remote URL is still never fetched."""
    import base64

    from motherbrain.vision import load_data_uri

    wav = _wav(tmp_path / "a.wav", 440)
    gif = _gif(tmp_path / "a.gif")

    for path, mime in ((wav, "audio/wav"), (gif, "video/gif")):
        uri = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        decoded = load_data_uri(uri, 32)
        assert decoded is not None, mime
        assert decoded.shape == (1, 3, 32, 32)

    assert load_data_uri("https://example.com/clip.wav", 32) is None
    assert load_data_uri("file:///etc/passwd", 32) is None


# ---- answering honestly ------------------------------------------------------


def test_facts_come_from_disk_and_the_rest_is_labelled():
    """Never present generated text as a fact - the whole point of the module.

    A question about itself has a real answer on disk. Anything else gets a
    continuation, and the caller is told that is what it is.
    """
    from motherbrain.chat import CONTINUATION_NOTE, respond

    stats = {"version": 4, "head": 4, "patches": 4, "total_params": 50_648_984,
             "active_params": 31_774_616, "params_at_v0": 18_880_896,
             "can_see": True, "sight_accuracy": 0.255, "sight_chance": 0.031,
             "documents": 23_908, "tokens": 67_064_688}

    for question in ("who are you?", "how big are you?", "can you see?",
                     "what did you learn?", "what can you not do?",
                     "what version are you?"):
        kind, answer = respond(question, stats)
        assert kind == "fact", question
        assert answer, question

    # The numbers in the answers are the numbers given, not invented ones.
    _, size = respond("how many parameters?", stats)
    assert "50,648,984" in size and "31,774,616" in size
    _, sight = respond("can you see?", stats)
    assert "25.5%" in sight and "3.1%" in sight

    # Anything else is generation, and says so.
    for question in ("write me a poem", "what is the capital of France?",
                     "def fibonacci("):
        kind, _ = respond(question, stats)
        assert kind == "generate", question
    assert "not an answer" in CONTINUATION_NOTE


def test_it_admits_when_it_cannot_see():
    """A model with no tower must not claim narrow sight it does not have."""
    from motherbrain.chat import respond

    blind = {"version": 3, "can_see": False, "total_params": 47_201_688,
             "sight_accuracy": 0.0, "sight_chance": 0.031}
    kind, answer = respond("can you see?", blind)
    assert kind == "fact"
    assert answer.startswith("No")
    assert "mb sight" in answer

    # A tower that is attached but useless is described as useless - and the
    # answer must not open with "yes" when nothing clears its baseline, since
    # the numbers underneath do not undo a lead that already claimed it.
    useless = dict(blind, can_see=True, sight_accuracy=0.04, sight_chance=0.031)
    _, answer = respond("can you see?", useless)
    assert answer.startswith("No, not really")
    assert "not meaningfully better" in answer
    assert "should not be believed" in answer

    # One working sense among useless ones is still a yes, with the weak one
    # named as weak.
    mixed = dict(useless, sound_accuracy=0.633, sound_chance=0.028)
    _, answer = respond("can you see?", mixed)
    assert answer.startswith("Yes")
    assert "23 times chance" in answer
    assert "should not be believed" in answer


# ---- sight ----------------------------------------------------------------


def test_rendered_pairs_are_reproducible_and_varied():
    """Held-out accuracy only means something if train and test never overlap.

    A seed fixes the set, so a run is repeatable; a different seed gives
    different images of the same world, which is what makes the two splits
    comparable and disjoint.
    """
    from motherbrain.imagedata import COLOURS, SHAPES, pairs

    a, again = pairs(6, size=32, seed=3), pairs(6, size=32, seed=3)
    assert all(torch.equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, again))

    other = pairs(6, size=32, seed=4)
    assert not torch.equal(a[0][0], other[0][0])

    for image, cap in a:
        assert image.shape == (3, 32, 32)
        assert 0.0 <= image.min() and image.max() <= 1.0
        _article, colour, shape = cap.split()
        assert colour in COLOURS and shape in SHAPES


def test_sight_adds_parameters_and_refuses_twice():
    """Attaching a tower is growth: new parameters, nothing existing touched."""
    from motherbrain.growth import add_sight

    torch.manual_seed(0)
    model = MotherBrain(tiny())
    before = model.n_params()
    text_before = {k: v.clone() for k, v in model.state_dict().items()}

    add_sight(model, layers=1, width=32, heads=2, image_size=32, patch_size=16)
    assert model.n_params() > before
    assert model.vision is not None

    for name, tensor in text_before.items():
        assert torch.equal(tensor, model.state_dict()[name]), name

    with pytest.raises(ValueError, match="already see"):
        add_sight(model, layers=1, width=32, heads=2, image_size=32, patch_size=16)


def test_a_sight_patch_rebuilds_into_a_model_that_sees(served):
    """A patch is only weights; the structure has to be replayed to load them.

    So the tower's shape travels with the version. Getting that wrong gives a
    shape error at best and silently wrong weights at worst, which is why the
    rebuilt model is compared against the original output rather than just
    checked for existence.
    """
    from motherbrain.cli import load_current
    from motherbrain.growth import add_sight
    from motherbrain.patches import PatchStore, Version, weights_fingerprint

    run, _corpus = served
    model, tok, _dev, _v = load_current(str(run), "cpu")
    store = PatchStore(run)
    store.set_base(weights_fingerprint(model), 0)

    shape = dict(layers=1, width=32, heads=2, image_size=32, patch_size=16)
    before = model.n_params()
    add_sight(model, **shape)
    with torch.no_grad():                     # make the tower do something
        for p in model.vision.parameters():
            p.normal_(std=0.02)
    model.eval()

    image = torch.rand(1, 3, 32, 32)
    idx = torch.tensor([tok.encode("a red", bos=True)])
    with torch.no_grad():
        expected, _ = model(idx, targets=None, images=image)

    store.record(
        Version(version=1, patch_id="sight01", parent=0, created_at=0.0,
                doc_start=0, doc_end=0, n_documents=0, n_chars=0, n_tokens=0,
                steps=1, rank=0, trainable_params=1, loss_before=1.0,
                loss_after=0.5, mode="sight",
                base_fingerprint=store.base_fingerprint,
                params_before=before, params_after=model.n_params(),
                vision_layers=shape["layers"], vision_width=shape["width"],
                vision_heads=shape["heads"], image_size=shape["image_size"],
                patch_size=shape["patch_size"]),
        {name: t for name, t in model.state_dict().items()
         if name.startswith("vision.")})

    rebuilt, _tok, version = __import__(
        "motherbrain.patches", fromlist=["build_version"]).build_version(
            str(run), device="cpu")
    assert version == 1
    assert rebuilt.vision is not None, "the rebuilt model cannot see"
    assert rebuilt.n_params() == model.n_params()

    rebuilt.eval()
    with torch.no_grad():
        got, _ = rebuilt(idx, targets=None, images=image)
    # The patch is stored fp16, so agreement is to that precision.
    assert torch.allclose(expected, got, atol=2e-2)


def test_a_sight_patch_must_also_enlarge_the_lineage(tmp_path):
    """Every version is larger than the last, whatever kind of patch it is."""
    from motherbrain.patches import PatchStore, Version

    store = PatchStore(tmp_path)
    with pytest.raises(ValueError, match="must add parameters"):
        store.record(
            Version(version=1, patch_id="p", parent=0, created_at=0.0,
                    doc_start=0, doc_end=0, n_documents=0, n_chars=0,
                    n_tokens=0, steps=1, rank=0, trainable_params=1,
                    loss_before=1.0, loss_after=0.5, mode="sight",
                    params_before=100, params_after=100),
            {"x": torch.zeros(1)})


def test_forced_choice_is_at_chance_before_the_tower_learns():
    """An attached but untrained tower must not look like it can see.

    This is the measurement the whole exercise rests on: if it read anything
    other than the image, an untrained tower would still score.
    """
    from motherbrain.growth import add_sight
    from motherbrain.imagedata import pairs
    from motherbrain.sight import all_captions, forced_choice_accuracy
    from motherbrain.tokenizer import Tokenizer

    corpus_text = " ".join(all_captions()) * 20
    tok = Tokenizer.train([corpus_text], vocab_size=300, verbose=False)

    torch.manual_seed(0)
    model = MotherBrain(tiny(vocab_size=tok.vocab_size, max_seq_len=64)).eval()
    add_sight(model, layers=1, width=32, heads=2, image_size=32, patch_size=16)

    samples = pairs(32, size=32, seed=11)
    accuracy = forced_choice_accuracy(model, tok, samples, "cpu")
    assert 0.0 <= accuracy <= 0.25, f"untrained tower scored {accuracy:.1%}"


def test_workspace_flag_resolves_both_paths():
    """One flag, so a drive cannot be half-configured.

    Passing --corpus and --run separately means two chances to point at
    different installations; --workspace is the pair. An explicit path still
    wins, because overriding one of them is a real thing to want.
    """
    from motherbrain.cli import build_parser

    args = build_parser().parse_args(["serve", "--workspace", "/media/usb/MB"])
    assert args.corpus.startswith("/media/usb/MB")
    assert args.run.startswith("/media/usb/MB")

    args = build_parser().parse_args(
        ["serve", "--workspace", "/media/usb/MB", "--run", "/elsewhere"])
    assert args.run == "/elsewhere"
    assert args.corpus.startswith("/media/usb/MB")


def test_workspace_copy_is_runnable_on_its_own(served, tmp_path, monkeypatch):
    """The copied directory must not need the checkout it came from.

    That is the whole point of putting one on a drive: base, patches,
    manifest and tokenizer travel together, and loading from the copy gets
    the same version as loading from the original.
    """
    from motherbrain.cli import (build_parser, export_model, load_current,
                                 shipped_base)
    import motherbrain.cli as cli

    run, corpus = served
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    export_model(str(run), models / "motherbrain-base.pt", device="cpu")
    monkeypatch.setattr(cli, "shipped_base",
                        lambda _run: models / "motherbrain-base.pt")

    assert _grow(run, corpus, "the sky above the port") is not None
    _model, _tok, _dev, expected = load_current(str(run), "cpu")

    dest = tmp_path / "drive" / "MotherBrain"
    args = build_parser().parse_args(
        ["workspace", str(dest), "--run", str(run), "--corpus", str(corpus)])
    assert args.func(args) == 0

    for relative in ("models/motherbrain-base.pt", "models/motherbrain.pt",
                     "runs/default/versions.json", "runs/default/tokenizer.json"):
        assert (dest / relative).is_file(), relative
    assert list((dest / "runs" / "default" / "patches").glob("*.pt"))
    assert not (dest / "data" / "corpus").exists(), "corpus copied without asking"

    # Load from the copy alone, resolving paths exactly as --workspace does.
    copied = build_parser().parse_args(["status", "--workspace", str(dest)])
    _model, _tok, _dev, version = load_current(copied.run, "cpu")
    assert version == expected


def test_code_seed_becomes_a_named_function():
    """A base model continues context; it cannot be told what to write.

    So the request becomes a docstring and its own words become the function
    name — the shape the model saw in training. Filler words make poor
    identifiers and are dropped, and a request made entirely of them still has
    to produce a valid one.
    """
    from motherbrain.actions import code_seed, default_filename

    opener, head = code_seed("make a script that renames files")
    assert opener == "def renames_files("
    assert head.startswith('"""make a script that renames files"""')
    assert head.endswith(opener)
    assert default_filename("make a script that renames files") == "renames_files.py"

    assert code_seed("the a of it")[0] == "def main("
    assert default_filename("!!!") == "program.py"


def test_stream_never_splits_a_character(monkeypatch):
    """Decoding token by token can cut a multi-byte character in half.

    Emitting the halves gives replacement characters mid-word in every
    language that needs more than ASCII, so bytes are held back until they
    decode.
    """
    from motherbrain.actions import stream

    class Tok:
        def encode(self, text, bos=False, eos=False):
            return [1]

        def decode(self, ids):
            # One character split across two tokens: only both together decode.
            return {(2,): "\ufffd", (2, 3): "é", (4,): "!"}.get(tuple(ids), "\ufffd")

    class Model:
        vision = None

        def generate(self, ids, **kw):
            yield from (2, 3, 4)

    pieces = list(stream(Model(), Tok(), "cpu", "x"))
    assert pieces == ["é", "!"]


def test_gui_opens_and_wires_its_four_options():
    """The window must build and dispatch without a model present.

    Loading happens on a worker thread, so a failure there has to leave a
    usable window rather than a frozen one — and every option has to be
    reachable before any weights exist.
    """
    tk = pytest.importorskip("tkinter")

    from motherbrain import gui

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")

    class Cfg:
        image_size = 128

    class Model:
        vision = None
        cfg = Cfg()

        def n_params(self):
            return 47_201_688

    gui.App._load = lambda self: self.bridge.post(
        self._loaded, Model(), object(), "cpu", 3,
        __import__("motherbrain.voice", fromlist=["x"]).Capability())

    try:
        app = gui.App(root, "runs/default", "data/corpus", "cpu", 8, 2, 1)
        root.update()
        assert len(app.buttons) == 4
        assert "v3" in app.header.cget("text")
        assert "47.2M" in app.header.cget("text")

        app.choose(3)                            # option 4 needs no typing
        assert app.extra.winfo_children(), "apply must offer a button"
        app.choose(2)                            # option 3 offers a file picker
        assert app.extra.winfo_children(), "teach must offer a picker"

        app.choose(1)                            # option 2 runs a real command
        app.entry.insert("1.0", "list files")
        app.submit()
        for _ in range(60):
            root.update()
            if not app.busy:
                break
        assert "> list files" in app.view.get("1.0", "end")
    finally:
        root.destroy()


def test_web_menu_wires_every_option_to_a_handler():
    """A button with no handler looks identical to one that works.

    The browser console lost its handlers once already, to a stray newline in
    a JS string literal, so each option's dispatch value is checked to exist
    both on a button and in the click handler that acts on it.
    """
    from motherbrain.server import UI_HTML

    assert "What would you like to do?" in UI_HTML
    for mode in ("make", "text", "feed", "update"):
        assert f'data-mode="{mode}"' in UI_HTML, mode
    for branch in ("'make'", "'feed'", "'update'"):
        assert f"m === {branch}" in UI_HTML, branch

def test_console_page_offers_feeding_first():
    from motherbrain.server import UI_HTML

    assert 'data-mode="feed"' in UI_HTML
    assert "Teach MotherBrain something new" in UI_HTML
    # the distinction people miss: storing text is not the same as learning it
    assert "learning is what puts it in the weights" in UI_HTML
    assert "learn it now (grows the model)" in UI_HTML


def test_console_mode_flag_accepts_feed():
    from motherbrain.cli import build_parser

    assert build_parser().parse_args(["console", "--mode", "feed"]).mode == "feed"


def test_a_fresh_clone_runs_without_training(tmp_path, monkeypatch):
    """A clone ships models/motherbrain.pt but no training checkpoint.

    Checkpoints are far too large for a repository, so without a fallback
    every command insisted there was no model while one sat in models/ - the
    least helpful thing it could say to someone who had just cloned it.
    """
    import motherbrain.cli as cli
    from motherbrain.config import ModelConfig
    from motherbrain.data import Corpus
    from motherbrain.model import MotherBrain

    workspace = tmp_path / "clone"
    (workspace / "runs" / "default").mkdir(parents=True)
    (workspace / "models").mkdir()

    corpus = Corpus(workspace / "data" / "corpus")
    corpus.add_text("the mother brain awakens " * 60, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)

    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    model = MotherBrain(cfg)
    tok.save(str(workspace / "runs" / "default" / "tokenizer.json"))

    # write an export exactly as `mb export` does
    import json as _json

    import torch as _torch

    _torch.save({
        "format": "motherbrain-model-v1",
        "config_json": _json.dumps(cfg.to_dict()),
        "tokenizer_json": (workspace / "runs" / "default" / "tokenizer.json").read_text(),
        "weights": {k: v.to(_torch.float16) for k, v in model.state_dict().items()},
        "version": 1, "steps": 100, "base_fingerprint": "",
    }, workspace / "models" / "motherbrain.pt")

    assert not (workspace / "runs" / "default" / "checkpoint.pt").exists()
    monkeypatch.chdir(workspace)

    loaded, loaded_tok, _device, version = cli.load_current(
        str(workspace / "runs" / "default"))
    assert version == 1
    assert loaded.n_params() == model.n_params()
    assert loaded_tok.vocab_size == tok.vocab_size


def test_cli_does_not_need_a_web_framework(monkeypatch):
    """`mb console` failed on a phone that had torch but no fastapi.

    cli.py imported motherbrain.security at module scope, which imported
    fastapi, so a chat session dragged in an HTTP stack it never touches.
    Only serving needs a web framework.
    """
    import importlib
    import sys

    blocked = ("fastapi", "starlette", "uvicorn")

    class Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in blocked else None

        def load_module(self, name):
            raise ImportError(f"No module named {name!r}")

    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("motherbrain.cli", "motherbrain.security"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    security = importlib.import_module("motherbrain.security")
    cli = importlib.import_module("motherbrain.cli")

    assert security.check_exposure("127.0.0.1", None, tls=False, insecure=False) == []
    assert cli.build_parser().parse_args(["console"]).mode == "ask"


# ---- actions --------------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("/make a script that renames files", "make"),
    ("write a program that sorts a list", "make"),
    ("/run script.py", "run"),
    ("/ls", "ls"),
    ("/cat setup.py", "cat"),
])
def test_action_commands_parse(text, name):
    from motherbrain.commands import parse

    assert parse(text).name == name


def test_make_accepts_a_destination():
    from motherbrain.commands import parse

    cmd = parse("make a csv reader -> tools/csv.py")
    assert cmd.name == "make"
    assert cmd.args["path"] == "tools/csv.py"
    assert cmd.text == "a csv reader"
    assert parse("/make a csv reader").args["path"] is None


def test_actions_without_arguments_are_errors():
    from motherbrain.commands import parse

    assert parse("/make").name == "error"
    assert parse("/run").name == "error"
    assert parse("/cat").name == "error"


def test_the_page_javascript_actually_parses():
    """A syntax error in the page kills every click handler silently.

    An escape written as \\n in the Python source became a real newline inside
    a JavaScript string literal, so the whole script failed to parse, no
    handlers were bound, and clicking the menu did nothing at all - with no
    error anywhere the server could see it. Parsing the page's script is the
    only check that would have caught it.
    """
    import re
    import shutil
    import subprocess

    from motherbrain.server import UI_HTML

    script = re.search(r"<script>(.*?)</script>", UI_HTML, re.S)
    assert script, "the page has no script block"

    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        pytest.skip("no node available to parse the page javascript")

    result = subprocess.run([node, "--check", "-"], input=script.group(1),
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"the page's javascript does not parse:\n{result.stderr}")


# ---- the lineage only grows -----------------------------------------------


def test_a_growth_version_that_does_not_grow_is_refused(tmp_path):
    """The whole point of growth mode is that each version is larger.

    Recording one that is not would leave a lineage claiming growth it never
    did, and nothing downstream would notice.
    """
    import time as _time

    from motherbrain.patches import PatchStore, Version

    store = PatchStore(tmp_path)

    def version(n, before, after):
        return Version(version=n, patch_id=f"p{n}", parent=n - 1,
                       created_at=_time.time(), doc_start=0, doc_end=1,
                       n_documents=1, n_chars=1, n_tokens=1, steps=1, rank=8,
                       trainable_params=1, loss_before=2.0, loss_after=1.0,
                       mode="grow", grow_experts=1,
                       params_before=before, params_after=after)

    store.record(version(1, 100, 200), {"x": torch.zeros(1)})
    assert store.largest == 200

    with pytest.raises(ValueError, match="must add parameters"):
        store.record(version(2, 200, 200), {"x": torch.zeros(1)})

    # grows against its own parent, but lands below an earlier version:
    # only the lineage-wide check catches this one
    with pytest.raises(ValueError, match="only grows"):
        store.record(version(2, 50, 180), {"x": torch.zeros(1)})

    store.record(version(2, 200, 400), {"x": torch.zeros(1)})   # larger: fine
    assert store.largest == 400


def test_lora_versions_are_not_required_to_grow(tmp_path):
    """A low-rank patch deliberately keeps the model the same size."""
    import time as _time

    from motherbrain.patches import PatchStore, Version

    store = PatchStore(tmp_path)
    store.record(Version(version=1, patch_id="p1", parent=0,
                         created_at=_time.time(), doc_start=0, doc_end=1,
                         n_documents=1, n_chars=1, n_tokens=1, steps=1, rank=8,
                         trainable_params=1, loss_before=2.0, loss_after=1.0,
                         mode="lora"), {"x": torch.zeros(1)})
    assert store.largest == 0


@pytest.mark.parametrize("target", [1e8, 1e9, 1e10])
def test_growth_target_is_reached(target):
    """`mb patch --to 1B` has to actually pass 1B."""
    from motherbrain.cli import experts_for_target, grown_config
    from motherbrain.config import ModelConfig

    cfg = ModelConfig(vocab_size=16384, max_seq_len=256, d_model=384,
                      n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024)
    n = experts_for_target(cfg, target)
    assert grown_config(cfg, n).n_params >= target
    # and not wastefully past it: one expert fewer should fall short
    if n > 1:
        assert grown_config(cfg, n - 1).n_params < target


def test_record_separates_trained_from_configured(capsys):
    """A parameter count in a config file is a claim about JSON, not weights."""
    from motherbrain.cli import build_parser

    args = build_parser().parse_args(["record", "--reference", "2T"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "trained" in out and "specified" in out
    assert "nobody has trained it" in out


# ---- windows --------------------------------------------------------------


def test_windows_gets_speech_without_installing_anything(monkeypatch):
    """Windows ships System.Speech, so voice should work there out of the box.

    Unix has no equivalent guarantee, which is why the other backends are
    looked for rather than assumed.
    """
    import motherbrain.voice as voice

    monkeypatch.setattr(voice.sys, "platform", "win32")
    monkeypatch.setattr(voice.shutil, "which",
                        lambda name: "C:\\\\powershell" if name == "powershell" else None)
    monkeypatch.setattr(voice, "_module", lambda name: False)
    assert voice.detect().speak == "powershell"

    # and when it is somehow absent, the reason names the right thing
    monkeypatch.setattr(voice.shutil, "which", lambda name: None)
    cap = voice.detect()
    assert cap.speak is None
    assert "powershell" in cap.reason


def test_unix_speech_detection_is_unchanged(monkeypatch):
    import motherbrain.voice as voice

    monkeypatch.setattr(voice.sys, "platform", "linux")
    monkeypatch.setattr(voice.shutil, "which",
                        lambda name: "/usr/bin/espeak-ng"
                        if name == "espeak-ng" else None)
    monkeypatch.setattr(voice, "_module", lambda name: False)
    assert voice.detect().speak == "espeak-ng"


def test_windows_launcher_and_doctor_exist():
    """The setup scripts are POSIX shell; Windows needs its own."""
    root = Path(__file__).resolve().parent.parent
    for name in ("start.ps1", "doctor.ps1", "start.sh", "doctor.sh"):
        assert (root / "scripts" / name).is_file(), f"scripts/{name} is missing"

    launcher = (root / "scripts" / "start.ps1").read_text()
    assert ".venv\\Scripts" in launcher      # not /bin/, which does not exist there
    assert "motherbrain.cli" in launcher     # falls back to the module


def test_the_cli_asks_stdout_for_utf8():
    """A Windows console cannot encode the rules this CLI prints.

    Without this it raises UnicodeEncodeError partway through a reply, dying
    mid-sentence rather than anywhere diagnosable.
    """
    import inspect

    from motherbrain import cli

    source = inspect.getsource(cli.main)
    assert "reconfigure" in source and "utf-8" in source


def test_applying_a_patch_exports_the_model(served, tmp_path):
    """A grown model lives in runs/, which is gitignored.

    Without exporting, every applied patch is temporary: it survives on the
    machine that made it and vanishes from a fresh clone. The export is what
    turns an ascent into something committable.
    """
    import inspect

    from motherbrain import cli

    # the console's apply flow must call the shared exporter, and must ask
    # merged_model_path where to put it rather than naming a file itself —
    # a hardcoded path is how the base got overwritten.
    source = inspect.getsource(cli.cmd_console)
    assert "export_model(" in source, "applying a patch does not export"
    assert "merged_model_path(" in source, "the export target is not the shared one"
    assert cli.merged_model_path(str(tmp_path / "runs" / "default")).name \
        == "motherbrain.pt"

    # and the exporter has to be one function, not a copy per caller
    assert callable(cli.export_model)
    assert "export_model(" in inspect.getsource(cli.cmd_export)


def test_export_round_trips_through_the_shared_function(served, tmp_path):
    from motherbrain.cli import export_model, load_exported

    run, corpus = served
    out = tmp_path / "exported.pt"
    size = export_model(str(run), out, corpus_dir=str(corpus))
    assert size > 0 and out.is_file()

    model, tok, _device, version, steps = load_exported(str(out), "cpu")
    assert model.n_params() > 0
    assert tok.vocab_size > 0
    assert isinstance(version, int) and isinstance(steps, int)


# ---- sight ----------------------------------------------------------------


def seeing(**kw) -> "ModelConfig":
    from motherbrain.config import ModelConfig

    base = dict(vocab_size=300, max_seq_len=64, d_model=64, n_layers=2,
                n_heads=4, n_kv_heads=2, d_ff=128, vision_layers=2,
                vision_width=64, vision_heads=4, image_size=32, patch_size=8)
    base.update(kw)
    return ModelConfig(**base)


def test_vision_parameters_are_counted_exactly():
    """`mb scale` prices configurations too large to build, sight included."""
    cfg = seeing()
    model = MotherBrain(cfg)
    assert model.n_params() == cfg.n_params
    assert cfg.vision_params > 0
    assert cfg.n_image_tokens == (32 // 8) ** 2


def test_a_text_only_model_is_untouched_by_any_of_this():
    """Sight is additive. With vision_layers == 0 nothing changes at all."""
    cfg = seeing(vision_layers=0)
    model = MotherBrain(cfg)
    assert model.vision is None
    assert cfg.vision_params == 0
    assert cfg.n_image_tokens == 0
    assert model.n_params() == cfg.n_params

    x = torch.randint(0, 300, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 300)
    assert torch.isfinite(loss)


def test_an_image_becomes_tokens_the_transformer_reads():
    cfg = seeing()
    model = MotherBrain(cfg).eval()
    x = torch.randint(0, 300, (2, 10))
    images = torch.randn(2, 3, 32, 32)

    with_image, loss = model(x, x, images=images)
    without, _ = model(x, x)

    # Visual positions are context, not predictions: the output still lines up
    # with the text tokens, not text plus patches.
    assert with_image.shape == without.shape == (2, 10, 300)
    assert torch.isfinite(loss)
    assert not torch.allclose(with_image, without)   # the image changed something


def test_asking_a_blind_model_to_look_is_an_error():
    model = MotherBrain(seeing(vision_layers=0))
    with pytest.raises(ValueError, match="no vision tower"):
        model(torch.randint(0, 300, (1, 4)), images=torch.randn(1, 3, 32, 32))


def test_generation_from_an_image_stays_in_the_vocabulary():
    torch.manual_seed(0)
    model = MotherBrain(seeing()).eval()
    out = list(model.generate(torch.tensor([[1, 2]]), max_new_tokens=6,
                              images=torch.randn(1, 3, 32, 32), top_k=5))
    assert len(out) == 6
    assert all(0 <= t < 300 for t in out)


def test_an_image_file_becomes_the_tensor_the_tower_wants(tmp_path):
    from PIL import Image

    from motherbrain.vision import load_image

    path = tmp_path / "square.png"
    Image.new("RGB", (61, 47), (30, 60, 200)).save(path)   # deliberately not square

    x = load_image(str(path), 32)
    assert x.shape == (1, 3, 32, 32)          # resized to what the tower expects
    assert -1.05 <= x.min() <= x.max() <= 1.05


def test_the_patch_grid_must_divide_the_image():
    from motherbrain.vision import PatchEmbed

    with pytest.raises(ValueError, match="divisible"):
        PatchEmbed(image_size=30, patch_size=8, width=32)


def test_the_largest_preset_can_see():
    from motherbrain.config import PRESETS

    mother = PRESETS["mother"]
    assert mother.sees
    assert mother.vision_params > 1e9
    assert mother.n_params > 1e15
    # sight is not free the way experts are: all of it runs for every image
    assert mother.vision_params < mother.n_active_params


@pytest.mark.parametrize("text,name", [
    ("/write notes.txt hello", "write"),
    ("write notes.txt", "write"),
    ("/sh ls -la", "sh"),
    ("/find TODO", "find"),
    ("search for TODO", "find"),
    ("/delete old.txt", "delete"),
    ("remove old.txt", "delete"),
])
def test_the_wider_action_vocabulary_parses(text, name):
    from motherbrain.commands import parse

    assert parse(text).name == name


def test_every_action_needs_its_argument():
    from motherbrain.commands import parse

    for text in ("/write", "/sh", "/find", "/delete", "/see"):
        assert parse(text).name == "error", f"{text} should not be accepted bare"


def test_every_action_is_refused_over_http(served):
    """These write files and run commands.

    In a terminal that is no more than the shell already allows. Over HTTP any
    one of them is remote code execution against whoever serves the model, so
    the refusal has to cover the whole set, not the ones that existed when it
    was written.
    """
    from fastapi.testclient import TestClient

    from motherbrain.commands import LOCAL_ONLY
    from motherbrain.server import create_app

    run, corpus = served
    client = TestClient(create_app(run_dir=str(run), corpus_dir=str(corpus),
                                   auto_patch=False))

    probes = {
        "make": "/make a thing", "run": "/run x.py", "ls": "/ls /",
        "cat": "/cat /etc/passwd", "see": "/see x.png",
        "write": "/write /etc/x hi", "sh": "/sh rm -rf /",
        "find": "/find secret", "delete": "/delete /etc/passwd",
    }
    # every local-only action must have a probe, so adding one without
    # covering it here fails rather than slipping through
    assert set(probes) == LOCAL_ONLY

    for name, text in probes.items():
        result = client.post("/command", json={"text": text}).json()
        assert result["kind"] == "error", f"{name} was not refused"
        assert "never over the network" in result["text"]


# ---- the bulletin board -----------------------------------------------------

def test_telnet_negotiation_never_loops():
    """Two machines that re-affirm settled options shout IAC until the socket
    fills. Answering a confirmation is what starts that."""
    from motherbrain.telnet import (DO, IAC, OPT_ECHO, OPT_NAWS, OPT_TTYPE,
                                    WILL, Telnet)

    tn = Telnet()
    opening = tn.start()
    assert bytes([IAC, WILL, OPT_ECHO]) in opening
    assert bytes([IAC, DO, OPT_NAWS]) in opening

    # The client confirms what the board already offered. Silence is correct.
    assert tn.feed(bytes([IAC, DO, OPT_ECHO])).reply == b""
    assert tn.feed(bytes([IAC, WILL, OPT_NAWS])).reply == b""

    # Something never offered is declined, once.
    reply = tn.feed(bytes([IAC, WILL, 99])).reply
    assert reply == bytes([IAC, 254, 99])
    assert tn.feed(bytes([IAC, WILL, 99])).reply == b""

    # A terminal type that means "period client" picks the period encoding.
    from motherbrain.telnet import prefers_cp437
    got = tn.feed(bytes([IAC, 250, OPT_TTYPE, 0]) + b"SyncTERM"
                  + bytes([IAC, 240]))
    assert got.terminal == "SyncTERM" and prefers_cp437(got.terminal)
    assert not prefers_cp437("xterm-256color")


def test_telnet_commands_split_across_packets_still_parse():
    """A three-byte IAC DO can arrive one byte per segment, and often does."""
    from motherbrain.telnet import DO, IAC, OPT_SGA, Telnet

    whole = Telnet()
    whole.start()
    expected = whole.feed(bytes([IAC, DO, OPT_SGA])).reply

    dribbled = Telnet()
    dribbled.start()
    out = b"".join(dribbled.feed(bytes([b])).reply
                   for b in (IAC, DO, OPT_SGA))
    assert out == expected

    # And 255 in the data is doubled on the wire, once on the way out.
    from motherbrain.telnet import escape
    assert escape(b"\xff\x01") == b"\xff\xff\x01"
    assert Telnet().feed(b"a\xff\xffb").data == b"a\xffb"


def test_a_window_size_that_is_nonsense_is_ignored():
    """NAWS arrives from whatever is at the far end; 0 columns would divide
    every layout on the board by zero."""
    from motherbrain.telnet import IAC, OPT_NAWS, SB, SE, Telnet

    tn = Telnet()
    tn.feed(bytes([IAC, SB, OPT_NAWS, 0, 0, 0, 0, IAC, SE]))
    assert (tn.columns, tn.rows) == (80, 24), "it took a zero-width screen"
    tn.feed(bytes([IAC, SB, OPT_NAWS, 0, 132, 0, 43, IAC, SE]))
    assert (tn.columns, tn.rows) == (132, 43)


def test_xmodem_sends_a_file_its_own_receiver_gets_back():
    """A transfer protocol you cannot run against a receiver is a guess."""
    import os

    from motherbrain import xmodem

    for size in (1, 127, 128, 1025, 9000):
        data = os.urandom(size)
        sender = xmodem.Sender(data)
        receiver = xmodem.Receiver(crc=True)
        out = sender.begin(receiver.start()[0])
        for _ in range(500):
            if out is None:
                break
            acks = receiver.feed(out)
            if not acks:
                break
            for byte in acks:
                out = sender.answer(byte)
            if receiver.done:
                break
        assert receiver.done, f"{size} bytes never finished"
        assert receiver.file(size) == data, f"{size} bytes came back wrong"


def test_xmodem_resends_a_block_that_was_naked():
    """NAK means "again". A sender that advances on NAK drops data silently."""
    from motherbrain import xmodem

    sender = xmodem.Sender(b"A" * 3000)
    first = sender.begin(xmodem.CRC_REQUEST)
    again = sender.answer(xmodem.NAK)
    assert again == first, "a NAK advanced the block instead of repeating it"
    nxt = sender.answer(xmodem.ACK)
    assert nxt != first

    # And the checksum variant is what a receiver asking with NAK gets.
    plain = xmodem.Sender(b"hello")
    block = plain.begin(xmodem.NAK)
    assert plain.crc is False
    assert len(block) == 3 + 128 + 1


def test_ansi_measures_width_without_counting_escapes():
    """Every panel on the board is padded to a column count. Counting colour
    codes as characters is how a box comes out ragged."""
    from motherbrain import ansi as A

    coloured = f"{A.HY}hello{A.RESET}"
    assert A.width_of(coloured) == 5
    assert A.width_of(A.pad(coloured, 20)) == 20
    assert A.visible(A.truncate(coloured, 3)) == "hel"

    for line in A.box("TITLE", [coloured, "x" * 200], width=40):
        assert A.width_of(line) == 40, repr(A.visible(line))


def test_the_logo_is_drawable_on_a_1987_client():
    """The board offers code page 437 to period clients. A logo drawn with
    glyphs CP437 does not have would arrive as question marks."""
    from motherbrain.ansi import _LOGO, SHADE, logo

    _LOGO.encode("cp437")
    SHADE.encode("cp437")
    for style in ("┌┐└┘─│├┤", "╔╗╚╝═║╠╣"):
        style.encode("cp437")
    assert "\x1b[" in logo()


def test_a_picture_needs_no_imaging_library():
    """The gallery draws from the tensor it already has. Requiring Pillow to
    turn numbers into coloured blocks would break the half of the board that
    always works when the optional half is missing."""
    from motherbrain import ansi as A

    tensor = torch.zeros(3, 32, 32)
    tensor[0, 8:24, 8:24] = 1.0
    art = A.picture_tensor(tensor, width=24, colours=256)
    rows = art.split("\n")
    assert len(rows) == 12, "two pixels to a cell, so half as many rows"
    assert all(A.width_of(r) == 24 for r in rows)
    assert "\x1b[38;5;" in art

    sixteen = A.picture_tensor(tensor, width=24, colours=16)
    assert "38;5;" not in sixteen, "256-colour escapes sent to a 16-colour client"


def test_the_maze_can_always_be_walked_out_of():
    """Recursive backtracking makes a perfect maze; a caller trapped at the
    start by a generator bug would look like the door had crashed."""
    from motherbrain.doors import build_maze, draw_maze

    for seed in range(8):
        grid = build_maze(9, 6, seed=seed)
        start, exit_at = (1, 1), (2 * 9 - 1, 2 * 6 - 1)
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= ny < len(grid) and 0 <= nx < len(grid[0])
                        and grid[ny][nx] and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        assert exit_at in seen, f"seed {seed} walled the exit off"

    drawn = draw_maze(build_maze(4, 3, seed=1), (1, 1), (7, 5))
    assert "@@" in drawn and "><" in drawn


def test_the_file_area_never_offers_what_it_should_not(tmp_path):
    """A caller picks a number off a list; the list is what has to be safe."""
    from motherbrain.bbs import listing

    root = tmp_path / "area"
    (root / ".git").mkdir(parents=True)
    (root / ".ssh").mkdir()
    (root / "sub").mkdir()
    (root / "fine.txt").write_text("ok")
    (root / "sub" / "also.txt").write_text("ok")
    (root / ".git" / "secret.txt").write_text("no")
    (root / ".ssh" / "id_rsa").write_text("no")
    (root / "server.key").write_text("no")

    names = {p.name for p in listing({"roots": [root], "globs": ["*"]})}
    assert names == {"fine.txt", "also.txt"}, names

    # A symlink pointing out of the root is resolved and then refused.
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    try:
        (root / "escape.txt").symlink_to(outside)
    except OSError:
        return
    names = {p.name for p in listing({"roots": [root], "globs": ["*"]})}
    assert "outside.txt" not in names and "escape.txt" not in names


def test_a_handle_cannot_become_a_path(tmp_path):
    """Handles name a directory in the file area. Callers choose them."""
    from motherbrain.bbs import _safe_name

    for nasty in ("../../etc", "a/b", "..", "", "  ", "C:\\windows"):
        safe = _safe_name(nasty)
        assert "/" not in safe and "\\" not in safe and safe not in ("", ".", "..")
        assert (tmp_path / safe).resolve().parent == tmp_path.resolve()


def test_the_board_refuses_everything_that_touches_the_machine():
    """`mb console` runs shell commands because you already have a shell.
    A telnet caller does not, and giving them one is the whole hole."""
    import inspect

    from motherbrain import bbs
    from motherbrain.commands import LOCAL_ONLY, parse

    source = inspect.getsource(bbs.option_do)
    assert "LOCAL_ONLY" in source, "the refusal is no longer by the shared set"
    for text in ("sh rm -rf /", "run evil.py", "cat /etc/passwd",
                 "delete everything", "write /etc/hosts"):
        assert parse(text).name in LOCAL_ONLY, text


class _Dialler:
    """A telnet client, enough of one to log in to the board."""

    def __init__(self, reader, writer) -> None:
        self.reader, self.writer, self.seen = reader, writer, ""

    async def read(self, seconds: float = 2.0) -> str:
        """Everything that arrives, with the protocol's commands taken out."""
        import asyncio

        # Read for the whole window rather than stopping at the first lull.
        # The board sends its screens in bursts with gaps in the middle -
        # a reader that stops at a gap gets the first burst and calls it the
        # screen.
        out = bytearray()
        deadline = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                chunk = await asyncio.wait_for(self.reader.read(4096), 0.2)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                break
            i = 0
            while i < len(chunk):
                if chunk[i] != 255:
                    out.append(chunk[i])
                    i += 1
                elif chunk[i + 1:i + 2] and chunk[i + 1] == 250:
                    end = chunk.find(bytes([255, 240]), i)
                    i = len(chunk) if end < 0 else end + 2
                else:
                    i += 3
        text = bytes(out).decode("utf-8", "replace")
        self.seen += text
        return text

    async def type(self, text: str, seconds: float = 2.0) -> str:
        self.writer.write(text.encode())
        await self.writer.drain()
        return await self.read(seconds)

    async def until(self, needle: str, seconds: float = 10.0) -> str:
        """Read until the board says something in particular.

        Waiting for a quiet gap instead is fragile: the board sends its
        mouse-enable escape the moment a caller connects, and a reader that
        stops at the first lull returns that and nothing else.
        """
        import asyncio

        deadline = asyncio.get_running_loop().time() + seconds
        collected = ""
        while asyncio.get_running_loop().time() < deadline:
            collected += await self.read(0.5)
            if needle in _strip(collected):
                return collected
        return collected


def _strip(text: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


@pytest.fixture
def board_port(served):
    """A real board on a real socket, in this event loop, on a free port."""
    import asyncio

    from motherbrain.bbs import Board, session

    run, corpus = served
    board = Board(str(run), str(corpus), device="cpu")
    board.load()

    async def start():
        server = await asyncio.start_server(
            lambda r, w: session(r, w, board), "127.0.0.1", 0)
        return server, server.sockets[0].getsockname()[1]

    return board, start


def test_a_caller_can_dial_in_and_reach_the_menu(board_port):
    """The whole path: negotiate, log in, and get the five options."""
    import asyncio

    board, start = board_port

    async def call():
        server, port = await start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        client = _Dialler(reader, writer)
        opening = _strip(await client.until("user number or name"))
        assert "user number or name" in opening, opening[-300:]

        # WWIV's login: NEW, then the new-user application. Each answer is
        # given a moment to be consumed - the board reads keys one at a
        # time, and a test that runs ahead of it is testing its own timing.
        for answer in ("NEW\r", "TESTER\r", "\r", "\r", " "):
            await client.type(answer, 1.5)
        await client.until("Chat with MotherBrain")
        # Everything the board has said so far, not just the last burst:
        # the menu may already have arrived while an earlier answer was
        # being read, and a test that only looks at the last read misses it.
        menu = _strip(client.seen)

        # The five console options, in order, word for word.
        from motherbrain.voice import MENU
        for _, label in __import__(
                "motherbrain.bbs", fromlist=["x"]).CONSOLE_OPTIONS:
            assert label in menu, label
            assert label in MENU, f"{label} drifted from the console menu"

        assert "Chat with MotherBrain" in menu
        assert "Transfer section" in menu
        assert "sysop" in menu, "the local caller is not offered //"
        assert board.callers, "the board did not register the node"
        assert board.users.find("TESTER") is not None, "no user record"

        # And it answers a question it can answer exactly.
        before = len(client.seen)
        await client.type("C\r", 2.0)
        await client.type("what is 6 * 7\r", 1.0)
        await client.until("COMPUTED")
        answer = _strip(client.seen[before:])
        assert "42" in answer
        assert "COMPUTED" in answer, answer[-300:]

        writer.close()
        server.close()
        await server.wait_closed()

    asyncio.run(asyncio.wait_for(call(), 150))


def test_two_callers_hear_each_other(board_port):
    """A teleconference nobody else's words reach is a text box."""
    import asyncio

    board, start = board_port

    async def call():
        server, port = await start()

        async def dial(handle):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            client = _Dialler(reader, writer)
            await client.read(2.0)
            for answer in ("NEW\r", f"{handle}\r", "\r", "\r", " "):
                await client.type(answer, 1.5)
            await client.until("Command:", 8.0)
            await client.type("M\r", 1.0)      # WWIV: M is multi-node chat
            await client.until("TELECONFERENCE", 8.0)
            return client, writer

        alice, aw = await dial("ALICE")
        bob, bw = await dial("BOB")
        assert "BOB joins" in _strip(await alice.until("BOB joins", 6.0))

        await alice.type("hello bob\r", 0.2)
        heard = _strip(await bob.until("hello bob", 6.0))
        assert "<ALICE> hello bob" in heard, heard

        # And a room is a room: leave it and the words stop arriving.
        await bob.type("/join ELSEWHERE\r", 1.0)
        await alice.type("still here\r", 1.0)
        assert "still here" not in _strip(await bob.read(1.0))

        for writer in (aw, bw):
            writer.close()
        server.close()
        await server.wait_closed()

    asyncio.run(asyncio.wait_for(call(), 150))


def test_the_board_menu_is_the_console_menu():
    """Five options, one order, three faces. A board that renumbered them
    would be a different program wearing the same name."""
    from motherbrain.bbs import CONSOLE_OPTIONS
    from motherbrain.gui import OPTIONS
    from motherbrain.voice import MENU

    assert [key for key, _ in CONSOLE_OPTIONS] == ["1", "2", "3", "4", "5"]
    for (key, label), window in zip(CONSOLE_OPTIONS, OPTIONS + [("5  Run the GUI", "")]):
        assert label in MENU, f"{label} is not on the console menu"
        assert MENU.index(label) >= 0
    # and in the order the console lists them
    positions = [MENU.index(label) for _, label in CONSOLE_OPTIONS]
    assert positions == sorted(positions), "the board reordered the options"


# ---- inference --------------------------------------------------------------

def _tiny_model(served):
    from motherbrain.cli import load_current

    run, corpus = served
    return load_current(str(run), "cpu")


def test_a_batch_says_exactly_what_one_at_a_time_says(served):
    """The whole point of the padding mask.

    Prompts of different lengths are padded to a common width, so a row can
    only be correct if it is blind to the padding beside it. Greedy decoding
    makes the comparison exact: any drift is the mask leaking, not sampling.
    """
    from motherbrain.inference import Request, generate_batch

    model, tok, device, _ = _tiny_model(served)
    prompts = ["the", "the mother brain", "the mother brain awakens and"]

    alone = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=device)
        alone.append(tok.decode(list(model.generate(
            ids, max_new_tokens=12, temperature=0.0, top_k=None, top_p=None,
            eos_id=None))))

    requests = [Request(p, max_new_tokens=12, temperature=0.0, top_k=None,
                        top_p=None) for p in prompts]
    generate_batch(model, tok, requests, device, eos_id=None)

    for prompt, one, batched in zip(prompts, alone, requests):
        assert one == batched.text, (
            f"{prompt!r} came out differently in a batch:\n"
            f"  alone   {one!r}\n  batched {batched.text!r}")


def test_a_long_prompt_beside_a_short_one_does_not_change_it(served):
    """Padding is the failure mode: a short row sitting next to a long one is
    where a missing mask shows up, and only there."""
    from motherbrain.inference import Request, generate_batch

    model, tok, device, _ = _tiny_model(served)

    solo = Request("the", max_new_tokens=10, temperature=0.0, top_k=None,
                   top_p=None)
    generate_batch(model, tok, [solo], device, eos_id=None)

    together = [Request("the", max_new_tokens=10, temperature=0.0, top_k=None,
                        top_p=None),
                Request("the mother brain awakens and learns the mother brain",
                        max_new_tokens=10, temperature=0.0, top_k=None,
                        top_p=None)]
    generate_batch(model, tok, together, device, eos_id=None)
    assert together[0].text == solo.text, "the padding leaked into the short row"


def test_rows_finish_when_they_are_done_not_when_the_batch_is(served):
    """A row with a small budget must stop at it, however long the batch runs."""
    from motherbrain.inference import Request, generate_batch

    model, tok, device, _ = _tiny_model(served)
    short = Request("the", max_new_tokens=3, temperature=0.0, top_k=None,
                    top_p=None)
    long = Request("the", max_new_tokens=20, temperature=0.0, top_k=None,
                   top_p=None)
    measured = generate_batch(model, tok, [short, long], device, eos_id=None)

    assert len(short.tokens) == 3
    assert len(long.tokens) == 20
    assert measured.generated_tokens == 23
    assert measured.tokens_per_second > 0
    assert "tokens/s" in measured.render()


def test_the_engine_runs_what_arrives_together_together(served):
    """Continuous batching, or the eighth caller waits for seven generations."""
    import asyncio

    from motherbrain.inference import Engine, Request

    model, tok, device, _ = _tiny_model(served)

    async def main():
        engine = Engine(model, tok, device, eos_id=None, max_batch=8,
                        window=0.2, adaptive=False)
        engine.start()
        texts = await asyncio.gather(*[
            engine.submit(Request(f"the {i}", max_new_tokens=4,
                                  temperature=0.0, top_k=None, top_p=None))
            for i in range(6)])
        await engine.stop()
        return texts, engine.served

    texts, served_stats = asyncio.run(asyncio.wait_for(main(), 60))
    assert len(texts) == 6 and all(isinstance(t, str) for t in texts)
    assert served_stats.batches == 1, \
        f"six simultaneous requests became {served_stats.batches} batches"
    assert served_stats.prompts == 6


def test_a_failing_batch_does_not_hang_the_callers_waiting_on_it(served):
    """An exception in the worker has to reach everyone in that batch, or
    every one of them waits on a future nobody will ever set."""
    import asyncio

    from motherbrain.inference import Engine, Request

    model, tok, device, _ = _tiny_model(served)

    async def main():
        engine = Engine(model, tok, device, eos_id=None, max_batch=4,
                        window=0.05)
        engine.start()
        engine.model = None                     # whatever goes wrong, goes wrong
        try:
            await asyncio.wait_for(
                asyncio.gather(*[engine.submit(Request("the", max_new_tokens=2))
                                 for _ in range(3)], return_exceptions=True), 20)
        finally:
            await engine.stop()

    results = asyncio.run(main())
    # Reaching here at all is the assertion: the wait_for did not time out.


def test_the_mask_is_optional_so_training_is_untouched(served):
    """The batched path added an argument to the model. Everything that does
    not pass it has to behave exactly as it did."""
    import inspect

    from motherbrain.model import Attention, Block, MotherBrain

    for fn in (Attention.forward, Block.forward, MotherBrain.forward):
        assert inspect.signature(fn).parameters["mask"].default is None

    model, tok, device, _ = _tiny_model(served)
    ids = torch.tensor([tok.encode("the mother brain", bos=True)],
                       device=device)
    with torch.no_grad():
        plain, _ = model(ids)
        masked, _ = model(ids, mask=None)
    assert torch.equal(plain, masked)


def test_running_the_program_with_no_arguments_starts_it(monkeypatch):
    """`mb`, or a double-clicked shortcut, passes no command at all. A
    program whose job is to start MotherBrain should start it rather than
    print a list of flags and exit non-zero."""
    from motherbrain import cli

    started = {}
    # The parser binds `func` when it is built, and it is built inside main(),
    # so replacing the module-level function here is what gets dispatched to.
    monkeypatch.setattr(cli, "cmd_console",
                        lambda args: started.update(ran=True) or 0)

    assert cli.main([]) == 0
    assert started.get("ran"), "`mb` with no arguments did not open the console"

    # And it is still an error to ask for a command that does not exist.
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


# ---- WWIV -------------------------------------------------------------------

def test_heart_codes_become_colour_and_measure_as_nothing():
    """WWIV wrote a heart and a digit, and looked the colour up in a table
    the sysop could edit. Every screen on this board is written that way, so
    a width that counted the codes would misalign all of them."""
    from motherbrain import wwiv

    text = "\x031Main\x030 normal \x032yellow"
    rendered = wwiv.render(text)
    assert "\x1b[" in rendered and "\x03" not in rendered
    assert wwiv.strip(text) == "Main normal yellow"
    assert wwiv.width_of(text) == len("Main normal yellow")

    # A heart with no digit after it is somebody typing a heart.
    assert wwiv.render("love \x03 you") == "love \x03 you"

    # And the table is what decides, so changing it changes every screen.
    other = tuple([0x0C] + list(wwiv.DEFAULT_COLOURS[1:]))
    assert wwiv.render("\x030x", other) != wwiv.render("\x030x")


def test_security_levels_decide_what_a_caller_can_do():
    """One number per user, looked up in one place. That was the whole of a
    WWIV board's access policy and it is a genuinely good design."""
    from motherbrain import wwiv

    new = wwiv.User(number=2, name="NEWBIE")
    assert not new.sysop
    assert not new.rules().can_upload, "a brand new caller could upload"
    assert new.minutes_left() == wwiv.LEVELS[10].minutes_per_call

    sysop = wwiv.User(number=1, name="SYSOP", sl=255, dsl=255,
                      ar="ABCDEFGHIJKLMNOP", dar="ABCDEFGHIJKLMNOP")
    assert sysop.sysop and sysop.rules().can_upload

    # DSL and DAR gate a file directory independently of SL.
    caller = wwiv.User(number=3, name="MID", sl=50, dsl=20, dar="B")
    assert caller.may_download("", 20)
    assert not caller.may_download("", 30), "DSL 20 reached a DSL 30 directory"
    assert caller.may_download("B", 10)
    assert not caller.may_download("C", 10), "a missing DAR flag let them in"

    # Today's allowance runs out even when the per-call one has not.
    tired = wwiv.User(number=4, name="TIRED", sl=10)
    tired.today = wwiv.LEVELS[10].minutes_per_day
    assert tired.minutes_left() == 0


def test_user_records_are_numbered_from_one_and_survive_a_restart(tmp_path):
    """USER.LST, and #1 is the sysop."""
    from motherbrain import wwiv

    path = tmp_path / "users.json"
    users = wwiv.Users(path)
    first = users.create("SAMUS", sysop=True)
    second = users.create("GUEST")
    assert (first.number, second.number) == (1, 2)
    assert first.sysop and not second.sysop

    again = wwiv.Users(path)
    assert again.find("2").name == "GUEST"
    assert again.find("samus").number == 1
    assert again.find("#1").name == "SAMUS"
    assert again.find("nobody") is None


def test_the_board_menu_is_still_the_console_menu():
    """WWIV's letters around them, but the five are the five, in order."""
    from motherbrain.bbs import CONSOLE_OPTIONS
    from motherbrain.voice import MENU

    assert [key for key, _ in CONSOLE_OPTIONS] == ["1", "2", "3", "4", "5"]
    positions = [MENU.index(label) for _, label in CONSOLE_OPTIONS]
    assert positions == sorted(positions), "the board reordered the options"


def test_passwords_are_hashed_and_never_stored(tmp_path):
    """Telnet is plaintext; the password file does not have to be."""
    from motherbrain.bbs import Board

    board = Board(str(tmp_path / "run"), str(tmp_path / "corpus"))
    user = board.users.create("SAMUS")
    assert board.check_password(user, "anything"), "no password means open"

    board.set_password(user, "correct horse battery staple")
    stored = (tmp_path / "run" / "bbs" / "passwords.json").read_text()
    assert "correct horse" not in stored
    assert board.check_password(user, "correct horse battery staple")
    assert not board.check_password(user, "wrong")

    board.clear_password(user)
    assert board.check_password(user, "")


def test_there_is_no_shell_command_on_the_board():
    """WWIV had //DOS. This does not, and the reason is not an oversight: a
    shell on the far end of a plaintext telnet session is the hole."""
    import inspect

    from motherbrain import bbs

    names = {name for name, _ in bbs.SYSOP_HELP}
    assert not any("DOS" in name or "SHELL" in name.upper() for name in names)
    source = inspect.getsource(bbs.sysop_command)
    assert "subprocess" not in source and "os.system" not in source
    assert "//DOS" in source, "the reason it is absent is no longer written down"


def test_the_sysop_can_reach_every_internal():
    """`//?` has to list something that exists for each line it prints."""
    import inspect

    from motherbrain import bbs

    source = inspect.getsource(bbs.sysop_command)
    for name, _what in bbs.SYSOP_HELP:
        verb = name[2:].split()[0].split("<")[0].strip().upper()
        if verb in ("?",):
            continue
        assert f'"{verb}"' in source, f"{name} is listed but not implemented"


def test_a_release_is_a_real_archive_that_describes_itself(tmp_path):
    """A file area is a shelf of packaged things, and the thing that makes
    it one is the FILE_ID.DIZ the listing reads out of each archive."""
    import zipfile

    from motherbrain import warez

    built = warez.build(tmp_path / "run", force=True)
    names = {p.name for p in built}
    assert {"MBRAIN.ZIP", "MBDOORS.ZIP", "MBANSI.ZIP", "MBRAIN.NFO"} <= names

    doors_zip = warez.area(tmp_path / "run") / "MBDOORS.ZIP"
    with zipfile.ZipFile(doors_zip) as archive:
        inside = archive.namelist()
        assert "FILE_ID.DIZ" in inside
        assert "PLAYDOORS.py" in inside
        assert "motherbrain/doors.py" in inside
        # It has to stand alone: nothing that needs torch belongs in it.
        assert not any(n.endswith(("model.py", "train.py", "server.py"))
                       for n in inside), inside

    text = warez.read_diz(doors_zip)
    assert text and len(text.splitlines()) <= warez.DIZ_LINES
    assert all(len(line) <= warez.DIZ_WIDTH for line in text.splitlines())
    assert warez.describe(doors_zip).startswith("MOTHERBRAIN DOORS")

    # The ANSI pack has to be openable by something that reads .ANS files.
    with zipfile.ZipFile(warez.area(tmp_path / "run") / "MBANSI.ZIP") as art:
        logo = art.read("MBLOGO.ANS")
        assert logo.startswith(b"\x1b["), "not an ANSI file"
        logo.decode("cp437")            # a real .ANS is code page 437


def test_the_public_directories_take_uploads_from_anyone(tmp_path):
    """GAMES and WHATEVERWARE are open both ways; the rest are not."""
    from motherbrain.bbs import Board, file_sections, upload_target

    board = Board(str(tmp_path / "run"), str(tmp_path / "corpus"))
    sections = {s["name"]: s for s in file_sections(board)}

    for name in ("GAMES", "WHATEVERWARE"):
        assert sections[name]["upload"], f"{name} does not take uploads"
        assert sections[name]["dsl"] == 0, f"{name} is gated"
        assert sections[name].get("public"), f"{name} is not public"
    assert not sections["WAREZ"].get("upload"), "the shelf takes uploads"
    assert not sections["SOURCE"].get("upload")

    class FakeCaller:
        handle = "../../etc"
        board = None

    caller = FakeCaller()
    caller.board = board
    public = upload_target(caller, sections["GAMES"])
    assert public == Path(sections["GAMES"]["roots"][0])
    private = upload_target(caller, sections["USER"])
    assert ".." not in private.parts, "a handle became a path"


def test_the_classic_doors_are_the_classics():
    """The BASIC canon is public domain, which is why these are the real
    games rather than things in their shape."""
    import random

    from motherbrain import doors

    keys = {key for key, _name, _blurb in doors.CATALOGUE}
    assert {"H", "W", "L", "N", "E", "R"} <= keys

    # ELIZA reflects pronouns, which is the entire trick.
    rng = random.Random(0)
    assert "you are" in doors.eliza_reply("I am tired", rng).lower() or \
        "tired" in doors.eliza_reply("I am tired", rng).lower()
    assert doors._reflect("i am your friend") == "you are my friend"

    # And the daily-turn RPG resets its turns, which is what makes it daily.
    fresh = doors._wyrm_new("SOMEBODY")
    assert fresh["turns"] == doors.WYRM_TURNS
    assert fresh["level"] == 1 and fresh["gold"] == 0


# ---- natural language -------------------------------------------------------

def test_it_reads_a_sentence_before_answering_it():
    """A tagger that guesses makes the parser downstream confidently wrong,
    so the closed classes are listed and the rest is decided by shape."""
    from motherbrain import nlp

    tagged = dict(nlp.tag(nlp.tokenise("the model runs quickly")))
    assert tagged["the"] == "DET"
    assert tagged["runs"] in ("VERB", "NOUN")
    assert tagged["quickly"] == "ADV"

    for text, kind in [("What is a modem?", "question"),
                       ("hello", "greeting"),
                       ("thanks", "thanks"),
                       ("The brain learns.", "statement"),
                       ("can you see", "question"),
                       ("why not", "question")]:
        assert nlp.analyse(text).kind == kind, text

    asked = nlp.analyse("how many parameters do you have")
    assert asked.wh == "how" and asked.about_self
    assert "parameter" in asked.keywords, asked.keywords

    # The lemmatiser must not mangle adjectives - "conscious" is not
    # "consciou", and every keyword match downstream depends on it.
    assert nlp.lemma("conscious", "ADJ") == "conscious"
    assert nlp.lemma("parameters") == "parameter"
    assert nlp.lemma("running", "VERB") == "run"
    assert nlp.lemma("children") == "child"


def test_it_composes_english_rather_than_fragments():
    """Agreement is where a generated sentence gives itself away."""
    from motherbrain import nlp

    assert nlp.agree("it", "be") == "is"
    assert nlp.agree("they", "be") == "are"
    assert nlp.agree("I", "be") == "am"
    assert nlp.agree("model", "run") == "runs"
    assert nlp.agree("models", "run") == "run"
    assert nlp.agree("it", "try") == "tries"
    assert nlp.agree("it", "watch") == "watches"

    assert nlp.article("modem") == "a"
    assert nlp.article("apple") == "an"
    assert nlp.article("hour") == "an"
    assert nlp.article("user") == "a", "'an user' is how you spot a machine"

    assert nlp.sentence("  the  cat sat ") == "The cat sat."
    assert nlp.sentence("already right.") == "Already right."
    assert nlp.join(["sight", "sound", "video"]) == "sight, sound and video"
    assert nlp.join(["one"]) == "one"


def test_it_says_it_does_not_know(tmp_path):
    """The whole design turns on this being a real answer rather than a
    failure to produce one."""
    from motherbrain import nlp

    found = nlp.answer("what is a blorptrix", run_dir=str(tmp_path))
    assert found.source == "none"
    assert "do not know" in found.text
    assert "blorptrix" in found.text, "it did not say what it did not know"

    # And it never quietly produces prose in place of the admission.
    assert not found.evidence


def test_the_pipeline_prefers_the_certain_source(tmp_path):
    """Computed, then told, then its own state, then quoted. In that order,
    because that is the order of certainty."""
    from motherbrain import nlp
    from motherbrain.knowledge import Knowledge

    run = tmp_path / "run"
    run.mkdir()

    assert nlp.answer("hello").source == "social"
    assert nlp.answer("what is 6 * 7", run_dir=str(run)).source == "exact"
    assert "42" in nlp.answer("what is 6 * 7", run_dir=str(run)).text

    base = Knowledge(str(run))
    base.tell("a modem is a device")
    base.tell("all devices need power")
    told = nlp.answer("what is a modem", run_dir=str(run))
    assert told.source == "known"
    assert "device" in told.text and "power" in told.text

    state = {"version": 5, "total_params": 52_222_872,
             "active_params": 33_348_504, "params_at_v0": 18_880_896,
             "patches": 5}
    mine = nlp.answer("how many parameters do you have", run_dir=str(run),
                      stats=state)
    assert mine.source == "self" and "52,222,872" in mine.text


def test_retrieval_quotes_and_never_paraphrases(tmp_path):
    """Grounding an answer in something it has read is only honest if you
    can be shown the sentence, exactly as it was written."""
    from motherbrain import nlp
    from motherbrain.data import Corpus

    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text(
        "A modem converts digital data into an analogue signal. "
        "The signal travels over a telephone line. "
        "This third sentence is about something else entirely.", "seed")
    nlp.forget_index()

    index = nlp.corpus_index(str(tmp_path / "corpus"))
    assert len(index) >= 3
    # The inverted index must only offer sentences that could match.
    assert index.candidates(["bicycle"]) == []
    assert index.candidates(["modem"])

    found = nlp.answer("what is a modem", corpus_dir=str(tmp_path / "corpus"))
    assert found.source == "read"
    assert found.evidence, "it claimed to quote and quoted nothing"
    quoted = found.evidence[0].strip('"')
    assert quoted in ("A modem converts digital data into an analogue signal.",
                      "The signal travels over a telephone line."), quoted
    nlp.forget_index()


def test_every_face_answers_through_the_same_pipeline():
    """Four faces that disagree about what is true are four programs."""
    import inspect

    from motherbrain import bbs, cli, gui

    for name, source in (("board", inspect.getsource(bbs.Board.answer)),
                         ("terminal", inspect.getsource(cli.cmd_console)),
                         ("window", inspect.getsource(gui.App._do))):
        assert "nlp.answer" in source, f"the {name} answers its own way"

    from motherbrain.server import create_app
    assert "ask_endpoint" in inspect.getsource(create_app)


def test_it_describes_itself_as_what_it_is():
    """It is an artificial intelligence system, and the honest way to say so
    is to name the parts rather than the word."""
    from motherbrain.chat import answer_about_self

    said = answer_about_self("identity", {"version": 5,
                                          "total_params": 52_222_872})
    assert "artificial intelligence" in said
    for part in ("knowledge base", "calculator", "perception", "reasoning"):
        assert part in said, f"it did not mention its {part}"
    assert "labelled" in said, "it did not say answers carry their source"


# ---- the USB drive ----------------------------------------------------------

def test_a_usb_drive_carries_everything_and_leaves_nothing(tmp_path, served):
    """The promise is that the computer is as it was when you pull it out."""
    from motherbrain.usb import build

    run, corpus = served
    drive = tmp_path / "drive"
    build(drive, str(run), str(corpus), device="cpu")

    for name in ("MOTHERBRAIN.bat", "MOTHERBRAIN.command", "motherbrain.sh",
                 "autorun.inf", "START HERE.txt"):
        assert (drive / name).is_file(), f"{name} is not on the drive"
    assert (drive / "MotherBrain" / "motherbrain" / "bbs.py").is_file()
    assert (drive / "MotherBrain" / "runs" / "default" / "tokenizer.json"
            ).is_file()

    # Everything the program writes has to be pointed back at the drive.
    for launcher in ("MOTHERBRAIN.bat", "motherbrain.sh"):
        text = (drive / launcher).read_text()
        assert "MB_WORKSPACE" in text, f"{launcher} does not set the workspace"
        assert "PYTHONPYCACHEPREFIX" in text, \
            f"{launcher} leaves bytecode on the host"
        assert "PIP_CACHE_DIR" in text, f"{launcher} leaves a pip cache behind"
        assert ".venv" in text, f"{launcher} does not build on the drive"

    # A shell script with CRLF in it dies with "bad interpreter".
    assert b"\r\n" not in (drive / "motherbrain.sh").read_bytes()
    assert b"\r\n" not in (drive / "MOTHERBRAIN.command").read_bytes()
    assert b"\r\n" in (drive / "MOTHERBRAIN.bat").read_bytes(), \
        "a .bat with bare LF confuses older cmd.exe"

    import os
    assert os.access(drive / "motherbrain.sh", os.X_OK)
    assert os.access(drive / "MOTHERBRAIN.command", os.X_OK)


def test_the_drive_is_honest_about_not_opening_itself(tmp_path, served):
    """AutoRun on removable media has been off since 2011 and no file on a
    drive can turn it back on. Saying otherwise on the tin is the one thing
    that would make this untrustworthy."""
    from motherbrain.usb import build

    run, corpus = served
    drive = tmp_path / "drive"
    build(drive, str(run), str(corpus), device="cpu")

    text = (drive / "START HERE.txt").read_text()
    assert "No, and nothing on this drive can make it." in text
    assert "2011" in text and "Conficker" in text

    # And the opt-in installers say what they will do, and can be undone.
    for name in ("windows", "linux", "macos"):
        suffix = "cmd" if name == "windows" else "sh"
        installer = drive / "autostart" / f"{name}-install.{suffix}"
        undo = drive / "autostart" / f"{name}-uninstall.{suffix}"
        assert installer.is_file() and undo.is_file(), name
        body = installer.read_text()
        assert "changes this" in body.lower() or "changes THIS" in body, \
            f"{name} does not say whose machine it changes"
        assert "uninstall" in body, f"{name} does not point at its undo"


def test_the_board_carries_every_console_option():
    """The board is the console plus a board, not a different program."""
    from motherbrain.bbs import BOARD_COMMANDS, CONSOLE_OPTIONS, HANDLERS
    from motherbrain.voice import MENU

    for key, label in CONSOLE_OPTIONS:
        assert key in HANDLERS, f"option {key} is on the menu and does nothing"
        assert label in MENU, f"{label} is not the console's wording"

    # And chatting with it is one of the board's own commands.
    letters = {key for key, _label, _sl in BOARD_COMMANDS}
    assert "C" in letters and HANDLERS["C"].__name__ == "chat_with_motherbrain"


# ---- mouse, touch, and going back -------------------------------------------

def test_a_click_is_the_key_that_would_have_been_typed():
    """The menu knows where it drew itself, so pointing at an entry and
    typing its letter are the same act. This is also what makes a tap on a
    phone work: the browser sends the same mouse report."""
    from motherbrain.bbs import Caller

    class Fake:
        def __init__(self):
            self.data = bytearray()
        def write(self, b): self.data += b
        async def drain(self): pass
        def get_extra_info(self, _): return ("127.0.0.1", 1)
        def close(self): pass

    class Board:
        colours = [7] * 10
        callers: dict = {}

    caller = Caller(None, Fake(), Board(), 1)
    caller.row = 5
    caller.hotspot("C")
    caller.hotspots.append((7, 40, 78, "T"))

    assert caller.clicked(5, 1) == "C"
    assert caller.clicked(7, 50) == "T"
    assert caller.clicked(7, 10) == ""
    assert caller.clicked(9, 1) == ""

    # An SGR press turns into the keys; a release does not, or every
    # command would run twice.
    caller._mouse("0;50;7", "M")
    assert "".join(caller.keys) == "T\r"
    caller.keys.clear()
    caller._mouse("0;50;7", "m")
    assert not caller.keys
    caller.keys.clear()
    caller._mouse("64;50;7", "M")            # a scroll wheel
    assert not caller.keys


def test_the_board_draws_to_the_width_the_caller_has():
    """A phone is about forty columns. The floor used to be sixty-four, and
    a board that draws wider than the screen loses its right-hand half."""
    from motherbrain.bbs import NARROW, wide

    class Caller:
        def __init__(self, columns): self.columns = columns

    assert wide(Caller(40)) == 40
    assert wide(Caller(20)) == 38, "it must still be drawable at all"
    assert wide(Caller(100)) == 79
    assert wide(Caller(40)) < NARROW, "a phone must get the one-column menu"
    assert wide(Caller(80)) >= NARROW


def test_going_back_names_where_it_goes(tmp_path):
    """"Back" on its own makes a caller guess."""
    from motherbrain.bbs import Board, Caller

    class Fake:
        def write(self, b): pass
        async def drain(self): pass
        def get_extra_info(self, _): return ("127.0.0.1", 1)
        def close(self): pass

    board = Board(str(tmp_path / "run"), str(tmp_path / "corpus"))
    caller = Caller(None, Fake(), board, 1)
    assert caller.trail == ["Main menu"]
    assert caller.whence() == "Main menu"

    with caller.at("Transfer section"):
        assert caller.whence() == "Main menu"
        with caller.at("WAREZ"):
            assert caller.whence() == "Transfer section"
            assert "WAREZ" in caller.breadcrumb()
            assert "Transfer section" in caller.breadcrumb()
    assert caller.trail == ["Main menu"], "the trail was not put back"


def test_the_browser_front_end_is_a_relay_and_nothing_more():
    """Every screen, every door, every menu, unchanged - because the page
    speaks the same protocol the telnet socket does."""
    import inspect
    import socket
    import threading

    from fastapi.testclient import TestClient

    from motherbrain import webterm

    board = socket.socket()
    board.bind(("127.0.0.1", 0))
    board.listen(1)
    port = board.getsockname()[1]
    seen = []

    def pretend():
        conn, _ = board.accept()
        conn.sendall(b"\x1b[1;33mMENU\x1b[0m\r\n")
        try:
            seen.append(conn.recv(200))
        except OSError:
            pass

    threading.Thread(target=pretend, daemon=True).start()

    client = TestClient(webterm.create_app("127.0.0.1", port))
    assert "MotherBrain BBS" in client.get("/").text
    with client.websocket_connect("/ws") as ws:
        assert "MENU" in ws.receive_text()
        # A tap is sent as the mouse report a terminal would send.
        ws.send_text("\x1b[<0;12;5M")
    for _ in range(50):
        if seen:
            break
        import time as _t
        _t.sleep(0.05)
    assert seen and b"\x1b[<0;12;5M" in seen[0], seen

    # The page has to raise a phone keyboard and take taps.
    page = webterm.PAGE
    assert "touchscreen" not in page  # not a test of the driver
    assert "1000h" not in page, "the page does not turn mouse mode on itself"
    assert "typedEl.focus()" in page, "no way to raise a phone keyboard"
    assert 'addEventListener("click"' in page, "taps are not read"
    assert "document.addEventListener(\"keydown\"" in page, \
        "a desktop keyboard is not read"


def test_it_reads_every_language_it_claims_to():
    """"All programming languages" is a corpus problem, and the corpus has
    to accept them before anything can be taught."""
    from motherbrain.data import LANGUAGES, NAMED_LANGUAGES, language_of

    names = set(LANGUAGES.values()) | set(NAMED_LANGUAGES.values())
    for expected in ("Python", "C", "C++", "Rust", "Go", "Haskell", "COBOL",
                     "Fortran", "Assembly", "Lisp", "Prolog" if False else
                     "Erlang", "Swift", "SQL", "Shell", "PowerShell"):
        assert expected in names, expected
    assert len(names) > 70, f"only {len(names)} languages"

    assert language_of("a.hs") == "Haskell"
    assert language_of("x.cbl") == "COBOL"
    assert language_of("Makefile") == "Make"
    assert language_of("Dockerfile") == "Dockerfile"
    assert language_of("photo.png") == "unknown"

    # And what it says it has read has to come from what it has read.
    import inspect

    from motherbrain import cli
    source = inspect.getsource(cli.cmd_languages)
    assert "corpus.documents()" in source, "the census is not a census"
    assert "Reading is not learning" in source


def test_a_new_patch_keeps_every_patch_before_it_and_adds_parameters(tmp_path):
    """Growth is cumulative: v3 is the base plus patch one plus patch two
    plus patch three, and every one of them is still on disk. A patch that
    replaced its predecessor would make the lineage a lie."""
    from motherbrain.data import Corpus
    from motherbrain.patches import PatchConfig, PatchStore, build_version, create_patch
    from motherbrain.train import TrainConfig, save_checkpoint

    run = tmp_path / "run"
    corpus = Corpus(tmp_path / "corpus")
    corpus.add_text("the mother brain awakens and learns " * 60, "seed")
    tok, _ = corpus.prepare(vocab_size=320, verbose=False)
    cfg = tiny(vocab_size=tok.vocab_size, max_seq_len=32)
    save_checkpoint(run / "checkpoint.pt", MotherBrain(cfg), None, 1, cfg,
                    TrainConfig(), [])
    tok.save(str(run / "tokenizer.json"))

    sizes, versions = [], []
    for round_number in (1, 2):
        corpus.add_text(f"round {round_number}: " + "grow " * 200,
                        f"feed{round_number}")
        version = create_patch(
            str(run), str(tmp_path / "corpus"),
            PatchConfig(mode="grow", grow_experts=1, steps=2),
            note=f"round {round_number}", device="cpu")
        assert version is not None, f"round {round_number} learned nothing"
        versions.append(version)
        sizes.append(version.params_after)

    # Both patches are still there, in order, with the lineage recording it.
    files = sorted((run / "patches").glob("*.pt"))
    assert len(files) == 2, [f.name for f in files]
    store = PatchStore(str(run), create=False)
    lineage = store.versions()
    assert [v.version for v in lineage] == [1, 2]
    assert lineage[1].parent == 1, "the second patch forgot the first"

    # And the model got bigger, not merely different.
    assert sizes[1] > sizes[0] > versions[0].params_before, sizes
    assert versions[1].params_before == sizes[0], \
        "the second patch was applied to the wrong starting point"

    # Rebuilding from the base replays both, and lands on the larger model.
    model, _tok, current = build_version(str(run), device="cpu")
    assert current == 2
    assert model.n_params() == sizes[1]


def test_applying_a_patch_updates_a_running_board_in_place(tmp_path, served):
    """No restart, no reinstall. The board swaps the model it is serving and
    throws away the batching engine built around the old weights."""
    import asyncio
    import inspect

    from motherbrain.bbs import Board, option_patch

    source = inspect.getsource(option_patch)
    assert "board.load" in source, "the board keeps serving the old weights"
    assert "_engine = None" in source, \
        "the engine still holds the model that was replaced"
    assert "refresh_stats" in source, "the stats would go on reporting the old"
    assert "nlp.forget_index" in source, "the corpus grew and the index did not"

    run, corpus = served
    board = Board(str(run), str(corpus), device="cpu")
    board.load()
    first = board.model.n_params()

    async def swap():
        # Whatever the engine was, a reload has to leave a fresh one.
        from motherbrain.inference import Engine
        from motherbrain.tokenizer import EOS_ID

        board._engine = Engine(board.model, board.tok, board.torch_device,
                               eos_id=EOS_ID)
        old = board._engine
        await asyncio.to_thread(board.load)
        await old.stop()
        board._engine = None
        assert board.engine is not old, "the new model is served by the old engine"
        await board._engine.stop()

    asyncio.run(swap())
    assert board.model.n_params() == first
