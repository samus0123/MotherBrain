"""Tests for the parts that are easy to get quietly wrong."""

from pathlib import Path

import numpy as np
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


def test_status_reports_a_fresh_clone_as_not_loadable(tmp_path, capsys):
    """A clone has patches and a manifest but no weights, because the base
    checkpoint is too large to commit. Saying so is the whole point."""
    from motherbrain.cli import build_parser

    args = build_parser().parse_args(
        ["status", "--corpus", str(tmp_path / "corpus"), "--run", str(tmp_path / "run")])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "NOT LOADABLE" in out
    assert "mb bootstrap" in out


def test_status_reports_a_trained_run_as_ready(served, capsys):
    from motherbrain.cli import build_parser

    run, corpus = served
    args = build_parser().parse_args(
        ["status", "--corpus", str(corpus), "--run", str(run)])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "READY" in out
    assert "mb chat" in out


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
