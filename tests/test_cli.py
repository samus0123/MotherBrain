import json

import pytest

from motherbrain.cli import build_parser, main
from motherbrain.config import RunConfig


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_params_ladder_prints_every_preset(capsys):
    main(["params", "--ladder"])
    out = capsys.readouterr().out
    for name in ("nano", "small", "moe-large", "titan", "motherbrain"):
        assert name in out


def test_params_preset_reports_counts(capsys):
    main(["params", "--preset", "moe-small"])
    out = capsys.readouterr().out
    assert "total params" in out and "active params" in out


def test_params_overrides_apply(capsys):
    main(["params", "--preset", "nano", "--dim", "1024", "--n-layers", "4"])
    assert "dim 1024 x 4 layers" in capsys.readouterr().out


def test_config_command_writes_loadable_json(tmp_path, capsys):
    out = tmp_path / "run.json"
    main(["config", "--preset", "mini", "--out", str(out)])
    run = RunConfig.load(out)
    assert run.model.dim == 512
    assert json.loads(out.read_text())["model"]["n_layers"] == 12


def test_tokenizer_and_data_commands(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello world, this is a small corpus for testing. " * 200))

    tok_path = tmp_path / "tok.json"
    main(["tokenizer", str(corpus), "--vocab-size", "400", "--out", str(tok_path)])
    assert "roundtrip check passed" in capsys.readouterr().out
    assert tok_path.exists()

    data_dir = tmp_path / "tokens"
    main(["data", str(corpus), "--tokenizer", str(tok_path), "--out-dir", str(data_dir)])
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["splits"]["train"] > 0
    assert (data_dir / "train.bin").exists()


def test_data_command_without_tokenizer_uses_bytes(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("raw bytes only")
    data_dir = tmp_path / "tokens"
    main(["data", str(corpus), "--out-dir", str(data_dir)])
    assert json.loads((data_dir / "meta.json").read_text())["vocab_size"] == 257


def test_missing_input_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["data", str(tmp_path / "nope*.txt")])


def test_train_and_sample_round_trip(tmp_path, capsys):
    corpus = tmp_path / "c.txt"
    corpus.write_text("abcabcabc " * 500)
    data_dir = tmp_path / "tokens"
    main(["data", str(corpus), "--out-dir", str(data_dir)])

    run_dir = tmp_path / "run"
    main([
        "train", "--data-dir", str(data_dir), "--out-dir", str(run_dir),
        "--vocab-size", "257", "--dim", "32", "--n-layers", "2", "--n-heads", "4",
        "--max-seq-len", "32", "--seq-len", "32", "--batch-size", "4",
        "--max-steps", "5", "--warmup-steps", "1", "--device", "cpu", "--dtype", "fp32",
        "--eval-every", "0", "--save-every", "0",
    ])
    assert (run_dir / "final.pt").exists()

    capsys.readouterr()
    main([
        "sample", str(run_dir / "final.pt"), "--prompt", "abc",
        "--max-new-tokens", "10", "--device", "cpu", "--seed", "0",
    ])
    assert capsys.readouterr().out.strip()


def test_seq_len_is_clamped_to_context(tmp_path):
    from motherbrain.cli import _run_from_args

    args = build_parser().parse_args([
        "train", "--max-seq-len", "64", "--seq-len", "4096",
    ])
    assert _run_from_args(args).train.seq_len == 64


def test_overriding_n_heads_alone_stays_valid():
    """`--n-heads` without `--n-kv-heads` must re-derive, not keep a stale value."""
    from motherbrain.cli import _config_from_args

    args = build_parser().parse_args(["params", "--n-heads", "4", "--dim", "32"])
    cfg = _config_from_args(args)
    assert cfg.n_heads == 4 and cfg.n_kv_heads == 4
    assert cfg.head_dim == 8


def test_overriding_dim_rederives_ffn_hidden():
    from motherbrain.cli import _config_from_args

    default = _config_from_args(build_parser().parse_args(["params"]))
    wider = _config_from_args(build_parser().parse_args(["params", "--dim", "1024"]))
    assert wider.ffn_hidden > default.ffn_hidden


def test_preset_gqa_ratio_is_preserved_when_only_dim_changes():
    from motherbrain.cli import _config_from_args

    cfg = _config_from_args(build_parser().parse_args(["params", "--preset", "small", "--dim", "384"]))
    assert cfg.n_kv_heads == 4  # the preset's GQA ratio survives


def test_incompatible_head_override_exits_cleanly():
    from motherbrain.cli import _config_from_args

    args = build_parser().parse_args(["params", "--n-heads", "6", "--n-kv-heads", "4"])
    with pytest.raises(SystemExit):
        _config_from_args(args)
