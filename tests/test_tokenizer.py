import pytest

from motherbrain.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Tokenizer

pytest.importorskip("tokenizers")

CORPUS = [
    "the cat sat on the mat and looked at the moon",
    "a dog ran through the tall grass in the morning",
    "engineers build machines that learn from data",
] * 40


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok")
    corpus = path / "corpus.txt"
    corpus.write_text("\n".join(CORPUS), encoding="utf-8")
    return Tokenizer.train([corpus], vocab_size=300, out_path=path / "tokenizer.json")


def test_special_tokens_take_the_first_ids(tokenizer):
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)


def test_roundtrip_is_lossless(tokenizer):
    text = "the cat sat on the mat"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_level_handles_unseen_characters(tokenizer):
    """Byte-level BPE has no OOV: even unseen scripts round-trip."""
    text = "こんにちは — café!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bos_and_eos_are_added_on_request(tokenizer):
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.encode("hello") == ids[1:-1]


def test_decode_skips_special_tokens_by_default(tokenizer):
    ids = tokenizer.encode("the cat", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "the cat"


def test_batch_encoding_matches_single(tokenizer):
    texts = ["the cat", "a dog ran"]
    batch = tokenizer.encode_batch(texts, add_bos=True)
    assert batch == [tokenizer.encode(t, add_bos=True) for t in texts]


def test_stream_encode_yields_one_list_per_document(tokenizer):
    docs = ["the cat", "a dog", "engineers build"]
    out = list(tokenizer.stream_encode(docs, add_bos=True, add_eos=True, chunk=2))
    assert len(out) == 3
    assert all(ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id for ids in out)


def test_vocab_size_covers_specials(tokenizer):
    assert tokenizer.vocab_size >= len([PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])


def test_save_and_reload(tokenizer, tmp_path):
    out = tmp_path / "copy.json"
    tokenizer._tok.save(str(out))
    reloaded = Tokenizer.load(out)
    assert reloaded.encode("the cat") == tokenizer.encode("the cat")


def test_load_accepts_a_directory(tokenizer, tmp_path):
    (tmp_path / "tokenizer.json").write_text(tokenizer._tok.to_str(), encoding="utf-8")
    assert Tokenizer.load(tmp_path).vocab_size == tokenizer.vocab_size


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer.load(tmp_path / "nope.json")


def test_training_without_files_raises(tmp_path):
    with pytest.raises(ValueError, match="no training files"):
        Tokenizer.train([], vocab_size=100, out_path=tmp_path / "t.json")
