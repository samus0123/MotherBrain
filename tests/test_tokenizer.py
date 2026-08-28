import pytest

from motherbrain.tokenizer import (
    DEFAULT_SPECIALS,
    END_OF_TEXT,
    SPLIT_PATTERN,
    ByteTokenizer,
    Tokenizer,
)

CORPUS = """
def greet(name_of_user):
    '''Say hello, politely.'''
    return f"hello {name_of_user}!"

class Thing_1:
    value = 42
    def __init__(self, x=0):
        self.x = x
""" * 40


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(CORPUS, vocab_size=512)


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "a_b__c",  # underscores: the pre-tokenizer must not drop them
        "def __init__(self, *args, **kwargs): pass",
        "héllo → 世界",
        "   leading and trailing   ",
        "\n\t\r mixed whitespace \n",
        "emoji 🧠 and \x00 control bytes",
        "1234567890",
        "",
    ],
)
def test_split_pattern_is_total(text):
    """Re-joining the pre-tokenizer's pieces must reproduce the input exactly."""
    assert "".join(SPLIT_PATTERN.findall(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "a_b__c",
        "def __init__(self, x=1): return x_y",
        "héllo → 世界 🧠",
        "   spaces   ",
        "",
        CORPUS[:1500],
    ],
)
def test_encode_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_are_single_ids(tok):
    ids = tok.encode(f"before{END_OF_TEXT}after")
    assert tok.eot_id in ids
    assert ids.count(tok.eot_id) == 1
    assert tok.decode(ids) == f"before{END_OF_TEXT}after"


def test_special_tokens_can_be_disabled(tok):
    ids = tok.encode(END_OF_TEXT, allowed_special=False)
    assert tok.eot_id not in ids
    assert tok.decode(ids) == END_OF_TEXT


def test_vocab_size_is_respected():
    t = Tokenizer.train(CORPUS, vocab_size=400)
    assert t.vocab_size <= 400
    assert len(t.special_tokens) == len(DEFAULT_SPECIALS)


def test_vocab_size_too_small_raises():
    with pytest.raises(ValueError):
        Tokenizer.train(CORPUS, vocab_size=100)


def test_merges_actually_compress(tok):
    raw = len(CORPUS.encode("utf-8"))
    assert len(tok.encode(CORPUS)) < raw / 2


def test_save_and_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(CORPUS[:500]) == tok.encode(CORPUS[:500])


def test_untrained_bytes_still_decode(tok):
    """Bytes never seen in training still survive a roundtrip."""
    text = "ẞ🜛 ௹"
    assert tok.decode(tok.encode(text)) == text


def test_byte_tokenizer_roundtrip():
    t = ByteTokenizer()
    text = "plain bytes ✓"
    assert t.decode(t.encode(text)) == text
