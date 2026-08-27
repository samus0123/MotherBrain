import pytest

from motherbrain.tokenizer import ByteTokenizer


@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.mark.parametrize(
    "text",
    ["hello", "", "MotherBrain 1.0", "emoji \U0001f9e0 and accents: café", "\n\t "],
)
def test_roundtrip(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_ids_stay_in_vocabulary(tokenizer):
    ids = tokenizer.encode("anything at all \U0001f9e0", bos=True, eos=True)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_control_tokens(tokenizer):
    ids = tokenizer.encode("hi", bos=True, eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hi"


def test_a_truncated_character_does_not_crash(tokenizer):
    ids = tokenizer.encode("café")
    assert isinstance(tokenizer.decode(ids[:-1]), str)
