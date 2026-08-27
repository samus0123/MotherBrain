import pytest

from motherbrain.version import Version


def test_parse_roundtrip():
    assert str(Version.parse("1.2.3")) == "1.2.3"
    assert Version.parse("1.2.3") == Version(1, 2, 3)


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "a.b.c", ""])
def test_parse_rejects_junk(bad):
    with pytest.raises(ValueError):
        Version.parse(bad)


def test_ordering_is_semantic():
    assert Version(0, 9, 9) < Version(1, 0, 0)
    assert Version(1, 0, 0) < Version(1, 0, 1) < Version(1, 1, 0)
    assert Version(2, 0, 0) > Version(1, 99, 99)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("major", Version(2, 0, 0)),
        ("minor", Version(1, 3, 0)),
        ("patch", Version(1, 2, 4)),
    ],
)
def test_bump(kind, expected):
    assert Version(1, 2, 3).bump(kind) == expected


def test_every_bump_increases_the_version():
    start = Version(1, 2, 3)
    for kind in ("major", "minor", "patch"):
        assert start.bump(kind) > start


def test_unknown_bump_kind():
    with pytest.raises(ValueError):
        Version(1, 0, 0).bump("mega")
