import json

import pytest

from motherbrain import SEED
from motherbrain.architecture import Architecture
from motherbrain.lineage import DEFAULT_LINEAGE_PATH, Lineage, LineageError, Release
from motherbrain.version import Version


@pytest.fixture
def lineage() -> Lineage:
    return Lineage([SEED])


def test_packaged_lineage_grows_at_every_version():
    packaged = Lineage.load(DEFAULT_LINEAGE_PATH)
    assert len(packaged) > 1
    for previous, nxt in zip(packaged, packaged.releases[1:]):
        assert nxt.version > previous.version
        assert nxt.parameter_count > previous.parameter_count


def test_evolve_appends_a_bigger_release(lineage):
    before = lineage.current
    release = lineage.evolve("minor", note="wider")
    assert release.version > before.version
    assert release.parameter_count > before.parameter_count
    assert lineage.current is release
    assert release.note == "wider"


def test_evolve_many_times_keeps_the_invariant(lineage):
    for kind in ("patch", "minor", "patch", "major", "minor", "patch"):
        lineage.evolve(kind)
    lineage.validate()
    counts = [release.parameter_count for release in lineage]
    versions = [release.version for release in lineage]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    assert versions == sorted(versions) and len(set(versions)) == len(versions)


def test_a_shrinking_lineage_is_rejected():
    smaller = Architecture.from_dict(
        {**SEED.architecture.to_dict(), "n_layers": SEED.architecture.n_layers - 1}
    )
    with pytest.raises(LineageError, match="parameters do not increase"):
        Lineage([SEED, Release(Version(0, 2, 0), smaller)])


def test_a_lineage_that_goes_backwards_in_version_is_rejected():
    bigger = Architecture.from_dict(
        {**SEED.architecture.to_dict(), "n_layers": SEED.architecture.n_layers + 1}
    )
    with pytest.raises(LineageError, match="version does not increase"):
        Lineage([SEED, Release(Version(0, 0, 9), bigger)])


def test_an_empty_lineage_is_rejected():
    with pytest.raises(LineageError):
        Lineage([])


def test_save_and_load_roundtrip(tmp_path, lineage):
    lineage.evolve("minor")
    lineage.evolve("patch")
    path = tmp_path / "lineage.json"
    lineage.save(path)

    loaded = Lineage.load(path)
    assert [r.version for r in loaded] == [r.version for r in lineage]
    assert [r.parameter_count for r in loaded] == [r.parameter_count for r in lineage]


def test_a_tampered_parameter_count_is_caught(tmp_path, lineage):
    lineage.evolve("minor")
    path = tmp_path / "lineage.json"
    lineage.save(path)

    data = json.loads(path.read_text())
    data["releases"][1]["parameters"] = 1
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="records 1 parameters"):
        Lineage.load(path)


def test_lookup_by_version(lineage):
    lineage.evolve("minor")
    assert lineage.get("0.2.0").version == Version(0, 2, 0)
    assert lineage.get(Version(0, 1, 0)) is lineage.seed
    with pytest.raises(KeyError):
        lineage.get("9.9.9")
