"""The evolutionary record of MotherBrain releases.

A lineage is an ordered chain of releases. Each release is grown from its
predecessor, so both the version and the parameter count only ever go up.
That invariant is enforced when a lineage is built *and* when one is loaded
back from disk, so a hand-edited file cannot smuggle in a shrinking model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .architecture import Architecture
from .growth import GrowthPolicy
from .version import BumpKind, Version

DEFAULT_LINEAGE_PATH = Path(__file__).with_name("lineage.json")


@dataclass(frozen=True)
class Release:
    """One version of the model."""

    version: Version
    architecture: Architecture
    note: str = ""

    @property
    def parameter_count(self) -> int:
        return self.architecture.parameter_count

    def to_dict(self) -> dict:
        return {
            "version": str(self.version),
            "note": self.note,
            "architecture": self.architecture.to_dict(),
            # Recorded for readability; recomputed from the architecture on
            # load and checked against this value.
            "parameters": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Release":
        architecture = Architecture.from_dict(data["architecture"])
        recorded = data.get("parameters")
        if recorded is not None and recorded != architecture.parameter_count:
            raise ValueError(
                f"release {data['version']} records {recorded} parameters but "
                f"its architecture implies {architecture.parameter_count}"
            )
        return cls(
            version=Version.parse(data["version"]),
            architecture=architecture,
            note=data.get("note", ""),
        )


class LineageError(ValueError):
    """Raised when a lineage would stop growing."""


class Lineage:
    """An ordered, strictly growing chain of releases."""

    def __init__(
        self,
        releases: Sequence[Release],
        policy: GrowthPolicy | None = None,
    ) -> None:
        if not releases:
            raise LineageError("a lineage needs at least one release")
        self._releases = list(releases)
        self.policy = policy or GrowthPolicy()
        self.validate()

    # -- access ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._releases)

    def __iter__(self) -> Iterator[Release]:
        return iter(self._releases)

    def __getitem__(self, index: int) -> Release:
        return self._releases[index]

    @property
    def releases(self) -> list[Release]:
        return list(self._releases)

    @property
    def seed(self) -> Release:
        return self._releases[0]

    @property
    def current(self) -> Release:
        return self._releases[-1]

    def get(self, version: Version | str) -> Release:
        wanted = Version.parse(version) if isinstance(version, str) else version
        for release in self._releases:
            if release.version == wanted:
                return release
        raise KeyError(f"no release {wanted} in this lineage")

    # -- invariant ------------------------------------------------------

    def validate(self) -> None:
        """Check that every step increases both version and parameters."""
        for previous, nxt in zip(self._releases, self._releases[1:]):
            if nxt.version <= previous.version:
                raise LineageError(
                    f"version does not increase: {previous.version} -> {nxt.version}"
                )
            if nxt.parameter_count <= previous.parameter_count:
                raise LineageError(
                    f"parameters do not increase from {previous.version} to "
                    f"{nxt.version}: {previous.parameter_count} -> "
                    f"{nxt.parameter_count}"
                )

    # -- evolution ------------------------------------------------------

    def evolve(self, kind: BumpKind, note: str = "") -> Release:
        """Grow the model by one bump and append the resulting release."""
        current = self.current
        release = Release(
            version=current.version.bump(kind),
            architecture=self.policy.grow(current.architecture, kind),
            note=note,
        )
        self._releases.append(release)
        self.validate()
        return release

    # -- persistence ----------------------------------------------------

    def to_dict(self) -> dict:
        return {"releases": [release.to_dict() for release in self._releases]}

    @classmethod
    def from_dict(cls, data: dict, policy: GrowthPolicy | None = None) -> "Lineage":
        releases = [Release.from_dict(entry) for entry in data["releases"]]
        return cls(releases, policy=policy)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_LINEAGE_PATH) -> "Lineage":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: Path | str = DEFAULT_LINEAGE_PATH) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
