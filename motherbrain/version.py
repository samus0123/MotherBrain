"""Semantic versions for MotherBrain releases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BumpKind = Literal["major", "minor", "patch"]

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True, order=True)
class Version:
    """A ``major.minor.patch`` version.

    Ordering is the usual lexicographic ordering on the three fields, which
    ``order=True`` gives us for free since they are declared in that order.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for field in ("major", "minor", "patch"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")

    @classmethod
    def parse(cls, text: str) -> "Version":
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise ValueError(f"not a major.minor.patch version: {text!r}")
        return cls(*(int(part) for part in match.groups()))

    def bump(self, kind: BumpKind) -> "Version":
        if kind == "major":
            return Version(self.major + 1, 0, 0)
        if kind == "minor":
            return Version(self.major, self.minor + 1, 0)
        if kind == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError(f"unknown bump kind: {kind!r}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
