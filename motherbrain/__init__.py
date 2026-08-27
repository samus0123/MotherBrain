"""MotherBrain: a language model that grows with every version."""

from .architecture import Architecture
from .growth import GrowthPolicy, GrowthStep
from .lineage import DEFAULT_LINEAGE_PATH, Lineage, LineageError, Release
from .tokenizer import ByteTokenizer
from .version import Version

#: The first MotherBrain: small enough to be honest about what it is.
SEED = Release(
    version=Version(0, 1, 0),
    architecture=Architecture(
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=704,
        vocab_size=ByteTokenizer.vocab_size,
        context_length=1024,
    ),
    note="seed",
)

__all__ = [
    "Architecture",
    "ByteTokenizer",
    "DEFAULT_LINEAGE_PATH",
    "GrowthPolicy",
    "GrowthStep",
    "Lineage",
    "LineageError",
    "Release",
    "SEED",
    "Version",
]
