"""MotherBrain — a personal large language model, built from scratch.

Public surface:
    ModelConfig, TrainConfig, RunConfig  — configuration
    MotherBrain                          — the model
    Tokenizer, ByteTokenizer             — byte-level BPE
    generate                             — sampling
    train                                — the training loop
"""

from .config import ModelConfig, RunConfig, TrainConfig

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "RunConfig",
    "MotherBrain",
    "Tokenizer",
    "ByteTokenizer",
    "generate",
    "train",
    "__version__",
]


def __getattr__(name: str):
    # Torch-backed pieces are imported lazily so `import motherbrain` stays cheap
    # and the config/scaling tools work without torch installed.
    if name == "MotherBrain":
        from .model import MotherBrain

        return MotherBrain
    if name in ("Tokenizer", "ByteTokenizer"):
        from . import tokenizer as _t

        return getattr(_t, name)
    if name == "generate":
        from .sample import generate

        return generate
    if name == "train":
        from .train import train

        return train
    raise AttributeError(f"module 'motherbrain' has no attribute '{name}'")
