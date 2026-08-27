"""MotherBrain: a scalable, trainable, servable language model."""

__version__ = "0.1.0"

from motherbrain.config import ModelConfig, PRESETS, human  # noqa: F401

__all__ = ["ModelConfig", "PRESETS", "human", "__version__"]
