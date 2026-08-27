"""Training MotherBrain on your own text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .architecture import Architecture
from .data import make_dataset
from .version import Version


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    tokens_seen: int
    final_loss: float
    final_val_loss: float | None
    history: dict


def train(
    model,
    data: Path | str,
    *,
    seq_len: int = 256,
    batch_size: int = 8,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    validation_split: float = 0.1,
    seed: int = 0,
    verbose: int = 1,
) -> TrainingResult:
    """Fit ``model`` on the corpus at ``data`` with next-token prediction."""
    import keras

    context_length = model.architecture.context_length
    if seq_len > context_length:
        raise ValueError(
            f"seq_len {seq_len} exceeds the model's context length {context_length}"
        )

    train_ds, val_ds, n_tokens = make_dataset(
        data,
        seq_len=seq_len,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=verbose,
    )

    steps = sum(1 for _ in train_ds) * epochs
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    return TrainingResult(
        steps=steps,
        tokens_seen=steps * batch_size * seq_len,
        final_loss=float(losses[-1]) if losses else float("nan"),
        final_val_loss=float(val_losses[-1]) if val_losses else None,
        history=history.history,
    )


def new_model(architecture: Architecture, seed: int = 0):
    """A freshly initialised model -- random weights, no knowledge yet."""
    import keras

    from .model import build_model

    keras.utils.set_random_seed(seed)
    return build_model(architecture)


def resolve_architecture(lineage, version: Version | str | None) -> tuple[Version, Architecture]:
    """Pick the release to train: a named version, or the latest one."""
    release = lineage.get(version) if version else lineage.current
    return release.version, release.architecture
