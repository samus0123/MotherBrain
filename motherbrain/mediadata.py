"""Sounds and clips with the answer known in advance.

The same argument as imagedata.py, for the other two senses. A tower can be
wired to sound and video and read nothing at all, and the loss will fall
either way because the language half learns what captions look like. The only
honest test is whether the output changes with the input, and that needs data
whose truth is known before the model sees it.

So these are generated. Sounds are tones with a pitch, a timbre and a shape
over time; clips are a coloured shape moving in a direction. Everything else -
loudness, phase, exact speed, where the shape starts - is randomised, so the
only things that survive averaging are the things the caption names.

Both worlds are small and closed, deliberately. A model trained on them can
hear exactly this much and no more, and that limit is the largest claim the
evidence will support.
"""

from __future__ import annotations

import math
import random

import torch

# ---- sound -----------------------------------------------------------------

# Two octaves apart each, so a model that fails has failed at hearing rather
# than at splitting hairs between neighbouring notes.
PITCHES = {"low": 150.0, "middle": 600.0, "high": 2400.0}

# A sine is a single bright line; a square adds harmonics above it; noise is
# bright everywhere. Three timbres that look genuinely different as pictures.
TIMBRES = ("pure", "buzzing", "hissing")

# What the sound does over its second: nothing, rises, falls, or pulses.
SHAPES = ("steady", "rising", "falling", "pulsing")

RATE = 16000


def sound_caption(pitch: str, timbre: str, shape: str) -> str:
    return f"a {shape} {timbre} {pitch} tone"


def render_sound(pitch: str, timbre: str, shape: str, rng: random.Random,
                 seconds: float = 1.0, rate: int = RATE) -> torch.Tensor:
    """One sound as mono samples in [-1, 1].

    Loudness and phase are randomised because neither is in the caption, and a
    model that learned to read loudness would score without hearing anything.
    """
    n = int(rate * seconds)
    base = PITCHES[pitch]
    amplitude = rng.uniform(0.3, 0.9)
    phase = rng.uniform(0, 2 * math.pi)
    samples = torch.empty(n)

    for i in range(n):
        t = i / rate
        progress = i / n
        if shape == "rising":
            freq = base * (1 + progress)
        elif shape == "falling":
            freq = base * (2 - progress)
        else:
            freq = base

        angle = 2 * math.pi * freq * t + phase
        if timbre == "pure":
            value = math.sin(angle)
        elif timbre == "buzzing":
            # A square wave: the same pitch, with harmonics stacked above it.
            value = 1.0 if math.sin(angle) >= 0 else -1.0
        else:
            # Noise band: no pitch of its own, so it is shaped by the envelope
            # and reads as bright across the spectrum.
            value = rng.uniform(-1.0, 1.0) * math.sin(angle) + \
                rng.uniform(-0.6, 0.6)

        if shape == "pulsing":
            # Four beats a second, which is a pattern across time rather than
            # frequency - a different axis of the picture entirely.
            value *= 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        samples[i] = value * amplitude

    peak = samples.abs().max()
    return samples / peak if peak > 0 else samples


def sound_pairs(n: int, size: int = 64, seed: int = 0
                ) -> list[tuple[torch.Tensor, str]]:
    """`n` (spectrogram, caption) pairs. A seed fixes the set exactly."""
    from motherbrain.perception import spectrogram

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pitch = rng.choice(list(PITCHES))
        timbre = rng.choice(TIMBRES)
        shape = rng.choice(SHAPES)
        samples = render_sound(pitch, timbre, shape, rng)
        out.append((spectrogram(samples, RATE, size).squeeze(0),
                    sound_caption(pitch, timbre, shape)))
    return out


def all_sound_captions() -> list[str]:
    return [sound_caption(p, t, s)
            for s in SHAPES for t in TIMBRES for p in PITCHES]


# ---- video -----------------------------------------------------------------

MOTIONS = ("moving right", "moving left", "moving down", "moving up",
           "growing", "still")


def video_caption(colour: str, shape: str, motion: str) -> str:
    return f"a {colour} {shape} {motion}"


def render_clip(colour: str, shape: str, motion: str, rng: random.Random,
                size: int = 64, frames: int = 4):
    """A clip as the contact sheet the tower actually reads.

    Motion survives being laid out as a grid: a shape that moves right appears
    further right in each successive cell. What is lost is when it moved, and
    the docstring in perception.py says so.
    """
    from PIL import Image, ImageDraw

    from motherbrain.imagedata import BACKGROUNDS, COLOURS

    background = rng.choice(list(BACKGROUNDS.values()))
    fill = COLOURS[colour]
    # Four frames in a 2x2 sheet, not nine in a 3x3. At 64 pixels a nine-cell
    # sheet leaves each frame about twenty pixels across, and a shape drawn in
    # twenty pixels is invisible to anything downstream - measured at chance
    # before this was changed. Four cells give thirty-two pixels each, and the
    # shape is drawn large within them.
    cell = 64
    extent = rng.randint(cell // 3, (cell * 3) // 5)
    x0, y0 = rng.randint(2, max(3, cell - extent - 2)), \
        rng.randint(2, max(3, cell - extent - 2))
    travel = max(4, cell - extent - 4)

    images = []
    for f in range(frames):
        progress = f / max(frames - 1, 1)
        x, y, grow = x0, y0, extent
        if motion == "moving right":
            x = 2 + int(progress * travel)
        elif motion == "moving left":
            x = 2 + int((1 - progress) * travel)
        elif motion == "moving down":
            y = 2 + int(progress * travel)
        elif motion == "moving up":
            y = 2 + int((1 - progress) * travel)
        elif motion == "growing":
            grow = max(3, int(extent * (0.3 + 1.4 * progress)))

        img = Image.new("RGB", (cell, cell), background)
        draw = ImageDraw.Draw(img)
        box = (x, y, min(x + grow, cell - 1), min(y + grow, cell - 1))
        if shape == "circle":
            draw.ellipse(box, fill=fill)
        elif shape == "square":
            draw.rectangle(box, fill=fill)
        elif shape == "triangle":
            draw.polygon([(x + grow // 2, y), (x, y + grow),
                          (x + grow, y + grow)], fill=fill)
        else:
            half = grow // 2
            draw.polygon([(x + half, y), (x + grow, y + half),
                          (x + half, y + grow), (x, y + half)], fill=fill)
        images.append(img)
    return images


def video_pairs(n: int, size: int = 64, seed: int = 0
                ) -> list[tuple[torch.Tensor, str]]:
    from motherbrain.imagedata import COLOURS, SHAPES as IMAGE_SHAPES
    from motherbrain.perception import _to_tensor, contact_sheet

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        colour = rng.choice(list(COLOURS))
        shape = rng.choice(IMAGE_SHAPES)
        motion = rng.choice(MOTIONS)
        frames = render_clip(colour, shape, motion, rng, size)
        sheet = contact_sheet(frames, size)
        out.append((_to_tensor(sheet, size).squeeze(0),
                    video_caption(colour, shape, motion)))
    return out


def all_video_captions() -> list[str]:
    """Every clip caption. Colour and shape are shared with sight, so only the
    motion is new - which is exactly the thing video is for."""
    from motherbrain.imagedata import COLOURS, SHAPES as IMAGE_SHAPES

    return [video_caption(c, s, m)
            for m in MOTIONS for s in IMAGE_SHAPES for c in COLOURS]
