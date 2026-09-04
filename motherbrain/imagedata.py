"""Image-caption pairs with known ground truth, rendered rather than collected.

A vision tower can be built, wired in and shipped without ever being able to
see, and nothing about the architecture will say so - the loss still falls,
because the language half alone can learn what captions tend to look like. The
only honest test is whether the model's output changes with the picture.

So the pairs here are generated, with the answer known in advance: coloured
shapes on varying backgrounds, at random positions and sizes. Position, size
and background are noise; shape and colour are the signal. A model that has
learned to see names the shape and the colour of an image it has never been
shown. One that has only learned the caption distribution cannot, and the
accuracy measured on held-out images says which happened.

This is a small, closed world - eight colours and four shapes - and a model
trained on it can see exactly that world and nothing else. That limit is the
point: it is the largest claim the evidence supports.
"""

from __future__ import annotations

import random

import torch

# Chosen to be far apart in RGB, so a model that fails has failed at seeing
# rather than at splitting hairs between two similar reds.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 70),
    "blue": (50, 90, 230),
    "yellow": (240, 210, 50),
    "purple": (150, 60, 200),
    "orange": (240, 140, 40),
    "brown": (140, 90, 50),
    "grey": (130, 130, 130),
}

SHAPES = ("circle", "square", "triangle", "diamond")

BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "pale": (225, 220, 205),
}


def caption(shape: str, colour: str) -> str:
    """The sentence a correct model should produce for this image."""
    return f"a {colour} {shape}"


def render(shape: str, colour: str, size: int, rng: random.Random):
    """Draw one shape and return it as a 3 x size x size tensor in [0, 1].

    Everything except shape and colour is randomised, so memorising pixels
    does not work: the model has to learn the two things the caption names.
    """
    from PIL import Image, ImageDraw

    background = rng.choice(list(BACKGROUNDS.values()))
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)

    # A shape between a third and two thirds of the frame, placed anywhere it
    # fits whole.
    extent = rng.randint(size // 3, (2 * size) // 3)
    x = rng.randint(0, size - extent)
    y = rng.randint(0, size - extent)
    fill = COLOURS[colour]
    box = (x, y, x + extent, y + extent)

    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    elif shape == "triangle":
        draw.polygon([(x + extent // 2, y), (x, y + extent),
                      (x + extent, y + extent)], fill=fill)
    elif shape == "diamond":
        half = extent // 2
        draw.polygon([(x + half, y), (x + extent, y + half),
                      (x + half, y + extent), (x, y + half)], fill=fill)
    else:
        raise ValueError(f"unknown shape: {shape}")

    data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return data.view(size, size, 3).permute(2, 0, 1).float() / 255.0


def pairs(n: int, size: int = 64, seed: int = 0) -> list[tuple[torch.Tensor, str]]:
    """`n` (image, caption) pairs. A given seed always gives the same set.

    Train and test differ only by seed, so the two never share an image while
    covering the same shapes and colours - which is what makes held-out
    accuracy mean anything.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        shape = rng.choice(SHAPES)
        colour = rng.choice(list(COLOURS))
        out.append((render(shape, colour, size, rng), caption(shape, colour)))
    return out


def vocabulary() -> list[str]:
    """Every word the captions can contain - what a tokenizer must cover."""
    return ["a"] + list(COLOURS) + list(SHAPES)
