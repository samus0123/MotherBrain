"""One way in for every medium: images, sound, video, and a live feed.

MotherBrain's transformer eats a sequence of vectors. Text becomes vectors by
lookup, and an image becomes vectors by cutting it into patches - so anything
that can be made to look like an image can be read by the same tower, with no
second architecture to build, train and keep honest.

That is the whole design here:

    an image     is already an image
    a sound      becomes a mel spectrogram, which is a picture of it
    a video      becomes a grid of frames, which is a picture of the whole clip
    a live feed  is a video that has not finished

Sound genuinely works this way - a spectrogram carries pitch, timbre and
rhythm as spatial structure, and reading it as a picture is how a good deal of
audio modelling is actually done. Video as a contact sheet is cruder: it keeps
what changed between frames and throws away exactly when, which is enough to
say what is in a clip and not enough to say what happened in it.

Nothing here needs ffmpeg, torchaudio or OpenCV. WAV comes from the standard
library, GIF from Pillow, and the spectrogram from torch.stft, because a
dependency that must be installed is a dependency that will not be.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif",
                  ".tiff"}
AUDIO_SUFFIXES = {".wav", ".wave"}
VIDEO_SUFFIXES = {".gif", ".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass
class Percept:
    """Something MotherBrain has been shown, ready for the tower.

    `tensor` is always (1, 3, size, size) - the shape the vision tower reads,
    whatever it started as. `display` is the same thing as a picture for a
    person to look at, which for sound is the spectrogram itself: seeing what
    the model is reading is the only way to tell whether it is reading
    anything.
    """

    kind: str                     # "image" | "audio" | "video"
    tensor: torch.Tensor
    description: str              # what it is, in words, for the transcript
    display: object = None        # a PIL image, when there is one
    detail: dict | None = None    # duration, frame count, sample rate ...


# ---- images ----------------------------------------------------------------


def _to_tensor(img, size: int) -> torch.Tensor:
    """A PIL image as (1, 3, size, size) in [0, 1]."""
    img = img.convert("RGB").resize((size, size))
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return (raw.view(size, size, 3).permute(2, 0, 1).float() / 255.0).unsqueeze(0)


def from_image(path, size: int = 64) -> Percept:
    from PIL import Image

    img = Image.open(path)
    w, h = img.size
    return Percept("image", _to_tensor(img, size),
                   f"an image, {w}x{h}", display=img.convert("RGB"),
                   detail={"width": w, "height": h})


# ---- sound -----------------------------------------------------------------


def read_wav(path) -> tuple[torch.Tensor, int]:
    """A WAV file as mono float samples in [-1, 1], and its sample rate.

    The standard library reads WAV and nothing else, which is a real limit and
    a deliberate one: the alternative is asking for ffmpeg, and a dependency
    that must be installed is a dependency that will not be.
    """
    with wave.open(str(path), "rb") as fh:
        channels, width, rate, frames = (fh.getnchannels(), fh.getsampwidth(),
                                         fh.getframerate(), fh.getnframes())
        raw = fh.readframes(frames)

    if width == 1:                                  # unsigned 8-bit
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float()
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif width == 4:
        data = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:                                # average to mono
        data = data.view(-1, channels).mean(dim=1)
    return data, rate


def spectrogram(samples: torch.Tensor, rate: int, size: int = 64
                ) -> torch.Tensor:
    """A picture of a sound: time across, frequency up, loudness as brightness.

    Log-magnitude, because loudness is perceived logarithmically and a linear
    scale leaves everything but the loudest moment black. Normalised per clip,
    so a quiet recording and a loud one of the same thing look alike - which is
    what we want the model to notice.
    """
    if samples.numel() < 2:
        return torch.zeros(1, 3, size, size)

    n_fft = 512 if samples.numel() >= 512 else 64
    window = torch.hann_window(n_fft)
    spec = torch.stft(samples, n_fft=n_fft, hop_length=n_fft // 4,
                      window=window, return_complex=True).abs()
    spec = torch.log1p(spec)

    lo, hi = spec.min(), spec.max()
    spec = (spec - lo) / (hi - lo) if hi > lo else spec * 0

    # Frequency low-to-high reads better bottom-to-top, and the tower wants a
    # square, three-channel picture like any other.
    spec = torch.flip(spec, dims=[0]).unsqueeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(spec, size=(size, size),
                                           mode="bilinear", align_corners=False)
    return spec.repeat(1, 3, 1, 1)


def from_audio(path, size: int = 64) -> Percept:
    samples, rate = read_wav(path)
    seconds = samples.numel() / rate if rate else 0.0
    tensor = spectrogram(samples, rate, size)
    return Percept("audio", tensor,
                   f"a sound, {seconds:.1f}s at {rate:,}Hz, read as a spectrogram",
                   display=to_pil(tensor),
                   detail={"seconds": seconds, "rate": rate,
                           "samples": int(samples.numel())})


# ---- video -----------------------------------------------------------------


def frames_from_gif(path, count: int) -> list:
    """Evenly spaced frames from an animated GIF, using Pillow alone."""
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        frames = [f.convert("RGB").copy()
                  for f in ImageSequence.Iterator(img)]
    if not frames:
        return []
    if len(frames) <= count:
        return frames
    step = len(frames) / count
    return [frames[int(i * step)] for i in range(count)]


def frames_with_ffmpeg(path, count: int, size: int) -> list:
    """Frames from any video, if ffmpeg happens to be installed."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg"):
        return []
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f%03d.png"
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path),
                 "-vf", f"fps=1,scale={size}:{size}", "-frames:v", str(count),
                 str(out)],
                check=True, capture_output=True, timeout=60)
        except (subprocess.SubprocessError, OSError):
            return []
        return [Image.open(p).convert("RGB").copy()
                for p in sorted(Path(tmp).glob("*.png"))]


def contact_sheet(frames: list, size: int = 64):
    """Lay frames out as a grid - one picture standing for the whole clip."""
    from PIL import Image

    if not frames:
        return None
    side = 1
    while side * side < len(frames):
        side += 1
    cell = max(size // side, 1)
    sheet = Image.new("RGB", (cell * side, cell * side), (0, 0, 0))
    for i, frame in enumerate(frames[:side * side]):
        sheet.paste(frame.resize((cell, cell)),
                    ((i % side) * cell, (i // side) * cell))
    return sheet.resize((size, size))


def from_video(path, size: int = 64, count: int = 9) -> Percept:
    frames = frames_from_gif(path, count) if Path(path).suffix.lower() == ".gif" \
        else frames_with_ffmpeg(path, count, size)
    if not frames:
        raise ValueError(
            f"cannot read frames from {Path(path).name}. GIF works with no "
            f"extra software; other formats need ffmpeg on your PATH.")
    sheet = contact_sheet(frames, size)
    return Percept("video", _to_tensor(sheet, size),
                   f"a video, {len(frames)} frames read as one contact sheet",
                   display=sheet, detail={"frames": len(frames)})


# ---- whatever it is --------------------------------------------------------


def perceive(path, size: int = 64) -> Percept:
    """Read a file of any supported kind, chosen by what it actually is."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    suffix = p.suffix.lower()

    if suffix in AUDIO_SUFFIXES:
        return from_audio(p, size)
    if suffix == ".gif" or suffix in VIDEO_SUFFIXES - IMAGE_SUFFIXES:
        try:
            return from_video(p, size)
        except ValueError:
            if suffix in IMAGE_SUFFIXES:
                return from_image(p, size)     # a one-frame GIF is a picture
            raise
    if suffix in IMAGE_SUFFIXES:
        return from_image(p, size)
    raise ValueError(
        f"{p.name}: MotherBrain reads images ({', '.join(sorted(IMAGE_SUFFIXES))}), "
        f"WAV sound, and GIF video. Other video needs ffmpeg installed.")


def to_pil(tensor: torch.Tensor):
    """A model-shaped tensor back to something a person can look at."""
    from PIL import Image

    array = (tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255)
    return Image.fromarray(array.to(torch.uint8).numpy())
