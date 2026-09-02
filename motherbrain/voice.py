"""Speech for the terminal console, and an honest account of when it works.

The browser console gets voice for free: the Web Speech API is built into
Chrome, Edge and Safari, so recognition and synthesis need no dependencies and
no network calls of our own. The terminal has no such guarantee - it needs a
microphone, an audio stack, and software to drive them - so this module reports
what is actually available on this machine rather than assuming, and the
console falls back to typing with a reason when something is missing.

Nothing here is installed as a dependency. Speech is optional, and a model you
can only talk to is worse than one you can also type at.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


def _module(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Capability:
    """What this machine can actually do, and what to say if it cannot."""

    speak: str | None = None      # backend name, or None
    listen: str | None = None
    reason: str = ""

    @property
    def any(self) -> bool:
        return bool(self.speak or self.listen)


def detect() -> Capability:
    """Find a speech backend without installing anything."""
    cap = Capability()

    for binary, name in (("say", "say"), ("espeak-ng", "espeak-ng"),
                         ("espeak", "espeak"), ("spd-say", "spd-say")):
        if shutil.which(binary):
            cap.speak = name
            break
    if cap.speak is None and _module("pyttsx3"):
        cap.speak = "pyttsx3"

    if _module("speech_recognition"):
        cap.listen = "speech_recognition"

    missing = []
    if cap.speak is None:
        missing.append("no speech synthesis (install espeak-ng, or pyttsx3)")
    if cap.listen is None:
        missing.append("no speech recognition "
                       "(pip install SpeechRecognition PyAudio)")
    cap.reason = "; ".join(missing)
    return cap


def speak(text: str, cap: Capability | None = None) -> bool:
    """Say `text` aloud. False when there is no way to."""
    cap = cap or detect()
    if not cap.speak or not text.strip():
        return False

    trimmed = text.strip()[:600]  # a long completion is not worth reciting
    try:
        if cap.speak == "pyttsx3":
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(trimmed)
            engine.runAndWait()
            return True
        args = {
            "say": ["say", trimmed],
            "espeak-ng": ["espeak-ng", trimmed],
            "espeak": ["espeak", trimmed],
            "spd-say": ["spd-say", "--wait", trimmed],
        }[cap.speak]
        subprocess.run(args, check=False, capture_output=True, timeout=60)
        return True
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False


def listen(cap: Capability | None = None, timeout: float = 10.0) -> str | None:
    """Capture one spoken line. None when recognition is unavailable or fails.

    Recognition runs through whatever engine SpeechRecognition is configured
    for. Its default sends audio to a Google web endpoint, so this is left to
    the caller to enable knowingly rather than being switched on by default.
    """
    cap = cap or detect()
    if not cap.listen:
        return None
    try:
        import speech_recognition as sr

        recogniser = sr.Recognizer()
        with sr.Microphone() as source:
            recogniser.adjust_for_ambient_noise(source, duration=0.3)
            audio = recogniser.listen(source, timeout=timeout,
                                      phrase_time_limit=timeout)
        return recogniser.recognize_google(audio)
    except Exception:
        return None


MENU = """What would you like to do?

  1  Tell MotherBrain what to do — by voice or by text
  2  Feed MotherBrain new information
  3  Update MotherBrain
"""


def choose_start(default: str = "tell") -> tuple[str, Capability]:
    """Show the opening menu and return the chosen action.

    Returns "tell", "feed" or "update", with the detected speech capability.
    Choosing to tell it something asks voice or text next, and only offers
    voice when this machine can actually provide it - presenting a choice that
    cannot be honoured is worse than explaining why it is missing.
    """
    cap = detect()
    print(MENU)

    try:
        answer = input("choose [1-3, default 1] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        # Not interactive - piped input, a scheduled run, no terminal at all.
        print()
        return default, cap

    choice = {
        "1": "tell", "2": "feed", "3": "update",
        "tell": "tell", "talk": "tell", "say": "tell",
        "feed": "feed", "teach": "feed", "learn": "feed",
        "update": "update", "upgrade": "update", "grow": "update",
        "": default,
    }.get(answer, default)
    print()
    return choice, cap


def choose_input(cap: Capability, default: str = "text") -> str:
    """Voice or text, asked only when voice is actually available."""
    if not cap.any:
        print(f"voice is unavailable here: {cap.reason}")
        print("using text.\n")
        return "text"

    parts = []
    if cap.listen:
        parts.append(f"speech in ({cap.listen})")
    if cap.speak:
        parts.append(f"speech out ({cap.speak})")
    print(f"voice available: {', '.join(parts)}")
    if not cap.listen:
        print("  (no dictation here, so voice reads replies aloud "
              "while you type)")

    try:
        answer = input(f"voice or text? [{default}] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return default
    mode = "voice" if answer.startswith("v") else "text"
    print(f"using {mode}.\n")
    return mode


def choose_mode(default: str = "text") -> tuple[str, Capability]:
    """Backwards-compatible input-mode question, kept for `--mode ask`.

    The opening menu (choose_start) is what a session normally sees; this
    remains for callers that only need to know how the user wants to type.
    """
    cap = detect()

    if cap.any:
        parts = []
        if cap.listen:
            parts.append(f"speech in ({cap.listen})")
        if cap.speak:
            parts.append(f"speech out ({cap.speak})")
        print(f"voice available: {', '.join(parts)}")
        if not cap.listen:
            print("  note: no recognition, so voice mode reads replies aloud "
                  "but you still type.")
        question = f"text, voice, or feed it something? [{default}] "
    else:
        print(f"voice is unavailable here: {cap.reason}")
        question = f"text, or feed it something? [{default}] "

    try:
        answer = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return default, cap

    if answer.startswith("f"):
        mode = "feed"
    elif answer.startswith("v") and cap.any:
        mode = "voice"
    else:
        mode = "text"
    print(f"using {mode}.\n" if mode != "feed" else "")
    return mode, cap
