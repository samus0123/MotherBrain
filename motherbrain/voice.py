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
import sys
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
    """Find a speech backend without installing anything.

    Windows ships one: System.Speech, reachable through PowerShell, so voice
    works there with nothing installed. Unix has no such guarantee, which is
    why the others are looked for rather than assumed.
    """
    cap = Capability()

    if sys.platform == "win32" and shutil.which("powershell"):
        cap.speak = "powershell"
    if cap.speak is None:
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
        missing.append("no speech synthesis (install espeak-ng, or pyttsx3)"
                       if sys.platform != "win32" else
                       "no speech synthesis (powershell not found)")
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
        if cap.speak == "powershell":
            # System.Speech is part of Windows; the text is passed on stdin
            # rather than the command line so quotes in it cannot break out.
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$s.Speak([Console]::In.ReadToEnd())")
            subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           input=trimmed, text=True, check=False,
                           capture_output=True, timeout=60)
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

  1  Tell MotherBrain what kind of program to make    (text or voice)
  2  Tell MotherBrain what to do                      (text or voice)
  3  Teach MotherBrain something new
  4  Apply new knowledge as a patch (update)
"""

# What each answer means. Digits are the documented way in; the words exist
# because people type the thing they want rather than its number.
_CHOICES = {
    "1": "make", "make": "make", "program": "make", "code": "make",
    "write a program": "make", "build": "make",

    "2": "do", "do": "do", "tell": "do", "command": "do",
    "text": "do", "type": "do", "voice": "do", "speak": "do", "talk": "do",

    "3": "learn", "learn": "learn", "teach": "learn", "feed": "learn",
    "new": "learn", "information": "learn",

    "4": "apply", "apply": "apply", "patch": "apply", "update": "apply",
    "grow": "apply", "ascend": "apply", "version": "apply",
}

# Two answers name a way of talking as well as a task, and mean it.
_IMPLIED_MODE = {"voice": "voice", "speak": "voice", "talk": "voice",
                 "text": "text", "type": "text"}


def choose_start(default: str = "do") -> tuple[str, str, Capability]:
    """Show the opening menu and return (action, mode, capability).

    The action is "make", "do", "learn" or "apply" - the four options, in
    order. The mode is "text" or "voice", and only options 1 and 2 ask for it,
    because teaching and patching are not conversations.

    Choosing voice on a machine that cannot hear falls back to text and says
    why. Honouring the choice matters more than pretending to.
    """
    cap = detect()
    print(MENU)

    try:
        answer = input("choose [1-4, default 2] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        # Not interactive - piped input, a scheduled run, no terminal at all.
        print()
        return default, "text", cap

    action = _CHOICES.get(answer, default if answer else default)
    mode = _IMPLIED_MODE.get(answer, "")

    if action in ("make", "do") and not mode:
        mode = _ask_mode(cap)
    elif not mode:
        mode = "text"

    if mode == "voice":
        mode = _honour_voice(cap)
    print()
    return action, mode, cap


def _ask_mode(cap: Capability) -> str:
    """Text or voice, asked only when voice is actually possible."""
    if not cap.any:
        return "text"
    try:
        answer = input("text or voice? [text] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return "text"
    return "voice" if answer.startswith(("v", "s", "t a")) else "text"


def _honour_voice(cap: Capability) -> str:
    """Downgrade to text when this machine cannot do voice, and say so."""
    if not cap.any:
        print(f"voice is unavailable here: {cap.reason}")
        print("using text instead.")
        return "text"
    if not cap.listen:
        print(f"speech out only ({cap.speak}); no dictation here, so replies "
              f"are read aloud while you type.")
    return "voice"


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
