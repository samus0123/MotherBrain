"""The file area, stocked with real releases.

A board's file area was not a directory listing. It was a shelf of
packaged things, each one an archive with a FILE_ID.DIZ inside it - forty-
five columns by ten lines, the description the board pulled out and showed
you - and usually an .NFO beside it. That packaging is most of what made a
file area feel like a place rather than a folder.

So this builds real archives out of MotherBrain and puts them on the
shelf: the whole program, the door games as a standalone release that runs
with nothing installed, and a pack of .ANS art. Each is a genuine zip a
caller can download over XMODEM, unpack, and run.

Everything here is MotherBrain's own software, which is free and always
was. Nothing else is stocked and nothing else will be: the area is called
WAREZ because that is what a board called this shelf, not because it is a
place to put other people's work.

Archives are built once, cached in the run directory, and rebuilt when the
source they were made from is newer. Building them at startup costs about
a tenth of a second.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "motherbrain"


def area(run_dir) -> Path:
    """Where built releases live. One per run directory, like everything else."""
    return Path(run_dir) / "bbs" / "warez"


# ---- FILE_ID.DIZ ------------------------------------------------------------

# Forty-five columns, ten lines. The convention is older than most of the
# people who still follow it, and the board's listing depends on it: this is
# what it reads to describe a file it has never seen before.
DIZ_WIDTH = 45
DIZ_LINES = 10


def diz(lines: list[str]) -> str:
    """A FILE_ID.DIZ, clipped to the shape every board expected."""
    out = [line[:DIZ_WIDTH] for line in lines[:DIZ_LINES]]
    return "\r\n".join(out) + "\r\n"


def _nfo(title: str, body: list[str]) -> str:
    """The .NFO: what the release is, who made it, and what it costs."""
    bar = "=" * 62
    head = [
        f" ,--{'-' * 56}--.",
        f" |  {title.center(56)}  |",
        f" `--{'-' * 56}--'",
        "",
    ]
    tail = [
        "",
        bar,
        " MotherBrain is a language model that grows by patching itself.",
        " Every version is the last one plus a patch, and every patch is",
        " committed. It is free, it always was, and there is no crack to",
        " apply because there is nothing to crack.",
        "",
        " Build it:      pip install -e .",
        " Run it:        mb          (or python -m motherbrain)",
        " Call it:       mb bbs      then telnet to it",
        bar,
    ]
    return "\r\n".join(head + body + tail) + "\r\n"


# ---- what is on the shelf ---------------------------------------------------

def _source_files() -> list[tuple[Path, str]]:
    """The whole program: every module, the scripts, the documentation."""
    out: list[tuple[Path, str]] = []
    for path in sorted(PACKAGE.glob("*.py")):
        out.append((path, f"motherbrain/{path.name}"))
    for path in sorted((REPO / "scripts").glob("*")):
        if path.is_file():
            out.append((path, f"scripts/{path.name}"))
    for name in ("README.md", "pyproject.toml", "requirements.txt"):
        path = REPO / name
        if path.is_file():
            out.append((path, name))
    return out


# The doors need very little, and that is the point of shipping them alone:
# no torch, no server, no model. All but three play with nothing at all
# installed beyond Python.
DOOR_MODULES = ("__init__.py", "ansi.py", "doors.py", "localterm.py",
                "logic.py")

PLAYDOORS = '''#!/usr/bin/env python3
"""MotherBrain Doors - the board's whole rack, at your own keyboard.

    python PLAYDOORS.py

Needs Python 3.10 or later and nothing else. All but three play with no
model at all; THE ORACLE, TURING and THE GALLERY want one, and say so:

    python PLAYDOORS.py --run /path/to/runs/default
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motherbrain.localterm import play

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", help="a MotherBrain run directory, for the two "
                                  "doors that use the model")
parser.add_argument("--corpus", help="its corpus, if you have it")
parser.add_argument("--device", default="auto")
args = parser.parse_args()
raise SystemExit(play(args.run, args.corpus, args.device))
'''


def _ansi_pack() -> list[tuple[str, bytes]]:
    """Real .ANS files: escape codes and CP437 bytes, as an art pack was.

    Written out rather than described, because an ANSI art pack whose files
    do not open in an ANSI viewer is a directory of text with the wrong
    extension on it.
    """
    from motherbrain import ansi as A
    from motherbrain import doors

    screens: list[tuple[str, str]] = []

    logo = "\r\n".join(A.logo().split("\n"))
    screens.append(("MBLOGO.ANS",
                    logo + "\r\n" + A.shaded_bar(78) + "\r\n"))

    menu = [A.rule(78), A.centre(f"{A.HW}M O T H E R B R A I N   B B S", 78),
            A.rule(78), ""]
    menu += A.box("WHAT WOULD YOU LIKE TO DO", [
        f"  {A.HY}[1]{A.RESET}  {A.HW}Tell MotherBrain what kind of program "
        f"to make",
        f"  {A.HY}[2]{A.RESET}  {A.HW}Tell MotherBrain what to do",
        f"  {A.HY}[3]{A.RESET}  {A.HW}Teach MotherBrain something new",
        f"  {A.HY}[4]{A.RESET}  {A.HW}Apply new knowledge as a patch (update)",
        f"  {A.HY}[5]{A.RESET}  {A.HW}Run the GUI",
    ], width=74)
    screens.append(("MBMENU.ANS", "\r\n".join(menu) + "\r\n"))

    door_rows = [f"  {A.HY}[{key}]{A.RESET}  {A.HW}{A.pad(name, 12)}"
                 f"{A.GREY}{blurb}{A.RESET}"
                 for key, name, blurb in doors.CATALOGUE]
    screens.append(("MBDOORS.ANS",
                    "\r\n".join(A.box("D O O R S", door_rows, width=74,
                                      frame=A.HM)) + "\r\n"))

    shades = []
    for i, row in enumerate("░▒▓█"):
        shades.append((A.HB if i % 2 else A.HC) + row * 78 + A.RESET)
    screens.append(("MBSHADE.ANS", "\r\n".join(shades) + "\r\n"))

    # CP437, because that is what an .ANS file is. Anything the code page
    # cannot hold would arrive as a question mark in a real viewer, so it is
    # replaced here rather than silently mangled there.
    return [(name, text.encode("cp437", "replace")) for name, text in screens]


def _write(zip_path: Path, entries: list[tuple[Path | None, str, bytes | None]],
           description: str, nfo: str) -> Path:
    """One archive, with its FILE_ID.DIZ and .NFO inside it."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("FILE_ID.DIZ", description)
        archive.writestr(f"{zip_path.stem}.NFO", nfo)
        for source, name, data in entries:
            if data is not None:
                archive.writestr(name, data)
            elif source is not None and source.is_file():
                archive.write(source, name)
    tmp.replace(zip_path)
    return zip_path


def _newest(paths: list[Path]) -> float:
    best = 0.0
    for path in paths:
        try:
            best = max(best, path.stat().st_mtime)
        except OSError:
            continue
    return best


def build(run_dir, force: bool = False) -> list[Path]:
    """Build every release that is missing or out of date. Returns all of them."""
    out_dir = area(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _source_files()
    newest = _newest([p for p, _ in sources])
    built: list[Path] = []

    def stale(path: Path) -> bool:
        if force or not path.is_file():
            return True
        try:
            return path.stat().st_mtime < newest
        except OSError:
            return True

    # -- the whole program --
    whole = out_dir / "MBRAIN.ZIP"
    if stale(whole):
        _write(whole, [(p, n, None) for p, n in sources], diz([
            "MOTHERBRAIN - the whole program",
            "-" * 40,
            "A language model that grows by patching",
            "itself. Console, window, browser and a",
            "telnet BBS, all the same five options.",
            "",
            "  pip install -e .",
            "  mb            or  python -m motherbrain",
            "",
            "Weights not included - see the .PT files.",
        ]), _nfo("MOTHERBRAIN - COMPLETE SOURCE", [
            " Every module, every script, the README, and the tests it",
            " passes. The model weights are separate: they are 112MB and",
            " they are in this same file area as .PT files.",
            "",
            " Unpack it anywhere and run it:",
            "",
            "     python -m motherbrain",
            "",
            " That opens the menu without installing anything at all.",
        ]))
        built.append(whole)

    # -- the doors, standing on their own --
    door_zip = out_dir / "MBDOORS.ZIP"
    door_sources = [(PACKAGE / name, f"motherbrain/{name}")
                    for name in DOOR_MODULES]
    if stale(door_zip):
        _write(door_zip,
               [(p, n, None) for p, n in door_sources if p.is_file()]
               + [(None, "PLAYDOORS.py", PLAYDOORS.encode())],
               diz([
                   "MOTHERBRAIN DOORS - the whole rack",
                   "-" * 40,
                   "HAMURABI 1968   WUMPUS 1973   LUNAR 1969",
                   "ANIMAL 1973 - it learns and keeps it",
                   "ELIZA 1966 - Weizenbaum's DOCTOR",
                   "THE WYRM - a daily-turn RPG",
                   "GUESS, THE MAZE, ORACLE, TURING",
                   "",
                   "  python PLAYDOORS.py",
                   "Python 3.10+. Nothing else required.",
               ]), _nfo("MOTHERBRAIN DOORS", [
                   " The doors from the board, running at your own",
                   " keyboard. They were never coupled to the BBS - a door",
                   " wants a screen that takes ANSI and a keyboard that gives",
                   " up one key at a time, and that is all this provides.",
                   "",
                   "     python PLAYDOORS.py",
                   "",
                   " All of them but three need nothing except Python.",
                   " THE ORACLE, TURING and THE GALLERY want a model:",
                   "",
                   "     python PLAYDOORS.py --run /path/to/runs/default",
               ]))
        built.append(door_zip)

    # -- the art pack --
    art = out_dir / "MBANSI.ZIP"
    if stale(art):
        pack = _ansi_pack()
        _write(art, [(None, name, data) for name, data in pack], diz([
            "MOTHERBRAIN ANSI PACK",
            "-" * 40,
            "The board's screens as real .ANS files:",
            "logo, main menu, door menu, shading.",
            "",
            "Code page 437, as an .ANS file is.",
            "Open them in SyncTERM, ansilove, or",
            "any terminal set to CP437:",
            "",
            "  iconv -f cp437 -t utf-8 MBLOGO.ANS",
        ]), _nfo("MOTHERBRAIN ANSI PACK", [
            " Four screens, drawn with the IBM PC's own glyphs and the",
            " sixteen colours a 1987 client had.",
            "",
            "   MBLOGO.ANS    the login screen",
            "   MBMENU.ANS    the main menu",
            "   MBDOORS.ANS   the door menu",
            "   MBSHADE.ANS   the shading, for anyone drawing their own",
        ]))
        built.append(art)

    # -- the NFO on its own, as it always was --
    nfo = out_dir / "MBRAIN.NFO"
    if stale(nfo):
        nfo.write_text(_nfo("MOTHERBRAIN", [
            " A decoder-only mixture-of-experts transformer that grows by",
            " patching itself. It started at 18.9M parameters and is",
            " 52.2M now, across five patches, each of them committed.",
            "",
            " It can see, after a fashion: a perception tower names shapes,",
            " sounds and clips well above chance and nowhere near well.",
            " The board tells you the numbers rather than the impression.",
            "",
            " There is no crack. It is free and it always was.",
        ]), encoding="cp437", errors="replace")
        built.append(nfo)

    return sorted(out_dir.glob("*"))


# ---- descriptions -----------------------------------------------------------

# What the non-packaged files are. A raw .pt in a file area with no
# description is a 112MB mystery.
KNOWN = {
    ".pt": "MotherBrain weights. Load with `mb chat` or the board.",
    ".json": "Board or run metadata, in plain JSON.",
    ".nfo": "Release information. Text, code page 437.",
    ".py": "Python source, from MotherBrain itself.",
    ".md": "Documentation.",
    ".txt": "Text.",
    ".sh": "A shell script, for Unix.",
    ".ps1": "A PowerShell script, for Windows 10 and 11.",
}

_NAMED = {
    "motherbrain.pt": "The merged model, every patch applied. This is the "
                      "one to run.",
    "motherbrain-base.pt": "The base, v0, before any patch. Patches apply "
                           "to this.",
    "tokenizer.json": "The tokenizer. Weights are useless without it.",
}


def describe(path: Path) -> str:
    """One line about a file, from its FILE_ID.DIZ if it has one."""
    name = path.name.lower()
    if name in _NAMED:
        return _NAMED[name]
    if path.suffix.lower() == ".zip":
        text = read_diz(path)
        if text:
            for line in text.splitlines():
                cleaned = line.strip()
                if cleaned and not set(cleaned) <= set("-=_ "):
                    return cleaned
        return "An archive."
    if name.startswith("00") and path.suffix.lower() == ".pt":
        return "A patch: one version's worth of new parameters."
    return KNOWN.get(path.suffix.lower(), "A file.")


def read_diz(path: Path) -> str:
    """The FILE_ID.DIZ inside an archive, if there is one."""
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.upper().endswith("FILE_ID.DIZ"):
                    return archive.read(name).decode("cp437", "replace")
    except (OSError, zipfile.BadZipFile, KeyError):
        return ""
    return ""


def contents(path: Path, limit: int = 40) -> list[str]:
    """What is inside an archive, for a caller deciding whether to take it."""
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()[:limit]
    except (OSError, zipfile.BadZipFile):
        return []


def when(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
    except OSError:
        return "?"
