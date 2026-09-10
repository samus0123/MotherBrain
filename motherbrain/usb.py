"""MotherBrain on a stick: the whole board, from a USB drive.

Everything lives on the drive and nothing is written to the machine you
plug it into - not the model, not the corpus, not the message base, not
even Python's bytecode cache. Pull the stick out and the computer is as it
was.

**About opening by itself.** It cannot, and no software on the stick can
make it. Windows switched AutoRun off for removable drives in 2011, after
Conficker spread on exactly that mechanism; macOS never had it; Linux
desktops offer to open a file manager and nothing more. Any page that
claims otherwise is describing 2009. What this builds instead:

* Launchers at the root of the drive, one per platform, so opening the
  drive and double-clicking one starts the board.
* An `autorun.inf` giving the drive a name and an icon in Explorer, and
  declaring the double-click action - the parts of AutoRun that still work.
* Three opt-in installers, one per platform, for people who do want it to
  start on insert. They set that up on *their* machine, once, knowingly -
  which is the only way it can now happen, and the right way round.

The launchers bootstrap: they find a Python, make a virtual environment on
the drive if there is not one, install what is missing, and start the
board. After the first run there is nothing to wait for.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DRIVE_LABEL = "MOTHERBRAIN"

# ---- the launchers ----------------------------------------------------------

WINDOWS_LAUNCHER = r"""@echo off
rem  MotherBrain BBS - Windows 10 and 11
rem  Everything happens on this drive. Nothing is installed on the computer.
setlocal enabledelayedexpansion
cd /d "%~dp0MotherBrain"

set "MB_HOME=%CD%"
set "MB_WORKSPACE=%CD%"
set "PYTHONPYCACHEPREFIX=%CD%\cache\pyc"
set "PIP_CACHE_DIR=%CD%\cache\pip"
set "PYTHONDONTWRITEBYTECODE="

title MotherBrain BBS

rem ---- find a Python -------------------------------------------------------
set "PY="
if exist "%CD%\python\python.exe" set "PY=%CD%\python\python.exe"
if not defined PY if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY (
    where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo   MotherBrain needs Python 3.10 or later, and this computer has none.
    echo.
    echo   Either install it from https://python.org  ^(tick "Add to PATH"^),
    echo   or put a portable Python in:   %CD%\python\
    echo.
    pause
    exit /b 1
)

rem ---- first run: build the environment on the drive ------------------------
if not exist "%CD%\.venv\Scripts\python.exe" (
    echo.
    echo   First run on this computer. Building the environment on the drive.
    echo   This happens once and needs the internet. A few minutes.
    echo.
    %PY% -m venv "%CD%\.venv" || goto :novenv
    "%CD%\.venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
    "%CD%\.venv\Scripts\python.exe" -m pip install -r "%CD%\requirements.txt"
    if errorlevel 1 goto :nodeps
)
set "PY=%CD%\.venv\Scripts\python.exe"

rem ---- start the board, then call it ---------------------------------------
echo.
echo   Starting MotherBrain. The board answers on port 2323 of this computer.
echo.
start "MotherBrain board" /min "%PY%" -m motherbrain bbs --port 2323 --workspace "%CD%"
"%PY%" -m motherbrain wait 2323
"%PY%" -m motherbrain call 127.0.0.1 2323
goto :eof

:novenv
echo   Could not build a virtual environment. Is this Python 3.10 or later?
pause
exit /b 1

:nodeps
echo.
echo   Could not install what MotherBrain needs. Usually that is no internet.
echo   The doors still play without any of it:
echo.
echo       %PY% -m motherbrain doors
echo.
pause
exit /b 1
"""

UNIX_LAUNCHER = r"""#!/usr/bin/env sh
#  MotherBrain BBS - macOS and Linux
#  Everything happens on this drive. Nothing is installed on the computer.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/MotherBrain"

export MB_HOME="$PWD"
export MB_WORKSPACE="$PWD"
export PYTHONPYCACHEPREFIX="$PWD/cache/pyc"
export PIP_CACHE_DIR="$PWD/cache/pip"

# ---- find a Python ----------------------------------------------------------
if [ -x "$PWD/python/bin/python3" ]; then
    PY="$PWD/python/bin/python3"
elif [ -x "$PWD/.venv/bin/python" ]; then
    PY="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    cat <<'MSG'

  MotherBrain needs Python 3.10 or later, and this computer has none.

    Debian, Ubuntu, Kali   sudo apt install python3 python3-venv
    Fedora                 sudo dnf install python3
    macOS                  brew install python@3.12
                           (or install it from https://python.org)

MSG
    exit 1
fi

# ---- first run: build the environment on the drive --------------------------
if [ ! -x "$PWD/.venv/bin/python" ]; then
    echo
    echo "  First run on this computer. Building the environment on the drive."
    echo "  This happens once and needs the internet. A few minutes."
    echo
    "$PY" -m venv "$PWD/.venv" || {
        echo "  Could not build a virtual environment."
        echo "  On Debian and Ubuntu that usually means:  sudo apt install python3-venv"
        exit 1
    }
    "$PWD/.venv/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true
    if ! "$PWD/.venv/bin/python" -m pip install -r "$PWD/requirements.txt"; then
        cat <<'MSG'

  Could not install what MotherBrain needs. Usually that is no internet.
  The doors still play without any of it:

      ./MotherBrain/.venv/bin/python -m motherbrain doors

MSG
        exit 1
    fi
fi
PY="$PWD/.venv/bin/python"

# ---- start the board, then call it ------------------------------------------
echo
echo "  Starting MotherBrain. The board answers on port 2323 of this computer."
echo

"$PY" -m motherbrain bbs --port 2323 --workspace "$PWD" >"$PWD/cache/board.log" 2>&1 &
BOARD=$!
trap 'kill "$BOARD" 2>/dev/null || true' EXIT INT TERM

"$PY" -m motherbrain wait 2323
"$PY" -m motherbrain call 127.0.0.1 2323
"""

# The .command extension is what makes a shell script double-clickable in
# the macOS Finder. It is the same script.
MAC_LAUNCHER = UNIX_LAUNCHER


AUTORUN = """[autorun]
; Windows has ignored the `open=` line on removable drives since 2011, and
; that is a good thing - it is how Conficker spread. What still works is
; the label, the icon, and the action offered when you open the drive.
label={label}
icon=MOTHERBRAIN.ico
action=Start the MotherBrain BBS
open=MOTHERBRAIN.bat

[Content]
MusicFiles=false
PictureFiles=false
VideoFiles=false
"""


START_HERE = """MOTHERBRAIN - The Bulletin Board System
=======================================================================

A language model, a 1980s bulletin board, and everything they need, on
this drive. Nothing is installed on the computer you plug it into, and
nothing is written to it: the model, the message base, the file area and
even Python's cache all live here.

  Windows      double-click  MOTHERBRAIN.bat
  macOS        double-click  MOTHERBRAIN.command
  Linux        run           ./motherbrain.sh

The first run on a new computer builds a Python environment on this drive
and needs the internet for a few minutes. Every run after that is instant,
on that computer or any other.


DOES IT OPEN BY ITSELF WHEN I PLUG IT IN?
-----------------------------------------------------------------------
No, and nothing on this drive can make it.

Windows switched AutoRun off for USB drives in 2011, after the Conficker
worm spread by exactly that route. macOS never had it. Linux desktops will
offer to open a file manager and nothing more. A USB stick that silently
runs software when you insert it is the thing all three of them were
changed to prevent.

What you get instead: the drive shows up named MOTHERBRAIN with its own
icon, and opening it puts the launcher in front of you.

If you do want it on insert, that is a setting on YOUR computer, not on
this drive, and there is an installer here for each:

  autostart/windows-install.cmd     a Task Scheduler task, run once as
                                    administrator
  autostart/linux-install.sh        a udev rule
  autostart/macos-install.sh        a launchd agent watching /Volumes

Each of them prints exactly what it will do before it does it, and each
has a matching -uninstall next to it. Read them first. Something that
starts a program when you plug hardware in is worth reading first.


WHAT IS ON HERE
-----------------------------------------------------------------------
  MotherBrain/motherbrain/    the program: model, board, doors, everything
  MotherBrain/models/         the weights
  MotherBrain/runs/default/   the base, the patches, and the board's state
  MotherBrain/data/corpus/    what it has read (only if you asked for it)
  MotherBrain/.venv/          built on first run, on this drive


ONCE IT IS RUNNING
-----------------------------------------------------------------------
The launcher starts the board and dials it for you. You will get a login
prompt: type NEW the first time.

  Ctrl-]        hang up
  G             log off from the menu
  ?             the menu, if you turned expert mode on

Other ways in, from MotherBrain/ on this drive:

  .venv/bin/python -m motherbrain            the menu, in a terminal
  .venv/bin/python -m motherbrain gui        a window
  .venv/bin/python -m motherbrain doors      the ten door games
  .venv/bin/python -m motherbrain bbs        the board, on its own

The doors need nothing but Python - no model, no environment - so they
work on a machine with no internet and a bare Python 3.10.


IF YOU LOSE THE DRIVE
-----------------------------------------------------------------------
Everything on it is on the machine it was made from, and the model and its
patches are in the repository it came from. The one thing that is only
here is the board's own state: the users, the messages, the file area and
the uploads, in MotherBrain/runs/default/bbs/. Copy that directory if it
matters to you.
"""


# ---- opt-in autostart, one per platform -------------------------------------

WINDOWS_AUTOSTART = r"""@echo off
rem  Make MotherBrain start when this drive is plugged into THIS computer.
rem
rem  What this does, exactly: registers a Scheduled Task that watches the
rem  Windows event log for a removable drive arriving, and runs the launcher
rem  on this drive when one does. It changes this computer, not the drive.
rem
rem  Undo it with:  windows-uninstall.cmd
rem
net session >nul 2>&1
if errorlevel 1 (
    echo   This has to run as administrator: right-click, "Run as administrator".
    pause
    exit /b 1
)
set "LAUNCH=%~dp0..\MOTHERBRAIN.bat"
echo.
echo   This will register a Scheduled Task named "MotherBrain USB" that runs
echo.
echo       %LAUNCH%
echo.
echo   when a removable drive is connected. Ctrl-C now if that is not what
echo   you want.
echo.
pause
schtasks /Create /TN "MotherBrain USB" /TR "\"%LAUNCH%\"" /SC ONEVENT ^
    /EC System /MO "*[System[Provider[@Name='Microsoft-Windows-Kernel-PnP'] and EventID=410]]" ^
    /RL LIMITED /F
echo.
echo   Done. Unplug and replug the drive to test it.
echo   Note: Windows runs the task for ANY removable drive, and the launcher
echo   only works when this one is present - so nothing happens otherwise.
pause
"""

WINDOWS_AUTOSTART_UNDO = r"""@echo off
net session >nul 2>&1
if errorlevel 1 (
    echo   Run this as administrator.
    pause
    exit /b 1
)
schtasks /Delete /TN "MotherBrain USB" /F
echo   Removed.
pause
"""

LINUX_AUTOSTART = r"""#!/usr/bin/env sh
#  Make MotherBrain start when this drive is plugged into THIS computer.
#
#  What this does, exactly: writes one udev rule to /etc/udev/rules.d that
#  matches this drive by its filesystem label and runs the launcher. It
#  changes this computer, not the drive.
#
#  Undo it with:  sudo ./linux-uninstall.sh
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCH="$HERE/../motherbrain.sh"
RULE=/etc/udev/rules.d/99-motherbrain-usb.rules

cat <<MSG

  This will write $RULE containing:

    ACTION=="add", SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="MOTHERBRAIN", \
      RUN+="/bin/sh -c '$LAUNCH &'"

  and reload udev. It changes this computer, not the drive.
  Ctrl-C now if that is not what you want.

MSG
printf '  Press enter to continue: '
read -r _

[ "$(id -u)" -eq 0 ] || { echo "  Run it with sudo."; exit 1; }

cat > "$RULE" <<RULEEOF
ACTION=="add", SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="MOTHERBRAIN", RUN+="/bin/sh -c '$LAUNCH &'"
RULEEOF
udevadm control --reload-rules
echo
echo "  Done. Unplug and replug the drive to test it."
echo "  Note: udev runs it without a terminal, so the board starts but"
echo "  nothing is drawn. Dial it yourself with:  mb call 127.0.0.1 2323"
"""

LINUX_AUTOSTART_UNDO = r"""#!/usr/bin/env sh
set -eu
[ "$(id -u)" -eq 0 ] || { echo "  Run it with sudo."; exit 1; }
rm -f /etc/udev/rules.d/99-motherbrain-usb.rules
udevadm control --reload-rules
echo "  Removed."
"""

MAC_AUTOSTART = r"""#!/usr/bin/env sh
#  Make MotherBrain start when this drive is plugged into THIS Mac.
#
#  What this does, exactly: installs a launchd agent that watches /Volumes
#  and runs the launcher when a volume named MOTHERBRAIN appears. It changes
#  this Mac, not the drive.
#
#  Undo it with:  ./macos-uninstall.sh
set -eu

PLIST="$HOME/Library/LaunchAgents/com.motherbrain.usb.plist"
WATCH="/Volumes/MOTHERBRAIN/MOTHERBRAIN.command"

cat <<MSG

  This will write $PLIST, a launchd agent that watches /Volumes and runs

      $WATCH

  when a volume named MOTHERBRAIN appears. It changes this Mac, not the
  drive. Ctrl-C now if that is not what you want.

MSG
printf '  Press enter to continue: '
read -r _

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.motherbrain.usb</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string><string>-c</string>
    <string>[ -x "$WATCH" ] &amp;&amp; open -a Terminal "$WATCH"</string>
  </array>
  <key>WatchPaths</key><array><string>/Volumes</string></array>
</dict>
</plist>
PLISTEOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo
echo "  Done. Eject and replug the drive to test it."
"""

MAC_AUTOSTART_UNDO = r"""#!/usr/bin/env sh
set -eu
PLIST="$HOME/Library/LaunchAgents/com.motherbrain.usb.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "  Removed."
"""


def icon(path: Path) -> bool:
    """A drive icon, drawn rather than shipped. False if Pillow is missing."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, size - 8, size - 8), radius=36,
                           fill=(10, 18, 46, 255), outline=(0, 190, 255, 255),
                           width=6)
    # Block letters, in the same shape the board's logo is drawn from.
    bar = 20
    for x in (52, 92):                                   # M
        draw.rectangle((x, 70, x + bar, 186), fill=(0, 220, 255, 255))
    draw.polygon([(52, 70), (72, 70), (92, 120), (72, 120)],
                 fill=(0, 220, 255, 255))
    draw.polygon([(112, 70), (92, 70), (72, 120), (92, 120)],
                 fill=(0, 220, 255, 255))
    draw.rectangle((146, 70, 166, 186), fill=(120, 160, 255, 255))   # B
    draw.rounded_rectangle((146, 70, 208, 124), radius=22,
                           outline=(120, 160, 255, 255), width=18)
    draw.rounded_rectangle((146, 128, 208, 186), radius=24,
                           outline=(120, 160, 255, 255), width=18)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                            (128, 128), (256, 256)])
    return True


def _write(path: Path, text: str, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # CRLF for the files Windows reads, LF for the ones a shell reads. A
    # .sh with CRLF in it fails with "bad interpreter", which is a
    # miserable first impression.
    newline = "\r\n" if path.suffix.lower() in (".bat", ".cmd", ".inf",
                                                ".txt") else "\n"
    path.write_text(text, encoding="utf-8", newline=newline)
    if executable:
        path.chmod(0o755)
    return path


def build(dest, run_dir: str, corpus_dir: str, device: str = "cpu",
          with_corpus: bool = False, label: str = DRIVE_LABEL) -> list[str]:
    """Lay out a complete, portable MotherBrain on `dest`. Returns what it did."""
    from motherbrain.cli import export_model, shipped_base

    dest = Path(dest).expanduser()
    home = dest / "MotherBrain"
    run = Path(run_dir)
    done: list[str] = []

    for directory in (home / "models", home / "runs" / "default" / "patches",
                      home / "cache", dest / "autostart"):
        directory.mkdir(parents=True, exist_ok=True)

    # -- the program --
    package = Path(__file__).resolve().parent
    shutil.copytree(package, home / "motherbrain",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    dirs_exist_ok=True)
    repo = package.parent
    for name in ("requirements.txt", "README.md", "pyproject.toml"):
        if (repo / name).is_file():
            shutil.copy2(repo / name, home / name)
    if (repo / "scripts").is_dir():
        shutil.copytree(repo / "scripts", home / "scripts", dirs_exist_ok=True)
    done.append("the program")

    # -- the model, its base, and every patch --
    base = shipped_base(run_dir)
    if base is not None:
        shutil.copy2(base, home / "models" / "motherbrain-base.pt")
        done.append("the base model")
    for name in ("config.json", "tokenizer.json", "versions.json"):
        if (run / name).is_file():
            shutil.copy2(run / name, home / "runs" / "default" / name)
    patches = sorted((run / "patches").glob("*.pt"))
    for patch in patches:
        shutil.copy2(patch, home / "runs" / "default" / "patches" / patch.name)
    if patches:
        done.append(f"{len(patches)} patch(es)")

    merged = home / "models" / "motherbrain.pt"
    try:
        export_model(run_dir, merged, device=device, corpus_dir=corpus_dir)
        done.append("the merged model")
    except Exception as exc:                              # noqa: BLE001
        done.append(f"no merged model ({exc})")

    if with_corpus and Path(corpus_dir).is_dir():
        shutil.copytree(corpus_dir, home / "data" / "corpus",
                        dirs_exist_ok=True)
        done.append("the corpus")

    # -- the launchers, at the root where a person will look --
    _write(dest / "MOTHERBRAIN.bat", WINDOWS_LAUNCHER)
    _write(dest / "MOTHERBRAIN.command", MAC_LAUNCHER, executable=True)
    _write(dest / "motherbrain.sh", UNIX_LAUNCHER, executable=True)
    _write(dest / "autorun.inf", AUTORUN.format(label=label))
    _write(dest / "START HERE.txt", START_HERE)
    done.append("launchers for Windows, macOS and Linux")

    if icon(dest / "MOTHERBRAIN.ico"):
        done.append("a drive icon")

    # -- the opt-in autostart installers --
    _write(dest / "autostart" / "windows-install.cmd", WINDOWS_AUTOSTART)
    _write(dest / "autostart" / "windows-uninstall.cmd", WINDOWS_AUTOSTART_UNDO)
    _write(dest / "autostart" / "linux-install.sh", LINUX_AUTOSTART,
           executable=True)
    _write(dest / "autostart" / "linux-uninstall.sh", LINUX_AUTOSTART_UNDO,
           executable=True)
    _write(dest / "autostart" / "macos-install.sh", MAC_AUTOSTART,
           executable=True)
    _write(dest / "autostart" / "macos-uninstall.sh", MAC_AUTOSTART_UNDO,
           executable=True)
    done.append("opt-in autostart installers")

    return done


def size_on_disk(dest) -> int:
    return sum(f.stat().st_size for f in Path(dest).rglob("*") if f.is_file())
