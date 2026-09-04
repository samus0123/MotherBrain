#!/usr/bin/env sh
# Install MotherBrain, working around the things that usually break.
#
# A plain `pip install -e .` fails on several common setups for reasons that
# have nothing to do with this project:
#
#   * Debian 12+, Ubuntu 23.04+ and Termux's proot images mark the system
#     Python as externally managed (PEP 668), so pip refuses to touch it.
#   * PyTorch only publishes wheels for some Python versions. On a very new
#     Python there is no wheel to install and pip says the requirement cannot
#     be satisfied, which reads as if the package does not exist.
#
# This script uses a virtual environment, which avoids the first entirely, and
# checks for the second before spending several hundred megabytes finding out.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || {
  echo "error: $PY not found. Install Python 3.10 or newer first."
  exit 1
}

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
MAJOR=$(echo "$VER" | cut -d. -f1)
MINOR=$(echo "$VER" | cut -d. -f2)
echo "python $VER at $(command -v "$PY")"

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
  echo "error: MotherBrain needs Python 3.10 or newer; this is $VER."
  exit 1
fi
if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -gt 13 ]; then
  echo
  echo "warning: PyTorch may not publish wheels for Python $VER yet."
  echo "If the torch install fails, install an older Python and re-run with:"
  echo "    PYTHON=python3.12 scripts/install.sh"
  echo
fi

VENV=${VENV:-.venv}
if [ ! -d "$VENV" ]; then
  echo "creating virtual environment in $VENV"
  "$PY" -m venv "$VENV" || {
    echo
    echo "error: could not create a virtual environment."
    echo "On Debian and Ubuntu: apt install python3-venv"
    exit 1
  }
fi

ARCH=$(uname -m 2>/dev/null || echo unknown)
if [ "${MB_CPU_ONLY:-}" = "1" ]; then
  echo "installing PyTorch (CPU build, ~200MB)"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
elif [ "$ARCH" = "x86_64" ]; then
  # On x86_64 the default wheel drags in the whole CUDA runtime, which is
  # around 2.5GB and useless without an NVIDIA GPU.
  echo "installing (PyTorch on x86_64 pulls the CUDA runtime, ~2.5GB)"
  echo "  no GPU? re-run as:  MB_CPU_ONLY=1 scripts/install.sh   (~200MB)"
  "$VENV/bin/pip" install --quiet --upgrade pip
else
  # ARM64, phones included, get a CPU-only wheel from PyPI anyway.
  echo "installing (PyTorch for $ARCH, a few hundred MB)"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi
"$VENV/bin/pip" install -e .

echo
"$VENV/bin/python" - <<'PYEOF'
import importlib
missing = [m for m in ("torch", "numpy", "fastapi", "uvicorn")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("error: missing after install: " + ", ".join(missing))
import torch
from motherbrain import __version__
print(f"MotherBrain {__version__} installed · torch {torch.__version__}")
PYEOF

# The window needs Tkinter, which Debian and its derivatives split into its own
# package. Everything else works without it, so this is a note rather than a
# failure.
if ! "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo "note: no Tkinter, so 'mb gui' (the window) will not open."
  echo "      Debian, Ubuntu, Kali:  sudo apt install python3-tk"
  echo "      then re-run this script."
  echo
fi

echo
echo "done. start it with:"
echo "    $VENV/bin/mb gui        # a window"
echo "    $VENV/bin/mb console    # the terminal"
echo "    $VENV/bin/mb serve      # HTTP, then open http://127.0.0.1:8000"
echo
echo "or activate the environment first:"
echo "    . $VENV/bin/activate && mb gui"
