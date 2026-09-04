#!/usr/bin/env sh
# Install MotherBrain, working around the things that usually break.
#
# A plain `pip install -e .` fails on several common setups for reasons that
# have nothing to do with this project:
#
#   * Debian 12+, Ubuntu 23.04+, Kali and Termux's proot images mark the system
#     Python as externally managed (PEP 668), so pip refuses to touch it.
#   * On x86_64 Linux, PyPI's `torch` wheel depends on the whole CUDA runtime.
#     That is about 5.5GB installed and completely useless without an NVIDIA
#     GPU. It is the most common reason this fails: the disk fills up, or the
#     download does not finish.
#   * PyTorch only publishes wheels for some Python versions. On a very new
#     Python there is no wheel and pip says the requirement cannot be
#     satisfied, which reads as if the package does not exist.
#
# So: a virtual environment, the CPU wheel unless there is a GPU to use, and a
# disk check before anything is downloaded rather than after.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

fail() {
  echo
  echo "install failed: $1"
  [ -n "${2:-}" ] && echo "$2"
  exit 1
}

# ---- python ---------------------------------------------------------------

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || fail "$PY not found." \
  "Install Python 3.10 or newer first:  sudo apt install python3"

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
MAJOR=$(echo "$VER" | cut -d. -f1)
MINOR=$(echo "$VER" | cut -d. -f2)
echo "python $VER at $(command -v "$PY")"

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
  fail "MotherBrain needs Python 3.10 or newer; this is $VER."
fi
if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -gt 13 ]; then
  echo
  echo "warning: PyTorch may not publish wheels for Python $VER yet."
  echo "If the torch install fails, install an older Python and re-run with:"
  echo "    PYTHON=python3.12 sh scripts/install.sh"
  echo
fi

# ---- virtual environment --------------------------------------------------

VENV=${VENV:-.venv}
if [ ! -d "$VENV" ]; then
  echo "creating virtual environment in $VENV"
  "$PY" -m venv "$VENV" || fail "could not create a virtual environment." \
    "On Debian, Ubuntu and Kali:  sudo apt install python3-venv"
fi
PIP="$VENV/bin/pip"
[ -x "$PIP" ] || fail "$PIP is missing; the virtual environment is incomplete." \
  "Delete $VENV and re-run, or:  sudo apt install python3-venv"

"$PIP" install --quiet --upgrade pip || true

# One retry: a torch wheel is large and a dropped connection is not a reason
# to make somebody start over.
retry_pip() {
  "$PIP" install "$@" && return 0
  echo "  that did not finish; retrying once ..."
  "$PIP" install "$@"
}

# ---- torch: which build, and is there room for it -------------------------

if "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  # Already there, so none of the sizing below matters. This is also what
  # makes the "install torch yourself and re-run" advice further down true.
  echo "torch is already installed in $VENV; leaving it alone"
else
  ARCH=$(uname -m 2>/dev/null || echo unknown)
  FLAVOUR=cpu
  NEED_MB=1600

  WHY=""
  if [ "${MB_CUDA:-}" = "1" ]; then
    FLAVOUR=cuda
    WHY="you asked for it"
  elif [ "${MB_CPU_ONLY:-}" = "1" ]; then
    FLAVOUR=cpu
  elif [ "$ARCH" = "x86_64" ] && command -v nvidia-smi >/dev/null 2>&1 \
       && nvidia-smi >/dev/null 2>&1; then
    # A working GPU is present, so the CUDA build earns its size.
    FLAVOUR=cuda
    WHY="an NVIDIA GPU was found"
  elif [ "$ARCH" != "x86_64" ]; then
    # ARM64, phones included, get a CPU-only wheel from PyPI anyway.
    FLAVOUR=pypi
  fi
  [ "$FLAVOUR" = "cuda" ] && NEED_MB=6500

  FREE_MB=$(df -Pk . 2>/dev/null | awk 'NR==2 {print int($4/1024)}')
  if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt "$NEED_MB" ]; then
    fail "not enough disk: ${FREE_MB}MB free, about ${NEED_MB}MB needed." \
"$(if [ "$FLAVOUR" = cuda ]; then
     echo 'The CUDA build of PyTorch is ~5.5GB. Without an NVIDIA GPU you do'
     echo 'not need it - re-run as:  MB_CPU_ONLY=1 sh scripts/install.sh'
   else
     echo 'Free some space and try again.'
   fi)"
  fi

  case "$FLAVOUR" in
    cuda) echo "installing PyTorch with CUDA (~5.5GB; $WHY)" ;;
    cpu)  echo "installing PyTorch, CPU build (~1.5GB)"
          echo "  have an NVIDIA GPU? re-run as:  MB_CUDA=1 sh scripts/install.sh" ;;
    pypi) echo "installing PyTorch for $ARCH (a few hundred MB)" ;;
  esac

  if [ "$FLAVOUR" = "cpu" ]; then
    # The CPU wheels live on PyTorch's own index. Some networks and corporate
    # proxies block it, and that failure is worth naming precisely: falling
    # back to PyPI would quietly install the 5.5GB CUDA build instead.
    if ! retry_pip torch --index-url https://download.pytorch.org/whl/cpu; then
      fail "could not reach download.pytorch.org (the CPU-only wheel index)." \
"Your network or proxy is blocking it. Two ways on:

  * install the PyPI build instead, which pulls the CUDA runtime with it
    (~5.5GB, and it still runs on the CPU):
        MB_CUDA=1 sh scripts/install.sh

  * or install torch yourself from wherever you can reach it, then re-run
    this script - it will see torch is present and skip that step."
    fi
  fi
fi

# ---- MotherBrain itself ---------------------------------------------------

retry_pip -e . || fail "could not install MotherBrain itself." \
  "The error above is from pip. If it mentions torch, see the notes at the top of this script."

# ---- verify ---------------------------------------------------------------

echo
"$VENV/bin/python" - <<'PYEOF'
import importlib.util          # `import importlib` alone does not expose .util
missing = [m for m in ("torch", "numpy", "fastapi", "uvicorn")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("error: missing after install: " + ", ".join(missing))
import torch
from motherbrain import __version__
build = "CUDA" if torch.version.cuda else "CPU"
print(f"MotherBrain {__version__} installed - torch {torch.__version__} ({build} build)")
PYEOF

# The window needs Tkinter, which Debian and its derivatives split into its own
# package. Everything else works without it, so this is a note, not a failure.
if ! "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo
  echo "note: no Tkinter, so 'mb gui' (the window) will not open."
  echo "      Debian, Ubuntu, Kali:  sudo apt install python3-tk"
  echo "      Everything else works without it."
fi

echo
echo "done. start it with:"
echo "    $VENV/bin/mb gui        # a window"
echo "    $VENV/bin/mb console    # the terminal"
echo "    $VENV/bin/mb serve      # HTTP, then open http://127.0.0.1:8000"
echo
echo "or activate the environment first:"
echo "    . $VENV/bin/activate && mb gui"
