#!/usr/bin/env sh
# Open MotherBrain's window, checking each thing that can stop it.
#
# "mb gui doesn't work" has four causes and they look nothing alike:
#   * the checkout predates the command, so there is no `mb gui`
#   * MotherBrain is not installed, so there is no `mb` at all
#   * Tkinter is missing - Debian and Kali ship it separately
#   * there is no display, because this is SSH or a headless box
#
# Each is checked here, in that order, and each says what to do about it.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

VENV=${VENV:-.venv}
MB="$VENV/bin/mb"
PY="$VENV/bin/python"

say() { printf '%s\n' "$1"; }
fail() { say ""; say "$1"; say ""; [ -n "${2:-}" ] && say "$2"; exit 1; }

# ---- 1. is MotherBrain installed at all? ----------------------------------

if [ ! -x "$MB" ]; then
  fail "MotherBrain is not installed in $VENV." \
"Install it first:
    sh scripts/install.sh"
fi

# ---- 2. does this copy have the gui command? ------------------------------

if ! "$MB" gui --help >/dev/null 2>&1; then
  say "this copy of MotherBrain has no 'gui' command - it predates the window."
  say "updating ..."
  git pull --ff-only || fail "could not update automatically." \
"Do it by hand:
    git pull
    $VENV/bin/pip install -e ."
  "$VENV/bin/pip" install -q -e . || fail "the update did not install."
  "$MB" gui --help >/dev/null 2>&1 || fail "still no 'gui' command after updating."
  say "updated."
fi

# ---- 3. Tkinter ------------------------------------------------------------

# A previous run of tk-local.sh may have put one in your home directory.
TKENV=${MB_TK_DIR:-$HOME/.local/share/motherbrain-tk}/env.sh
if [ -f "$TKENV" ] && ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  . "$TKENV"
  say "using the Tkinter in $(dirname "$TKENV")"
fi

if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  fail "the window needs Tkinter, which is not installed." \
"With root:
    sudo apt install python3-tk          Debian, Ubuntu, Kali
    sudo dnf install python3-tkinter     Fedora
    sudo pacman -S tk                    Arch

Without root - unpacks it into your home directory, touching nothing else:
    sh scripts/tk-local.sh

Or skip the window entirely. The browser console has the same four options
and needs nothing installed:
    $MB serve       then open http://127.0.0.1:8000"
fi

# ---- 4. a display to open it on --------------------------------------------

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  fail "there is no display to open a window on (\$DISPLAY is not set)." \
"You are on a headless machine, or SSH without X forwarding, or WSL
without an X server. Serve it to a browser instead:

    $MB serve
    then open http://127.0.0.1:8000

Over SSH, 'ssh -X you@host' forwards a display and this will then work."
fi

# ---- 5. is there a model? --------------------------------------------------

if ! "$MB" status 2>/dev/null | grep -q "READY"; then
  say "warning: no model is loaded yet. The window will open and say so."
  say "         '$MB status' explains what is missing."
  say ""
fi

say "starting MotherBrain ..."
exec "$MB" gui "$@"
