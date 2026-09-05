#!/usr/bin/env sh
# Install Tkinter into your home directory, without root.
#
# The window needs Tkinter, and on Debian, Ubuntu and Kali that lives in a
# separate package which normally wants `sudo apt install python3-tk`. On a
# machine where you are not root, that is the end of the road - except that
# downloading a .deb and unpacking it needs no privileges at all. This does
# exactly what apt would have done, into ~/.local/share/motherbrain-tk
# instead of /usr, and leaves the system untouched.
#
# The Python version has to match: a _tkinter built for 3.12 will not load
# into 3.11. That is checked before anything is downloaded.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

VENV=${VENV:-.venv}
PY="$VENV/bin/python"
[ -x "$PY" ] || PY=$(command -v python3) || {
  echo "no python3 found."; exit 1; }

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
DEST=${MB_TK_DIR:-$HOME/.local/share/motherbrain-tk}

if "$PY" -c "import tkinter" >/dev/null 2>&1; then
  echo "Tkinter already works for python $VER. Nothing to do."
  exit 0
fi

command -v apt-get >/dev/null 2>&1 || {
  echo "This script is for Debian and its relatives (it uses apt-get download)."
  echo "Elsewhere, the browser console needs nothing installed:"
  echo "    $VENV/bin/mb serve"
  exit 1
}
command -v dpkg-deb >/dev/null 2>&1 || { echo "dpkg-deb not found."; exit 1; }

echo "installing Tkinter for python $VER into $DEST"
echo "(no root; nothing outside your home directory is touched)"
echo

mkdir -p "$DEST"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# python3-tk carries the module; the tcl/tk shared libraries it links against
# may also be absent on a machine that never had it, so they come too.
for pkg in "python$VER-tk" python3-tk libtcl8.6 libtk8.6; do
  if apt-get download "$pkg" >/dev/null 2>&1; then
    echo "  fetched $pkg"
  else
    echo "  (no $pkg available; skipping)"
  fi
done

ls ./*.deb >/dev/null 2>&1 || {
  echo
  echo "Could not download any of them. Your apt sources may be unreachable."
  echo "The browser console needs nothing installed:"
  echo "    $VENV/bin/mb serve      then open http://127.0.0.1:8000"
  exit 1
}

for deb in ./*.deb; do
  dpkg-deb -x "$deb" "$DEST"
done
cd "$ROOT"

LIBDIR="$DEST/usr/lib/python$VER"
[ -d "$LIBDIR/tkinter" ] || {
  echo
  echo "The packages unpacked, but there is no tkinter for python $VER in them."
  echo "Debian only ships one Python version's tk module; yours is $VER, and"
  echo "the archive had a different one. Use the browser console instead:"
  echo "    $VENV/bin/mb serve"
  exit 1
}

# The tcl and tk shared libraries land in a multiarch directory whose name
# depends on the machine - x86_64-linux-gnu on a PC, aarch64-linux-gnu on a
# Pi or an ARM laptop. Hardcoding one of them breaks the other, so it is
# looked up rather than assumed.
ARCHLIB=""
for candidate in "$DEST"/usr/lib/*-linux-gnu*; do
  [ -d "$candidate" ] && ARCHLIB="$candidate" && break
done

ENVFILE="$DEST/env.sh"
{
  echo "# Written by scripts/tk-local.sh. Source this, or let gui.sh find it."
  echo "export PYTHONPATH=\"$LIBDIR:$LIBDIR/lib-dynload\${PYTHONPATH:+:\$PYTHONPATH}\""
  [ -n "$ARCHLIB" ] && \
    echo "export LD_LIBRARY_PATH=\"$ARCHLIB\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
} > "$ENVFILE"

# shellcheck disable=SC1090
. "$ENVFILE"
if "$PY" -c "import tkinter" >/dev/null 2>&1; then
  echo
  echo "done - Tkinter works for python $VER without touching the system."
  echo "scripts/gui.sh will find it from now on:"
  echo "    sh scripts/gui.sh"
else
  echo
  echo "unpacked, but python $VER still cannot import tkinter."
  echo "Most likely the tcl/tk libraries are missing and not downloadable here."
  echo "The browser console needs none of this:"
  echo "    $VENV/bin/mb serve      then open http://127.0.0.1:8000"
  exit 1
fi
