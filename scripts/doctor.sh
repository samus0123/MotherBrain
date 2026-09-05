#!/usr/bin/env sh
# Everything that could stop MotherBrain running, checked in one pass.
#
# Run this and paste the output. It is meant to answer "it doesn't work"
# without another round of guessing: each line is a fact, and the end names
# the most likely problem rather than leaving you to work it out.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

VENV=${VENV:-.venv}
PROBLEMS=""
note() { PROBLEMS="$PROBLEMS
  * $1"; }

echo "MotherBrain doctor"
echo "=================="
echo

echo "--- where ---"
echo "  directory   $ROOT"
echo "  user        $(id -un) (uid $(id -u))"
echo "  shell       ${SHELL:-unknown}"
echo "  os          $(uname -s) $(uname -m)"
[ -f /etc/os-release ] && echo "  distro      $(. /etc/os-release; echo "$PRETTY_NAME")"
echo

echo "--- this checkout ---"
if [ -d .git ]; then
  echo "  branch      $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "  commit      $(git log --oneline -1 2>/dev/null)"
  if git rev-parse origin/main >/dev/null 2>&1; then
    BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')
    echo "  behind main $BEHIND commit(s)"
    [ "$BEHIND" != "0" ] && [ "$BEHIND" != "?" ] && \
      note "this checkout is $BEHIND commit(s) behind origin/main - run: git pull"
  fi
  DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
  echo "  local edits $DIRTY file(s)"
else
  echo "  not a git checkout"
  note "this is not a git checkout, so 'git pull' cannot update it"
fi
echo

echo "--- python ---"
for p in python3 python3.13 python3.12 python3.11 python3.10; do
  W=$(command -v "$p" 2>/dev/null) && \
    echo "  $p -> $W ($($p -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>&1))"
done
echo

echo "--- virtual environment ---"
if [ -x "$VENV/bin/mb" ]; then
  echo "  ok          $VENV/bin/mb"
  echo "  its python  $("$VENV/bin/python" -c 'import sys;print("%d.%d at %s"%(sys.version_info[0],sys.version_info[1],sys.executable))' 2>&1)"
elif [ -d "$VENV" ]; then
  echo "  $VENV exists but has no mb"
  note "the install did not finish - run: sh scripts/install.sh"
else
  echo "  no $VENV"
  note "MotherBrain is not installed - run: sh scripts/install.sh"
fi
echo

echo "--- dependencies ---"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY=python3
for m in torch numpy fastapi uvicorn; do
  if "$PY" -c "import $m" >/dev/null 2>&1; then
    V=$("$PY" -c "import $m;print(getattr($m,'__version__',''))" 2>/dev/null)
    echo "  ok          $m $V"
  else
    echo "  MISSING     $m"
    note "$m is missing - the install did not finish; run: sh scripts/install.sh"
  fi
done
if "$PY" -c "import tkinter" >/dev/null 2>&1; then
  echo "  ok          tkinter"
else
  echo "  MISSING     tkinter (only the window needs it, not mb serve)"
  note "tkinter is missing, so 'mb gui' cannot open a window - either
    sudo apt install python3-tk, or sh scripts/tk-local.sh, or use mb serve"
fi
echo

echo "--- the model ---"
for f in models/motherbrain-base.pt runs/default/versions.json runs/default/tokenizer.json; do
  if [ -f "$f" ]; then
    echo "  ok          $f ($(du -h "$f" | cut -f1))"
  else
    echo "  MISSING     $f"
    note "$f is missing - the clone is incomplete; try a fresh git clone"
  fi
done
N=$(ls runs/default/patches/*.pt 2>/dev/null | wc -l)
echo "  patches     $N"
echo

echo "--- display ---"
echo "  DISPLAY         '${DISPLAY:-}'"
echo "  WAYLAND_DISPLAY '${WAYLAND_DISPLAY:-}'"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo "  no display - a window cannot open here"
  note "there is no display (SSH without -X, or a headless machine), so
    'mb gui' cannot work here at all - use mb serve and a browser"
fi
echo

echo "--- disk ---"
df -h . 2>/dev/null | tail -1 | sed 's/^/  /'
echo

echo "--- does mb run? ---"
if [ -x "$VENV/bin/mb" ]; then
  if "$VENV/bin/mb" --help >/dev/null 2>&1; then
    echo "  ok          mb --help"
    "$VENV/bin/mb" gui --help >/dev/null 2>&1 \
      && echo "  ok          mb gui exists" \
      || { echo "  MISSING     mb gui (this checkout predates it)"
           note "'mb gui' does not exist in this checkout - run: git pull && $VENV/bin/pip install -e ."; }
    echo "  status:"
    "$VENV/bin/mb" status 2>&1 | tail -6 | sed 's/^/    /'
  else
    echo "  mb --help FAILED:"
    "$VENV/bin/mb" --help 2>&1 | tail -6 | sed 's/^/    /'
    note "mb itself will not start - the error above is the real problem"
  fi
else
  echo "  (no mb to run)"
fi
echo

echo "=================="
if [ -n "$PROBLEMS" ]; then
  echo "most likely problem(s):$PROBLEMS"
else
  echo "nothing obviously wrong. If it still fails, paste the exact error."
fi
