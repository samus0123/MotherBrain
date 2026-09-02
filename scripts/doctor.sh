#!/usr/bin/env sh
# Report everything needed to diagnose a MotherBrain that will not start.
# Run it from the repository directory and paste the whole output.

echo "--- where ---"
pwd
echo
echo "--- branch (must NOT be main: main holds only a README) ---"
git rev-parse --abbrev-ref HEAD 2>&1 || echo "not a git repository"
git log --oneline -1 2>&1
echo
echo "--- files that must exist ---"
for f in motherbrain/cli.py models/motherbrain.pt scripts/install.sh pyproject.toml; do
  if [ -e "$f" ]; then echo "  ok      $f"; else echo "  MISSING $f"; fi
done
echo
echo "--- python ---"
for p in python3 python; do
  if command -v "$p" >/dev/null 2>&1; then
    echo "  $p -> $(command -v $p) $($p -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>&1)"
  fi
done
echo
echo "--- virtual environment ---"
if [ -x .venv/bin/mb ]; then
  echo "  ok      .venv/bin/mb"
elif [ -d .venv ]; then
  echo "  .venv exists but has no mb — install did not finish"
else
  echo "  no .venv — run scripts/install.sh"
fi
echo
echo "--- dependencies ---"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
for m in torch numpy fastapi; do
  if "$PY" -c "import $m" 2>/dev/null; then
    echo "  ok      $m"
  else
    echo "  MISSING $m"
  fi
done
echo
echo "--- can it start? ---"
if [ -x .venv/bin/mb ]; then
  echo "" | .venv/bin/mb status 2>&1 | head -5
else
  echo "" | "$PY" -m motherbrain.cli status 2>&1 | head -5
fi
