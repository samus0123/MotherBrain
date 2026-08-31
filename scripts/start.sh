#!/usr/bin/env sh
# Start MotherBrain, doing whatever setup is still missing first.
#
# Safe to run repeatedly: each step checks whether it is already done. If
# something fails, it stops at that step and says which one, so the failure is
# visible instead of silent.

set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
echo "MotherBrain: $ROOT"

# 1. the code. main holds only a LICENSE and a README, so a plain clone lands
#    on a branch with nothing to run.
if [ ! -f motherbrain/cli.py ]; then
  echo
  echo "step 1: getting the code (this checkout is missing it)"
  git fetch origin claude/massive-parameter-llm-mcs613 \
    && git checkout claude/massive-parameter-llm-mcs613 || {
      echo
      echo "could not switch to the branch that holds the code."
      echo "start again with:"
      echo "  git clone -b claude/massive-parameter-llm-mcs613 \\"
      echo "      https://github.com/samus0123/MotherBrain"
      exit 1
    }
fi
echo "  ok: code present"

# 2. somewhere to install to.
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || {
  echo
  echo "step 2 failed: no $PY on this system."
  echo "  Debian/Ubuntu:  apt install python3 python3-venv"
  echo "  Termux:         this needs proot-distro, not Termux's own python"
  exit 1
}

if [ ! -x .venv/bin/mb ]; then
  echo
  echo "step 2: installing (downloads PyTorch, several minutes)"
  sh scripts/install.sh || {
    echo
    echo "step 2 failed: the install did not finish."
    echo "Run 'sh scripts/doctor.sh' and send the output."
    exit 1
  }
fi
echo "  ok: installed"

# 3. something to run. A clone ships models/motherbrain.pt, so this is only
#    missing if the checkout is incomplete.
if [ ! -f models/motherbrain.pt ] && [ ! -f runs/default/checkpoint.pt ]; then
  echo
  echo "step 3 failed: no model found."
  echo "models/motherbrain.pt is missing, so the checkout is incomplete."
  echo "Try:  git checkout claude/massive-parameter-llm-mcs613 -- models"
  exit 1
fi
echo "  ok: model present"

echo
exec .venv/bin/mb console "$@"
