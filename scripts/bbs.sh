#!/usr/bin/env sh
# Run MotherBrain as a bulletin board.
#
#     sh scripts/bbs.sh              loopback, port 2323, no privilege needed
#     sh scripts/bbs.sh 23           telnet's own port; needs root or setcap
#
# Port 23 is privileged on every Unix. This defaults to 2323 and tells you
# the three ways to get 23 rather than failing with EACCES and no advice.
set -eu

cd "$(dirname "$0")/.."
PORT="${1:-2323}"
BIND="${MB_BBS_HOST:-127.0.0.1}"

if [ -x .venv/bin/mb ]; then
    RUN=".venv/bin/mb"
elif [ -x .venv/bin/python ]; then
    RUN=".venv/bin/python -m motherbrain.cli"
else
    echo "MotherBrain is not installed yet. Run:  sh scripts/install.sh" >&2
    exit 1
fi

if [ "$PORT" -lt 1024 ] && [ "$(id -u)" -ne 0 ]; then
    cat >&2 <<MSG
Port $PORT is privileged: only root may bind below 1024.

Three ways, best first:

  sh scripts/bbs.sh 2323
      no privilege needed. Callers use:  telnet $BIND 2323

  sudo setcap 'cap_net_bind_service=+ep' "\$(readlink -f "\$(command -v python3)")"
      lets this Python bind low ports without running as root

  sudo $RUN bbs --port $PORT
      runs the whole board as root. Least good of the three.
MSG
    exit 1
fi

command -v telnet >/dev/null 2>&1 || cat >&2 <<'MSG'
Note: no telnet client found here. To call the board from this machine:
  Debian/Ubuntu/Kali   sudo apt install telnet
  Fedora               sudo dnf install telnet
  For the real thing, with proper ANSI:  syncterm

MSG

echo "Starting the board. Call it with:  telnet $BIND $PORT"
echo
exec $RUN bbs --host "$BIND" --port "$PORT"
