#!/bin/sh
# Double-click to start MotherBrain on macOS. All the work is in START;
# this exists because Finder will only launch a .command file, and because
# a double-clicked script starts in the user's home directory rather than
# next to itself.
cd "$(dirname "$0")" || exit 1
exec ./START "$@"
