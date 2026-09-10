"""MotherBrain, run as a program.

    python -m motherbrain              the menu, in this terminal
    python -m motherbrain gui          the window
    python -m motherbrain bbs          the bulletin board, on telnet
    python -m motherbrain --help       every command

`mb` is the same thing with a shorter name, installed by pip. This module
exists so that a checkout runs without installing anything at all - which
is what you want on a machine you have just copied it to, and what a
`.venv\\Scripts\\python.exe -m motherbrain` line in a Windows shortcut needs.

With no arguments it opens the console rather than printing usage: the
program's job is to start MotherBrain, and a program that answers "run me"
with a list of flags has not.
"""

from __future__ import annotations

import sys

from motherbrain.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["console"]
    sys.exit(main(argv))
