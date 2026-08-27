#!/usr/bin/env python3
"""Generate text from a trained MotherBrain.

    ./generate.py "Once upon a time"

A thin wrapper around ``python -m motherbrain generate``.
"""

import sys

from motherbrain.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
