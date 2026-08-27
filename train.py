#!/usr/bin/env python3
"""Train MotherBrain on your own text.

    ./train.py --data path/to/your/text --epochs 3

A thin wrapper around ``python -m motherbrain train``.
"""

import sys

from motherbrain.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
