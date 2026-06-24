#!/usr/bin/env python3
"""Standalone launcher: run extrinsic-checker from any directory without install.

    uv run python tools/extrinsic-checker/extrinsic_check.py \
        --config tools/extrinsic-checker/configs/a2d.json --input FILE.h5

Adds this directory to sys.path so the ``extrinsic_checker`` package imports
regardless of the current working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extrinsic_checker.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
