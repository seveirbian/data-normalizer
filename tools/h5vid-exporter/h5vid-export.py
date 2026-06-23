#!/usr/bin/env python3
"""Standalone launcher: run h5vid-exporter from any directory without install.

    python3 tools/h5vid-exporter/h5vid-export.py --input FILE.h5 \
        --topics cameras/head/color cameras/head/depth --output out.mp4

Adds this directory to sys.path so the ``h5vid_exporter`` package imports
cleanly regardless of the current working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h5vid_exporter.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
