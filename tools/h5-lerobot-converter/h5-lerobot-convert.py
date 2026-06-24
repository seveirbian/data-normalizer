#!/usr/bin/env python3
"""Standalone launcher: run h5-lerobot-converter from any directory without install.

    python3 tools/h5-lerobot-converter/h5-lerobot-convert.py <mode> \
        --input_dir DIR --output_dir DIR --task "..."

Modes:
    a2d      A2D  HDF5 -> LeRobot 2.x format   (h5_lerobot_converter.convert)
    a2d-v3   A2D  HDF5 -> LeRobot v3.0 format  (h5_lerobot_converter.convert_v3)
    r1-v3    R1   HDF5 -> LeRobot v3.0 format  (h5_lerobot_converter.r1_convert_v3)

All args after <mode> are passed through to the selected converter unchanged.
Adds this directory to sys.path so the ``h5_lerobot_converter`` package imports
cleanly regardless of the current working directory.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODES = {
    "a2d": "h5_lerobot_converter.convert",
    "a2d-v3": "h5_lerobot_converter.convert_v3",
    "r1-v3": "h5_lerobot_converter.r1_convert_v3",
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] not in MODES:
        prog = os.path.basename(sys.argv[0])
        print(f"usage: {prog} {{{','.join(MODES)}}} [converter args...]", file=sys.stderr)
        print("\nmodes:", file=sys.stderr)
        for mode, module in MODES.items():
            print(f"  {mode:<8} {module}", file=sys.stderr)
        return 2

    mode = argv[0]
    # Hand the remaining args to the converter's own argparse via sys.argv.
    sys.argv = [f"{sys.argv[0]} {mode}", *argv[1:]]
    converter = importlib.import_module(MODES[mode])
    return converter.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
