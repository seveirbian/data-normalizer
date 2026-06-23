"""Command-line entry point for h5vid-exporter."""

import argparse
import sys

from .exporter import export_video


def build_parser():
    p = argparse.ArgumentParser(
        prog="h5vid-export",
        description="Render h5 camera image-stream topics into a side-by-side mp4.",
    )
    p.add_argument("--input", required=True, help="Path to the input .h5 file")
    p.add_argument("--topics", required=True, nargs="+",
                   help="One or more full topic paths, e.g. cameras/head/color")
    p.add_argument("--output", required=True, help="Output .mp4 path")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second (default 30)")
    p.add_argument("--height", type=int, default=480,
                   help="Uniform per-tile height in pixels (default 480)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        export_video(args.input, args.topics, args.output,
                     fps=args.fps, height=args.height)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0
