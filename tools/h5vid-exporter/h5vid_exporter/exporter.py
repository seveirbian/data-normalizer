"""Orchestrate reading, rendering, composing and encoding into an mp4.

Encoding uses cv2.VideoWriter with the mp4v codec (see the design spec for why
this is preferred over imageio/ffmpeg in this environment).
"""

import os
import warnings

import cv2
import numpy as np

from .compose import compose_row
from .reader import H5Reader
from .render import render_tile


def _pad_even(frame):
    h, w = frame.shape[:2]
    ph, pw = h % 2, w % 2
    if ph or pw:
        frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), constant_values=0)
    return frame


def export_video(input_path, topics, output_path, fps=30, height=480):
    if not topics:
        raise ValueError("At least one topic is required")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with H5Reader(input_path) as reader:
        for topic in topics:
            reader.validate_topic(topic)

        counts = {t: reader.frame_count(t) for t in topics}
        n = min(counts.values())
        if len(set(counts.values())) > 1:
            warnings.warn(
                f"Topics have differing frame counts {counts}; using minimum {n}"
            )

        writer = None
        try:
            for i in range(n):
                tiles = [render_tile(reader.frame_bytes(t, i), t, height)
                         for t in topics]
                frame = _pad_even(compose_row(tiles))  # RGB
                if writer is None:
                    fh, fw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (fw, fh))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer for {output_path!r}"
                        )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()

    return output_path
