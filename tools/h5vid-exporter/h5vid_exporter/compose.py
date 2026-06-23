"""Composite same-frame tiles into a single horizontal-row frame (1xN)."""

import numpy as np


def compose_row(tiles):
    if not tiles:
        raise ValueError("No tiles to compose")
    heights = {t.shape[0] for t in tiles}
    if len(heights) != 1:
        raise ValueError(f"Tiles have differing heights: {sorted(heights)}")
    return np.hstack(tiles)
