"""Turn encoded frame bytes into a labeled RGB tile ready for compositing.

Color vs depth is decided by the *decoded array shape*, not by the topic name,
so it is robust to naming differences across robot platforms.
"""

import cv2
import numpy as np


def decode_frame(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode frame bytes as an image")
    return img


def _depth_to_rgb(depth):
    d = depth.astype(np.float32)
    valid = d[d > 0]
    if valid.size:
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        span = max(hi - lo, 1.0)
        norm = np.clip((d - lo) / span * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.clip(d, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_rgb_tile(img):
    """Decoded image array -> 3-channel uint8 RGB tile."""
    if img.ndim == 2:
        return _depth_to_rgb(img)
    if img.shape[2] == 1:
        return _depth_to_rgb(img[:, :, 0])
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def resize_to_height(tile, height):
    h, w = tile.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(tile, (new_w, height))


def add_label(tile, text):
    out = tile.copy()
    w = out.shape[1]
    cv2.rectangle(out, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_tile(buf, topic, height):
    """bytes -> resized, labeled RGB tile of the given height."""
    img = decode_frame(buf)
    rgb = to_rgb_tile(img)
    resized = resize_to_height(rgb, height)
    return add_label(resized, topic)
