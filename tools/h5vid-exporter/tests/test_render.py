import cv2
import numpy as np
import pytest

from h5vid_exporter.render import (
    decode_frame,
    to_rgb_tile,
    add_label,
    resize_to_height,
    render_tile,
)


def _jpg_bytes():
    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _png16_bytes():
    depth = (np.arange(48 * 64).reshape(48, 64) % 1000).astype(np.uint16)
    ok, buf = cv2.imencode(".png", depth)
    assert ok
    return buf.tobytes()


def test_decode_frame_color():
    img = decode_frame(_jpg_bytes())
    assert img.shape == (48, 64, 3)


def test_decode_frame_bad_bytes_raises():
    with pytest.raises(ValueError):
        decode_frame(b"not an image")


def test_to_rgb_tile_color_keeps_size_3ch():
    img = decode_frame(_jpg_bytes())
    tile = to_rgb_tile(img)
    assert tile.shape == (48, 64, 3) and tile.dtype == np.uint8


def test_to_rgb_tile_depth_2d_becomes_3ch():
    depth = decode_frame(_png16_bytes())
    assert depth.ndim == 2  # 16-bit png decodes single-channel
    tile = to_rgb_tile(depth)
    assert tile.shape == (48, 64, 3) and tile.dtype == np.uint8


def test_resize_to_height():
    tile = np.zeros((48, 64, 3), np.uint8)
    out = resize_to_height(tile, 240)
    assert out.shape[0] == 240 and out.shape[2] == 3


def test_add_label_keeps_shape():
    tile = np.zeros((100, 120, 3), np.uint8)
    out = add_label(tile, "cameras/head/color")
    assert out.shape == (100, 120, 3)


def test_render_tile_end_to_end():
    out = render_tile(_jpg_bytes(), "cameras/head/color", 120)
    assert out.shape[0] == 120 and out.shape[2] == 3 and out.dtype == np.uint8
