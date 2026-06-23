import cv2
import h5py
import numpy as np
import pytest


def _encode_jpg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _encode_png16(depth):
    ok, buf = cv2.imencode(".png", depth)
    assert ok
    return buf.tobytes()


def _write_h5(path, topic_to_count):
    vlen = h5py.vlen_dtype(np.uint8)
    with h5py.File(path, "w") as f:
        for topic, (kind, n) in topic_to_count.items():
            ds = f.create_dataset(f"{topic}/data", (n,), dtype=vlen)
            for i in range(n):
                if kind == "color":
                    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
                    b = _encode_jpg(img)
                else:  # depth
                    depth = (np.arange(48 * 64).reshape(48, 64) % 1000).astype(np.uint16)
                    b = _encode_png16(depth)
                ds[i] = np.frombuffer(b, dtype=np.uint8)


@pytest.fixture
def tiny_h5(tmp_path):
    """h5 with one color + one depth topic, 3 aligned frames each."""
    path = tmp_path / "tiny.h5"
    _write_h5(path, {
        "cameras/head/color": ("color", 3),
        "cameras/head/depth": ("depth", 3),
    })
    return str(path)


@pytest.fixture
def mismatch_h5(tmp_path):
    """h5 where topics have differing frame counts (3 vs 2)."""
    path = tmp_path / "mismatch.h5"
    _write_h5(path, {
        "cameras/head/color": ("color", 3),
        "cameras/hand_left/color": ("color", 2),
    })
    return str(path)
