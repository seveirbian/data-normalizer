import numpy as np

from extrinsic_checker.h5read import (
    open_h5, read_intrinsic, read_extrinsic, decode_image,
    read_frame_values, has_modality,
)


def test_read_intrinsic(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        I = read_intrinsic(h5, "head")
    assert I["fx"] == 600.0 and I["ppx"] == 320.0


def test_read_extrinsic_identity(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        E = read_extrinsic(h5, "head")
    assert E.shape == (4, 4)
    assert np.allclose(E, np.eye(4))


def test_decode_depth_and_color(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        d = decode_image(h5, "head", "depth", 0)
        c = decode_image(h5, "head", "color", 0)
    assert d.shape == (48, 64) and d.dtype == np.uint16 and d[0, 0] == 700
    assert c.shape == (48, 64, 3)


def test_has_modality(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        assert has_modality(h5, "head", "depth") is True
        assert has_modality(h5, "head", "infrared") is False


def test_read_frame_values(tiny_a2d_h5):
    mapping = {"head": {"h5_path": "joints/state/head/position", "entries": []}}
    with open_h5(tiny_a2d_h5) as h5:
        fv = read_frame_values(h5, mapping, 1)
    assert np.allclose(fv["head"], [0.3, 0.4])
