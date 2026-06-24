import numpy as np

from extrinsic_checker.depth_check import plane_verdict

TH = {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]}


def _horizontal_plane(z, x_center):
    xs = np.random.uniform(-0.3, 0.3, 4000) + x_center
    ys = np.random.uniform(-0.3, 0.3, 4000)
    zs = np.full(4000, z)
    return np.stack([xs, ys, zs], 1)


def test_horizontal_plane_in_front_passes():
    pts = _horizontal_plane(z=0.84, x_center=-0.6)   # base_forward -x
    passed, m = plane_verdict(pts, TH, "-x")
    assert passed is True
    assert m["vertical_align"] > 0.9 and m["height_ok"] and m["front_ok"]


def test_vertical_plane_fails():
    # plane spanning y-z at fixed x -> normal ~ +x -> not horizontal
    ys = np.random.uniform(-0.3, 0.3, 4000)
    zs = np.random.uniform(0.4, 1.0, 4000)
    xs = np.full(4000, -0.6)
    passed, m = plane_verdict(np.stack([xs, ys, zs], 1), TH, "-x")
    assert passed is False
    assert m["vertical_align"] < 0.5


def test_wrong_height_fails():
    pts = _horizontal_plane(z=2.0, x_center=-0.6)
    passed, m = plane_verdict(pts, TH, "-x")
    assert passed is False and m["height_ok"] is False


def test_wrong_side_fails():
    pts = _horizontal_plane(z=0.84, x_center=+0.6)   # in +x, but forward is -x
    passed, m = plane_verdict(pts, TH, "-x")
    assert passed is False and m["front_ok"] is False
