import numpy as np

from extrinsic_checker.projection_check import project_point

I4 = np.eye(4)
FX = FY = 600.0
CX, CY, W, H = 320.0, 240.0, 640, 480


def test_point_in_front_center():
    r = project_point([0.0, 0.0, 2.0], I4, I4, FX, FY, CX, CY, W, H)
    assert r["in_front"] and r["in_image"]
    assert abs(r["u"] - CX) < 1e-6 and abs(r["v"] - CY) < 1e-6


def test_point_behind_fails():
    r = project_point([0.0, 0.0, -1.0], I4, I4, FX, FY, CX, CY, W, H)
    assert r["in_front"] is False and r["in_image"] is False


def test_point_out_of_image_fails():
    r = project_point([5.0, 0.0, 1.0], I4, I4, FX, FY, CX, CY, W, H)
    assert r["in_front"] is True and r["in_image"] is False
