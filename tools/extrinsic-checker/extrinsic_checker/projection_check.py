"""Color-camera verification: project known link points into the image."""

import os

import cv2
import numpy as np

from .h5read import read_intrinsic, decode_image
from .report import Verdict


def project_point(p_base, T_base_link, E, fx, fy, cx, cy, W, H):
    """Project a base-frame point into the camera image.

    p_cam = E^-1 @ T_base_link^-1 @ p_base. Returns dict(u, v, Z, in_front, in_image).
    """
    p = np.array([p_base[0], p_base[1], p_base[2], 1.0])
    p_cam = np.linalg.inv(E) @ (np.linalg.inv(T_base_link) @ p)
    X, Y, Z = p_cam[:3]
    in_front = bool(Z > 0)
    if in_front:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        in_image = bool(0 <= u < W and 0 <= v < H)
    else:
        u = v = float("nan")
        in_image = False
    return {"u": float(u), "v": float(v), "Z": float(Z),
            "in_front": in_front, "in_image": in_image}


def run_projection_check(h5, cam, mount_link, E, targets, kin, cfg, frame, out_dir):
    I = read_intrinsic(h5, cam)
    color = decode_image(h5, cam, "color", frame)
    H, W = color.shape[:2]
    T_base_link = kin.link_transform(cfg, mount_link)

    overlay = color.copy()
    details = {}
    all_ok = bool(targets)
    for link in targets:
        p_base = kin.link_transform(cfg, link)[:3, 3]
        r = project_point(p_base, T_base_link, E, I["fx"], I["fy"], I["ppx"], I["ppy"], W, H)
        details[link] = r
        all_ok = all_ok and r["in_image"]
        if r["in_front"] and not np.isnan(r["u"]):
            uv = (int(r["u"]), int(r["v"]))
            cv2.drawMarker(overlay, uv, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(overlay, link, (uv[0] + 5, uv[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{cam}_overlay.png")
    cv2.imwrite(png, overlay)
    return Verdict(cam, "projection", bool(all_ok), {"targets": details}, [png])
