"""Depth-camera verification: point-cloud reprojection + dominant-plane verdict."""

import os

import cv2
import numpy as np
import open3d as o3d

from .h5read import read_intrinsic, decode_image
from .report import Verdict

_AXIS_COMP = {"x": 0, "y": 1}


def plane_verdict(points, thresholds, forward_axis):
    """points: Nx3 in base frame. Returns (passed, metrics)."""
    pts = np.asarray(points, float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    n = np.array(model[:3], float)
    n /= np.linalg.norm(n)
    inlier_pts = pts[inliers]
    height = float(np.median(inlier_pts[:, 2]))
    vertical_align = float(abs(n[2]))
    centroid = pts.mean(axis=0)

    comp = _AXIS_COMP[forward_axis[1]]
    sgn = 1.0 if forward_axis[0] == "+" else -1.0
    lo, hi = thresholds["table_height_range"]
    horiz_ok = vertical_align >= thresholds["plane_vertical_min"]
    height_ok = bool(lo <= height <= hi)
    front_ok = bool(centroid[comp] * sgn > 0)
    passed = bool(horiz_ok and height_ok and front_ok)

    metrics = {
        "plane_normal": [round(float(x), 3) for x in n],
        "vertical_align": round(vertical_align, 3),
        "plane_height": round(height, 3),
        "centroid": [round(float(x), 3) for x in centroid],
        "horiz_ok": bool(horiz_ok), "height_ok": height_ok, "front_ok": front_ok,
    }
    return passed, metrics


def _topdown(points, colors, out_path, forward_axis):
    """Orthographic top-down (view along -z) colored render; topmost point wins."""
    P = np.asarray(points)
    C = (np.asarray(colors) * 255).astype(np.uint8) if len(colors) else None
    if len(P) == 0:
        cv2.imwrite(out_path, np.zeros((400, 400, 3), np.uint8))
        return
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    res = 400
    img = np.zeros((res, res, 3), np.uint8)
    iy = ((P[:, 0] - xmin) / max(xmax - xmin, 1e-6) * (res - 1)).astype(int)
    ix = ((ymax - P[:, 1]) / max(ymax - ymin, 1e-6) * (res - 1)).astype(int)
    order = np.argsort(P[:, 2])   # paint low z first; high z overwrites
    for k in order:
        col = C[k][::-1] if C is not None else (200, 200, 200)
        img[iy[k], ix[k]] = col
    cv2.imwrite(out_path, img)


def run_depth_check(h5, cam, mount_link, E, kin, cfg, thresholds, forward_axis, frame, out_dir):
    I = read_intrinsic(h5, cam)
    depth = decode_image(h5, cam, "depth", frame)
    color = decode_image(h5, cam, "color", frame)
    H, W = depth.shape
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, I["fx"], I["fy"], I["ppx"], I["ppy"])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB).copy()),
        o3d.geometry.Image(depth.copy()),
        depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    T_base_cam = kin.link_transform(cfg, mount_link) @ E
    pcd.transform(T_base_cam)

    passed, metrics = plane_verdict(np.asarray(pcd.points), thresholds, forward_axis)
    os.makedirs(out_dir, exist_ok=True)
    ply = os.path.join(out_dir, f"{cam}_base.ply")
    o3d.io.write_point_cloud(ply, pcd)
    png = os.path.join(out_dir, f"{cam}_topdown.png")
    _topdown(np.asarray(pcd.points), np.asarray(pcd.colors), png, forward_axis)
    return Verdict(cam, "depth", passed, metrics, [ply, png])
