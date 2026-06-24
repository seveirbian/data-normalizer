"""Small helpers for reading camera params, frames, and joint arrays from the h5."""

import json

import cv2
import h5py
import numpy as np


def open_h5(path):
    return h5py.File(path, "r")


def _decode_str(x):
    return x.decode() if isinstance(x, bytes) else x


def _params(h5, cam):
    return json.loads(_decode_str(h5[f"parameters/camera/{cam}.json"][()]))


def read_intrinsic(h5, cam):
    return _params(h5, cam)["intrinsic"]


def extrinsic_matrix(extr):
    """Build a 4x4 E from a {rotation_matrix, translation_vector} dict."""
    E = np.eye(4)
    E[:3, :3] = np.array(extr["rotation_matrix"], float)
    E[:3, 3] = np.array(extr["translation_vector"], float)
    return E


def read_extrinsic(h5, cam):
    return extrinsic_matrix(_params(h5, cam)["extrinsic"])


def resolve_extrinsic(cam_cfg, h5, cam):
    """Prefer the camera's `extrinsic` from the config; fall back to the h5."""
    if "extrinsic" in cam_cfg:
        return extrinsic_matrix(cam_cfg["extrinsic"])
    return read_extrinsic(h5, cam)


def decode_image(h5, cam, modality, frame):
    raw = h5[f"cameras/{cam}/{modality}/data"][frame]
    raw = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
    flag = cv2.IMREAD_UNCHANGED if modality == "depth" else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), flag)
    if img is None:
        raise ValueError(f"failed to decode {cam}/{modality} frame {frame}")
    return img


def has_modality(h5, cam, modality):
    return f"cameras/{cam}/{modality}/data" in h5


def read_frame_values(h5, joint_mapping, frame):
    return {g: np.asarray(h5[spec["h5_path"]][frame]) for g, spec in joint_mapping.items()}
