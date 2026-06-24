"""Load and validate the external checker config; resolve the URDF path."""

import json
import os

REQUIRED = {"urdf", "base_link", "base_forward_axis", "joint_mapping", "cameras", "thresholds"}
FORWARD_AXES = {"+x", "-x", "+y", "-y"}
MODALITIES = {"depth", "color"}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)

    missing = REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if cfg["base_forward_axis"] not in FORWARD_AXES:
        raise ValueError(
            f"bad base_forward_axis {cfg['base_forward_axis']!r}; must be one of {sorted(FORWARD_AXES)}"
        )
    for name, cam in cfg["cameras"].items():
        if cam.get("modality") not in MODALITIES:
            raise ValueError(
                f"camera {name!r}: bad/missing modality {cam.get('modality')!r}; "
                f"must be one of {sorted(MODALITIES)}"
            )

    # resolve urdf: relative to the config file's dir first, then cwd
    cfg_dir = os.path.dirname(os.path.abspath(path))
    raw = cfg["urdf"]
    candidates = [raw] if os.path.isabs(raw) else [os.path.join(cfg_dir, raw), os.path.abspath(raw)]
    for c in candidates:
        if os.path.exists(c):
            cfg["urdf_resolved"] = c
            break
    else:
        raise FileNotFoundError(f"urdf not found; tried {candidates}")
    return cfg
