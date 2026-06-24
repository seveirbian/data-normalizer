# extrinsic-checker — Design

**Date:** 2026-06-24
**Status:** Approved (design phase)

## Purpose

A tool that verifies camera extrinsics in a robot dataset are correct, by
reconstructing what each camera sees into the robot base frame using the
camera intrinsics/extrinsics plus forward kinematics from a URDF, and checking
the result is geometrically sensible. Lives at `tools/extrinsic-checker/`.

The package code is **platform-agnostic**; everything robot/dataset-specific
lives in an external, self-contained JSON config passed via `--config`. An a2d
config (`configs/a2d.json`) is provided, populated from the joint mapping and
sign conventions already validated in earlier work.

## Background (validated facts the a2d config encodes)

- Camera intrinsics and extrinsics come from the **h5 file**
  (`parameters/camera/<cam>.json`: `intrinsic`, `extrinsic`). The config does
  NOT carry these; it only names each camera's mount link and modality.
- Extrinsic usage (validated on the a2d head camera): `E = [[R|t],[0,1]]` from
  the JSON maps camera-optical-frame points into the mount-link frame, so
  `T_base_cam = T_base_link @ E`. Camera optical frame is +x right / +y down /
  +z forward (RealSense / open3d convention).
- a2d FK quirks: h5 `joint_body_pitch` must be negated to match URDF
  `idx02_body_joint2`; base_link forward axis is `-x`. The head camera looks
  down at a tabletop.
- a2d cameras: `head` has color+depth; `hand_left`/`hand_right` are color-only.

## Two validation methods (auto-selected by camera modality)

**Depth cameras → point-cloud reprojection.**
Deproject depth with the intrinsic into a camera-frame cloud, transform to base
via `T_base_cam`, fit the dominant plane (RANSAC), and judge:
1. dominant plane is near-horizontal: `|n_z| >= plane_vertical_min`;
2. plane height within `table_height_range`;
3. cloud centroid lies on the robot's forward side (per `base_forward_axis`).
PASS if all hold. Artifacts: base-frame colored PLY + top-down ortho PNG.

**Color-only cameras → known-point projection.**
For each configured target link (e.g. the same arm's gripper), take its origin
in base from FK, transform into the camera frame (`p_cam = E^-1 @ T_base_link^-1
@ p_base`), and project with the intrinsic (`u = fx*X/Z + cx`, `v = fy*Y/Z + cy`).
Judge:
1. point is in front of the camera: `Z > 0`;
2. pixel falls within the image: `0 <= u < W and 0 <= v < H`.
PASS if all target links hold. Artifact: RGB overlay PNG with projected markers.

## Config schema (self-contained JSON)

```jsonc
{
  "urdf": "example-dataset/guodi/a2d/g1/g1_flat.urdf",  // path (resolved relative to config file dir, then cwd)
  "base_link": "base_link",
  "base_forward_axis": "-x",                            // one of +x,-x,+y,-y
  "joint_mapping": {
    "<group>": {
      "h5_path": "joints/state/waist/position",
      "entries": [
        {"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1},
        {"h5_index": 1, "urdf_joint": "idx01_body_joint1", "sign": 1}
      ]
    }
    // ... head, arm, etc. (state/robot floating base is NOT used — checks are in base_link frame)
  },
  "cameras": {
    "head":       {"mount_link": "head_link2",     "modality": "depth"},
    "hand_left":  {"mount_link": "arm_l_end_link", "modality": "color",
                   "projection_targets": ["gripper_left_center_link"]},
    "hand_right": {"mount_link": "arm_r_end_link", "modality": "color",
                   "projection_targets": ["gripper_right_center_link"]}
  },
  "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]}
}
```

`sign` defaults to 1 if omitted. `depth_scale` for the depth image is fixed at
1000.0 (mm → m), standard for these RealSense streams.

## Architecture

```
tools/extrinsic-checker/
  extrinsic_checker/
    loader.py            # load + validate config; resolve urdf path
    kinematics.py        # yourdfpy wrapper: read h5 frame joints, build cfg (mapping+sign), T_base_link(link)
    depth_check.py       # deproject + transform + RANSAC plane verdict; emit PLY + top-down PNG
    projection_check.py  # project target-link points into the color image; emit overlay PNG
    report.py            # Verdict dataclass + aggregation + printing
    cli.py               # argparse: --config --input --camera --frame --out-dir; orchestrate; exit code
    __main__.py
  extrinsic_check.py      # standalone launcher (runs from any cwd, like h5vid-export.py)
  configs/
    a2d.json
  tests/
  README.md
  pyproject.toml          # deps: open3d, yourdfpy, h5py, opencv-python-headless, numpy
```

Run with the uv environment (Python 3.12), which has these deps installed.

### Component responsibilities

- **loader.py** — parse the JSON config, validate required keys and enum values
  (`base_forward_axis`, `modality`), resolve `urdf` path (relative to the config
  file directory, then cwd), raise clear errors on missing/invalid fields.
- **kinematics.py** — load the URDF once via yourdfpy. `build_cfg(h5, frame,
  joint_mapping)` reads each group's array at `frame`, applies `sign`, and
  returns a `{urdf_joint: value}` dict (joints not mapped default to 0).
  `link_transform(cfg, link)` returns `T_base_link` (`get_transform(link,
  base_link)`).
- **depth_check.py** — read the camera intrinsic + depth frame from the h5,
  deproject with open3d, transform by `T_base_cam = T_base_link @ E`, RANSAC
  `segment_plane`, compute the three criteria, return a Verdict; write the
  base-frame colored PLY and a top-down orthographic PNG.
- **projection_check.py** — read intrinsic + color frame; for each target link
  compute `p_base` from FK, transform to camera frame, project; return a Verdict
  with per-target (u,v,Z,in_image) detail; write the RGB overlay PNG.
- **report.py** — `Verdict` dataclass (camera, method, passed, metrics dict,
  artifacts list); aggregate and pretty-print; provide overall pass/fail.
- **cli.py** — parse args; load config; open h5; for each requested camera pick
  the method by modality and run it; print the report; exit non-zero if any
  camera FAILs or errors.

## CLI

```
python3 tools/extrinsic-checker/extrinsic_check.py \
    --config configs/a2d.json \
    --input  FILE.h5 \
    [--camera head hand_left hand_right]   # default: all cameras in the config
    [--frame 0]                            # default 0
    [--out-dir DIR]                        # default: ./extrinsic_check_out
```

Per camera: print the method, metrics, and PASS/FAIL. Exit 0 only if every
checked camera passes.

## Error handling

- Config missing/invalid field, or unknown `base_forward_axis`/`modality` →
  error naming the field.
- Requested camera not in config, or its modality data absent in the h5 →
  error listing available cameras / present modalities.
- URDF file or a referenced link/joint not found → error naming it.
- Joint array width at the frame shorter than a mapping `h5_index` → error.

## Testing (TDD)

Pure-function unit tests (synthetic inputs, no large files):
- **loader**: valid config parses; missing key / bad enum raises.
- **kinematics.build_cfg**: array + mapping + signs → expected cfg dict (incl.
  sign negation and unmapped-joint defaulting to 0).
- **depth_check verdict** (`plane_verdict(points, thresholds, forward_axis)`):
  a synthetic horizontal plane PASSes; a vertical plane FAILs; a plane at wrong
  height FAILs; centroid on the wrong side FAILs.
- **projection_check** (`project_point(p_base, T_base_link, E, fx,fy,cx,cy,W,H)`):
  with identity transforms a known point projects to the expected pixel; a point
  behind the camera (Z<0) FAILs; an out-of-image point FAILs.

Integration / smoke (use real files, run via uv):
- **FK**: load `g1_flat.urdf`, apply the a2d head config at frame 0, assert
  `head_link2` origin ≈ `[0.495, 0, 1.385]` (tolerance 1 cm).
- **a2d head smoke**: run the depth check on the real a2d h5 head camera, assert
  PASS and that the PLY + top-down PNG are written.
