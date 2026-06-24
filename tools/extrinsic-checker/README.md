# extrinsic-checker

Verify a dataset's camera extrinsics by reconstructing each camera into the
robot base frame. Platform-agnostic package; all robot/dataset specifics live in
an external JSON config (`configs/<robot>.json`).

## Methods (auto-selected by camera modality)

- **depth** → deproject depth with the intrinsic, transform to base via
  `T_base_cam = T_base_link @ E` (FK ∘ extrinsic), fit the dominant plane, and
  check it is horizontal, at a sensible height, and in front of the robot.
  Artifacts: base-frame `<cam>_base.ply` + `<cam>_topdown.png`.
- **color** → project configured target links (e.g. the gripper) into the image
  and check they land in front of the camera and within the frame. Artifact:
  `<cam>_overlay.png`.

Each camera's **extrinsic** is taken from the config's `extrinsic` field if
present (so the checker validates the exact extrinsic downstream consumers use),
otherwise it falls back to the h5 (`parameters/camera/<cam>.json`). **Intrinsics**
are always read from the h5. The config names each camera's `mount_link`,
`modality`, optional `projection_targets` (color), and optional `extrinsic`.

## Usage

Run from the repo root with the uv environment:

```bash
uv run python tools/extrinsic-checker/extrinsic_check.py \
  --config tools/extrinsic-checker/configs/a2d.json \
  --input  /path/to/file.h5 \
  [--camera head hand_left hand_right] \
  [--frame 0] \
  [--out-dir DIR]
```

Exit code is 0 only if every checked camera passes.

## Config schema

See `configs/a2d.json`. Keys: `urdf`, `base_link`, `base_forward_axis`
(`+x/-x/+y/-y`), `joint_mapping` (per group: `h5_path` + `entries` of
`{h5_index, urdf_joint, sign}`), `cameras` (per camera: `mount_link`,
`modality`, optional `projection_targets`), `thresholds`
(`plane_vertical_min`, `table_height_range`).

## Tests

```bash
uv run --with pytest python -m pytest tools/extrinsic-checker/tests -v
```
