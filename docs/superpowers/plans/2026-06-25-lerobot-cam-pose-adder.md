# lerobot-cam-pose-adder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the reusable kinematics/config/camera core from extrinsic-checker into a shared `tools/common/robotkit` package, then build `tools/lerobot-cam-pose-adder/` which adds a per-frame `observation.camera_poses` feature (21-dim, base_link, pos+quat) to a copy of a LeRobot v3 dataset.

**Architecture:** A shared `robotkit` package (config, kinematics, camera params, h5 io) imported by both extrinsic-checker (refactored) and the new tool via `sys.path` (each tool's conftest/launcher adds `tools/common`). The new tool reads joints from the LeRobot `observation.state` (located via the field-map config), runs yourdfpy FK + the config extrinsic, and writes poses into a `--output` dataset copy.

**Tech Stack:** Python 3.12 (uv env): yourdfpy, numpy, scipy (quaternions), pandas + pyarrow (parquet), opencv/h5py (robotkit). Run via `uv run`; tests via `uv run --with pytest`.

**Conventions:**
- All commands run from repo root `/root/codes/data-normalizer`.
- Test command form: `uv run --with pytest python -m pytest <path> -v`.
- Quaternion order is xyzw (scipy `Rotation.as_quat()` default), matching `base_ori_x..w`.

---

## File Structure

```
tools/common/
  conftest.py                      # empty (pytest rootdir for robotkit tests)
  robotkit/
    __init__.py
    config.py                      # load_config (moved from extrinsic_checker/loader.py)
    kinematics.py                  # Kinematics, build_cfg (moved)
    camera.py                      # extrinsic_matrix, resolve_extrinsic, read_intrinsic, read_extrinsic
    h5io.py                        # open_h5, decode_image, has_modality, read_frame_values
  tests/
    conftest.py                    # tiny_a2d_h5 fixture
    test_config.py  test_kinematics.py  test_camera.py  test_h5io.py

tools/extrinsic-checker/           # refactored to import robotkit; loader/kinematics/h5read removed
  conftest.py                      # adds ../common to sys.path
  extrinsic_check.py               # launcher adds ../common
  extrinsic_checker/{depth_check,projection_check,report,cli,__main__}.py
  tests/{test_report,test_depth_check,test_projection_check}.py

tools/lerobot-cam-pose-adder/
  conftest.py                      # adds ../common to sys.path
  lerobot_cam_pose_adder.py        # launcher
  lerobot_cam_pose_adder/
    __init__.py  fieldmap.py  poses.py  dataset.py  cli.py  __main__.py
  tests/{test_fieldmap,test_poses,test_dataset}.py
  README.md  pyproject.toml
```

---

## Task 1: Create the `robotkit` shared package

**Files:** create `tools/common/conftest.py`, `tools/common/robotkit/{__init__,config,kinematics,camera,h5io}.py`, `tools/common/tests/conftest.py`, `tools/common/tests/{test_config,test_kinematics,test_camera,test_h5io}.py`

- [ ] **Step 1: Create package files**

`tools/common/conftest.py`:
```python
# Presence of this file makes pytest add this directory to sys.path,
# so `import robotkit` works for robotkit's own tests.
```

`tools/common/robotkit/__init__.py`:
```python
"""robotkit — shared config loading, URDF FK, and camera-parameter helpers."""

__version__ = "0.1.0"
```

`tools/common/robotkit/config.py`:
```python
"""Load and validate a checker/tool config; resolve the URDF path."""

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
```

`tools/common/robotkit/kinematics.py`:
```python
"""Joint-config assembly (pure) and a thin yourdfpy FK wrapper."""

import yourdfpy


def build_cfg(frame_values, joint_mapping, actuated_joints):
    """Assemble a {urdf_joint: value} dict from group-keyed frame values."""
    cfg = {j: 0.0 for j in actuated_joints}
    for group, spec in joint_mapping.items():
        arr = frame_values[group]
        for e in spec["entries"]:
            idx = e["h5_index"]
            if idx >= len(arr):
                raise ValueError(
                    f"group {group!r}: h5_index {idx} out of range (array width {len(arr)})"
                )
            cfg[e["urdf_joint"]] = float(arr[idx]) * e.get("sign", 1)
    return cfg


class Kinematics:
    def __init__(self, urdf_path, base_link):
        self.robot = yourdfpy.URDF.load(
            urdf_path, load_meshes=False, build_collision_scene_graph=False
        )
        self.base_link = base_link

    def link_transform(self, cfg, link):
        if link not in self.robot.link_map:
            raise ValueError(f"link {link!r} not found in URDF")
        self.robot.update_cfg(cfg)
        return self.robot.get_transform(link, self.base_link)
```

`tools/common/robotkit/camera.py`:
```python
"""Camera intrinsic/extrinsic helpers (read from h5 or build from a config dict)."""

import json

import numpy as np


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
```

`tools/common/robotkit/h5io.py`:
```python
"""h5 image/array reading helpers."""

import cv2
import h5py
import numpy as np


def open_h5(path):
    return h5py.File(path, "r")


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
```

- [ ] **Step 2: Create the test fixture**

`tools/common/tests/conftest.py`:
```python
import json

import cv2
import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_a2d_h5(tmp_path):
    path = tmp_path / "a2d_tiny.h5"
    vlen = h5py.vlen_dtype(np.uint8)

    def enc(img, ext):
        ok, buf = cv2.imencode(ext, img)
        assert ok
        return np.frombuffer(buf.tobytes(), np.uint8)

    params = {
        "intrinsic": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0},
        "extrinsic": {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      "translation_vector": [0, 0, 0]},
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("parameters/camera/head.json", data=json.dumps(params))
        cds = f.create_dataset("cameras/head/color/data", (2,), dtype=vlen)
        dds = f.create_dataset("cameras/head/depth/data", (2,), dtype=vlen)
        for i in range(2):
            cds[i] = enc((np.random.rand(48, 64, 3) * 255).astype(np.uint8), ".jpg")
            dds[i] = enc(np.full((48, 64), 700, np.uint16), ".png")
        hp = f.create_dataset("joints/state/head/position", (2, 2), dtype="float32")
        hp[:] = [[0.1, 0.2], [0.3, 0.4]]
    return str(path)
```

- [ ] **Step 3: Create the robotkit tests**

`tools/common/tests/test_config.py`:
```python
import json
import pytest

from robotkit.config import load_config


def _write(tmp_path, cfg, urdf_name="r.urdf"):
    (tmp_path / urdf_name).write_text("<robot name='r'></robot>")
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _good(urdf_name="r.urdf"):
    return {"urdf": urdf_name, "base_link": "base_link", "base_forward_axis": "-x",
            "joint_mapping": {}, "cameras": {"head": {"mount_link": "h", "modality": "depth"}},
            "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]}}


def test_load_valid(tmp_path):
    cfg = load_config(_write(tmp_path, _good()))
    assert cfg["urdf_resolved"].endswith("r.urdf")


def test_missing_key_raises(tmp_path):
    bad = _good(); del bad["thresholds"]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_bad_forward_axis_raises(tmp_path):
    bad = _good(); bad["base_forward_axis"] = "north"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_missing_urdf_raises(tmp_path):
    bad = _good(urdf_name="nope.urdf")
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(bad))
    with pytest.raises(FileNotFoundError):
        load_config(str(p))
```

`tools/common/tests/test_kinematics.py`:
```python
import os

import numpy as np
import pytest

from robotkit.kinematics import build_cfg, Kinematics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
G1_URDF = os.path.join(REPO, "example-dataset/guodi/a2d/g1/g1_flat.urdf")


def test_build_cfg_applies_sign_and_defaults():
    mapping = {"head": {"h5_path": "x", "entries": [
        {"h5_index": 0, "urdf_joint": "jA", "sign": -1},
        {"h5_index": 1, "urdf_joint": "jB"}]}}
    cfg = build_cfg({"head": np.array([0.5, 0.2])}, mapping, ["jA", "jB", "jC"])
    assert cfg == {"jA": -0.5, "jB": 0.2, "jC": 0.0}


def test_build_cfg_index_out_of_range():
    mapping = {"head": {"h5_path": "x", "entries": [{"h5_index": 5, "urdf_joint": "jA"}]}}
    with pytest.raises(ValueError):
        build_cfg({"head": np.array([0.1])}, mapping, ["jA"])


@pytest.mark.skipif(not os.path.exists(G1_URDF), reason="g1 urdf not present")
def test_fk_head_link2_matches_validated_pose():
    kin = Kinematics(G1_URDF, "base_link")
    mapping = {
        "waist": {"h5_path": "w", "entries": [
            {"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1},
            {"h5_index": 1, "urdf_joint": "idx01_body_joint1", "sign": 1}]},
        "head": {"h5_path": "h", "entries": [
            {"h5_index": 0, "urdf_joint": "idx11_head_joint1", "sign": 1},
            {"h5_index": 1, "urdf_joint": "idx12_head_joint2", "sign": 1}]}}
    fv = {"waist": np.array([0.7083, 0.3885]), "head": np.array([0.0, 0.4363])}
    cfg = build_cfg(fv, mapping, kin.robot.actuated_joint_names)
    T = kin.link_transform(cfg, "head_link2")
    assert np.allclose(T[:3, 3], [-0.157, 0.0, 1.450], atol=0.01)
```

`tools/common/tests/test_camera.py`:
```python
import numpy as np

from robotkit.camera import (
    read_intrinsic, read_extrinsic, extrinsic_matrix, resolve_extrinsic,
)
from robotkit.h5io import open_h5


def test_read_intrinsic(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        assert read_intrinsic(h5, "head")["fx"] == 600.0


def test_read_extrinsic_identity(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        assert np.allclose(read_extrinsic(h5, "head"), np.eye(4))


def test_extrinsic_matrix():
    E = extrinsic_matrix({"rotation_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                          "translation_vector": [1, 2, 3]})
    assert np.allclose(E[:3, 3], [1, 2, 3])
    assert np.allclose(E[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def test_resolve_extrinsic_prefers_config(tiny_a2d_h5):
    cam_cfg = {"extrinsic": {"rotation_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                             "translation_vector": [1.0, 2.0, 3.0]}}
    with open_h5(tiny_a2d_h5) as h5:
        E = resolve_extrinsic(cam_cfg, h5, "head")
    assert np.allclose(E[:3, 3], [1.0, 2.0, 3.0])


def test_resolve_extrinsic_falls_back_to_h5(tiny_a2d_h5):
    with open_h5(tiny_a2d_h5) as h5:
        E = resolve_extrinsic({"mount_link": "head_link2"}, h5, "head")
    assert np.allclose(E, np.eye(4))
```

`tools/common/tests/test_h5io.py`:
```python
import numpy as np

from robotkit.h5io import open_h5, decode_image, has_modality, read_frame_values


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
```

- [ ] **Step 4: Run robotkit tests**

Run: `uv run --with pytest python -m pytest tools/common/tests -v`
Expected: all PASS (config 4, kinematics 3, camera 5, h5io 3 = 15).

- [ ] **Step 5: Commit**

```bash
git add tools/common
git commit -m "feat(robotkit): add shared config/kinematics/camera/h5io package"
```

---

## Task 2: Refactor extrinsic-checker onto robotkit

**Files:** modify `tools/extrinsic-checker/conftest.py`, `tools/extrinsic-checker/extrinsic_check.py`, `tools/extrinsic-checker/extrinsic_checker/{depth_check,projection_check,cli}.py`; delete `extrinsic_checker/{loader,kinematics,h5read}.py` and `tests/{test_loader,test_kinematics,test_h5read}.py` and the fixture in `tests/conftest.py`.

- [ ] **Step 1: Point conftest + launcher at tools/common**

Replace `tools/extrinsic-checker/conftest.py` with:
```python
import os
import sys

# Make both this tool's package and the shared robotkit importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

In `tools/extrinsic-checker/extrinsic_check.py`, replace the single sys.path line:
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
with:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 2: Update imports in depth_check / projection_check / cli**

In `extrinsic_checker/depth_check.py`, replace:
```python
from .h5read import read_intrinsic, decode_image
```
with:
```python
from robotkit.camera import read_intrinsic
from robotkit.h5io import decode_image
```

In `extrinsic_checker/projection_check.py`, replace:
```python
from .h5read import read_intrinsic, decode_image
```
with:
```python
from robotkit.camera import read_intrinsic
from robotkit.h5io import decode_image
```

In `extrinsic_checker/cli.py`, replace these imports:
```python
from .loader import load_config
from .h5read import open_h5, read_frame_values, has_modality, resolve_extrinsic
from .kinematics import Kinematics, build_cfg
```
with:
```python
from robotkit.config import load_config
from robotkit.h5io import open_h5, read_frame_values, has_modality
from robotkit.camera import resolve_extrinsic
from robotkit.kinematics import Kinematics, build_cfg
```

- [ ] **Step 3: Delete moved modules and tests**

```bash
git rm tools/extrinsic-checker/extrinsic_checker/loader.py \
       tools/extrinsic-checker/extrinsic_checker/kinematics.py \
       tools/extrinsic-checker/extrinsic_checker/h5read.py \
       tools/extrinsic-checker/tests/test_loader.py \
       tools/extrinsic-checker/tests/test_kinematics.py \
       tools/extrinsic-checker/tests/test_h5read.py \
       tools/extrinsic-checker/tests/conftest.py
```
(The `tests/conftest.py` fixture moved to robotkit; the remaining extrinsic-checker tests — report, depth_check, projection_check — use only synthetic arrays.)

- [ ] **Step 4: Run extrinsic-checker tests**

Run: `uv run --with pytest python -m pytest tools/extrinsic-checker/tests -v`
Expected: report (2) + depth_check (4) + projection_check (3) = 9 PASS.

- [ ] **Step 5: Smoke-test the checker still works end-to-end**

Run:
```bash
uv run python tools/extrinsic-checker/extrinsic_check.py \
  --config tools/configs/a2d.json \
  --input  example-dataset/guodi/a2d/s1a1a435bba44e46a0abac239f78df74.h5 \
  --camera head --out-dir /tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad/extcheck_out
```
Expected: `[PASS] head (depth)` ... `OVERALL: PASS`.

- [ ] **Step 6: Commit**

```bash
git add tools/extrinsic-checker
git commit -m "refactor(extcheck): depend on shared robotkit package"
```

---

## Task 3: Scaffold lerobot-cam-pose-adder

**Files:** create `tools/lerobot-cam-pose-adder/conftest.py`, `lerobot_cam_pose_adder/__init__.py`, `pyproject.toml`.

- [ ] **Step 1: Verify parquet/scipy deps**

Run: `uv run python -c "import pandas, pyarrow, scipy, yourdfpy, numpy; print('deps ok')"`
Expected: `deps ok`. (If pyarrow missing: ask the user to add it to the uv env.)

- [ ] **Step 2: Create conftest + package init**

`tools/lerobot-cam-pose-adder/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/__init__.py`:
```python
"""lerobot_cam_pose_adder — add per-frame camera poses to a LeRobot v3 dataset."""

__version__ = "0.1.0"
```

`tools/lerobot-cam-pose-adder/pyproject.toml`:
```toml
[project]
name = "lerobot-cam-pose-adder"
version = "0.1.0"
description = "Add per-frame camera poses (base_link, pos+quat) to a LeRobot v3 dataset."
requires-python = ">=3.10"
dependencies = ["yourdfpy", "numpy", "scipy", "pandas", "pyarrow"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["lerobot_cam_pose_adder"]
```

- [ ] **Step 3: Verify imports**

Run: `PYTHONPATH=tools/common:tools/lerobot-cam-pose-adder uv run python -c "import lerobot_cam_pose_adder, robotkit; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add tools/lerobot-cam-pose-adder/conftest.py tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/__init__.py tools/lerobot-cam-pose-adder/pyproject.toml
git commit -m "feat(campose): scaffold lerobot-cam-pose-adder"
```

---

## Task 4: fieldmap.py

**Files:** create `lerobot_cam_pose_adder/fieldmap.py`, `tests/test_fieldmap.py`.

- [ ] **Step 1: Write failing tests**

`tools/lerobot-cam-pose-adder/tests/test_fieldmap.py`:
```python
import pytest

from lerobot_cam_pose_adder.fieldmap import load_fieldmap, state_index_lookup


def _fm(tmp_path):
    import json
    p = tmp_path / "fm.json"
    p.write_text(json.dumps({"features": {"observation.state": [
        {"lerobot_index": 0, "name": "a", "h5_path": "joints/state/arm/position", "h5_index": 0},
        {"lerobot_index": 18, "name": "bp", "h5_path": "joints/state/waist/position", "h5_index": 0},
    ]}}))
    return str(p)


def test_lookup(tmp_path):
    fm = load_fieldmap(_fm(tmp_path))
    lut = state_index_lookup(fm, "observation.state")
    assert lut[("joints/state/waist/position", 0)] == 18
    assert lut[("joints/state/arm/position", 0)] == 0


def test_missing_feature_raises(tmp_path):
    fm = load_fieldmap(_fm(tmp_path))
    with pytest.raises(ValueError):
        state_index_lookup(fm, "nope")
```

- [ ] **Step 2: Run (expect fail)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_fieldmap.py -v`
Expected: FAIL — `ModuleNotFoundError: ... fieldmap`.

- [ ] **Step 3: Implement fieldmap.py**

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/fieldmap.py`:
```python
"""Load the h5<->lerobot field map and index it for state-column lookup."""

import json


def load_fieldmap(path):
    with open(path) as f:
        fm = json.load(f)
    if "features" not in fm:
        raise ValueError("fieldmap missing 'features'")
    return fm


def state_index_lookup(fieldmap, feature="observation.state"):
    """Return {(h5_path, h5_index): lerobot_index} for a vector feature."""
    comps = fieldmap["features"].get(feature)
    if comps is None:
        raise ValueError(f"fieldmap has no feature {feature!r}")
    out = {}
    for c in comps:
        if "lerobot_index" in c:
            out[(c["h5_path"], c["h5_index"])] = c["lerobot_index"]
    return out
```

- [ ] **Step 4: Run (expect pass)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_fieldmap.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/fieldmap.py tools/lerobot-cam-pose-adder/tests/test_fieldmap.py
git commit -m "feat(campose): add field-map loader + state index lookup"
```

---

## Task 5: poses.py

**Files:** create `lerobot_cam_pose_adder/poses.py`, `tests/test_poses.py`.

- [ ] **Step 1: Write failing tests**

`tools/lerobot-cam-pose-adder/tests/test_poses.py`:
```python
import numpy as np
import pytest

from lerobot_cam_pose_adder.poses import mat_to_pose, state_to_cfg, camera_poses_row


def test_mat_to_pose_z90():
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   # +90 deg about z
    T[:3, 3] = [1, 2, 3]
    p = mat_to_pose(T)
    assert np.allclose(p[:3], [1, 2, 3])
    assert np.allclose(p[3:], [0, 0, 0.70710678, 0.70710678], atol=1e-6)


def test_state_to_cfg_applies_sign():
    jm = {"waist": {"h5_path": "joints/state/waist/position",
                    "entries": [{"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1}]}}
    lut = {("joints/state/waist/position", 0): 18}
    state = np.zeros(27); state[18] = 0.7
    cfg = state_to_cfg(state, jm, lut, ["idx02_body_joint2", "other"])
    assert cfg == {"idx02_body_joint2": -0.7, "other": 0.0}


def test_state_to_cfg_missing_key_raises():
    jm = {"waist": {"h5_path": "p", "entries": [{"h5_index": 0, "urdf_joint": "j"}]}}
    with pytest.raises(ValueError):
        state_to_cfg(np.zeros(5), jm, {}, ["j"])


def test_camera_poses_row_concatenates():
    class StubKin:
        def link_transform(self, cfg, link):
            return np.eye(4)
    E = np.eye(4); E[:3, 3] = [1, 2, 3]
    out = camera_poses_row({}, StubKin(), [("head", "L1", E), ("hand_left", "L2", np.eye(4))])
    assert out.shape == (14,) and out.dtype == np.float32
    assert np.allclose(out[:3], [1, 2, 3])
    assert np.allclose(out[7:10], [0, 0, 0])
```

- [ ] **Step 2: Run (expect fail)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_poses.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement poses.py**

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/poses.py`:
```python
"""Compute camera poses from a LeRobot state row + config."""

import numpy as np
from scipy.spatial.transform import Rotation


def mat_to_pose(T):
    """4x4 homogeneous -> [x, y, z, qx, qy, qz, qw] (quaternion xyzw)."""
    t = np.asarray(T)[:3, 3]
    q = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()  # xyzw
    return np.concatenate([t, q])


def state_to_cfg(state_row, joint_mapping, index_lookup, actuated_joints):
    """Build {urdf_joint: value} from a LeRobot state row.

    joint_mapping: config's joint_mapping (group -> {h5_path, entries:[{h5_index,urdf_joint,sign}]}).
    index_lookup:  {(h5_path, h5_index): lerobot_index} from the field map.
    """
    cfg = {j: 0.0 for j in actuated_joints}
    for group, spec in joint_mapping.items():
        h5_path = spec["h5_path"]
        for e in spec["entries"]:
            key = (h5_path, e["h5_index"])
            if key not in index_lookup:
                raise ValueError(f"joint {key} not found in fieldmap observation.state")
            cfg[e["urdf_joint"]] = float(state_row[index_lookup[key]]) * e.get("sign", 1)
    return cfg


def camera_poses_row(cfg, kin, cameras):
    """cameras: ordered list of (name, mount_link, E 4x4). Returns float32 vector of 7*len."""
    parts = []
    for _name, mount_link, E in cameras:
        T = kin.link_transform(cfg, mount_link) @ E
        parts.append(mat_to_pose(T))
    return np.concatenate(parts).astype(np.float32)
```

- [ ] **Step 4: Run (expect pass)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_poses.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/poses.py tools/lerobot-cam-pose-adder/tests/test_poses.py
git commit -m "feat(campose): add pose math + state->cfg chaining"
```

---

## Task 6: dataset.py

**Files:** create `lerobot_cam_pose_adder/dataset.py`, `tests/test_dataset.py`.

- [ ] **Step 1: Write failing tests**

`tools/lerobot-cam-pose-adder/tests/test_dataset.py`:
```python
import json
import os

import numpy as np
import pytest

from lerobot_cam_pose_adder.dataset import (
    copy_dataset, feature_names, add_feature_to_info, add_feature_to_stats, stats_for,
)


def test_feature_names():
    cams = [("head", "L", None), ("hand_left", "L", None)]
    names = feature_names(cams)
    assert len(names) == 14
    assert names[0] == "head_x" and names[6] == "head_qw" and names[7] == "hand_left_x"


def test_stats_for():
    arr = np.array([[0.0, 10.0], [2.0, 14.0]])
    s = stats_for(arr)
    assert s["count"] == [2]
    assert np.allclose(s["mean"], [1.0, 12.0])
    assert np.allclose(s["min"], [0.0, 10.0]) and np.allclose(s["max"], [2.0, 14.0])
    assert set(s) == {"mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"}


def test_add_feature_to_info(tmp_path):
    p = tmp_path / "info.json"
    p.write_text(json.dumps({"features": {"observation.state": {"dtype": "float32", "shape": [27]}}}))
    add_feature_to_info(str(p), "observation.camera_poses", 21, ["head_x"] * 21)
    info = json.loads(p.read_text())
    assert info["features"]["observation.camera_poses"]["shape"] == [21]


def test_add_feature_to_info_duplicate_raises(tmp_path):
    p = tmp_path / "info.json"
    p.write_text(json.dumps({"features": {"observation.camera_poses": {}}}))
    with pytest.raises(ValueError):
        add_feature_to_info(str(p), "observation.camera_poses", 21, [])


def test_add_feature_to_stats(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"observation.state": {"mean": [0.0]}}))
    add_feature_to_stats(str(p), "observation.camera_poses", np.zeros((3, 21)))
    stats = json.loads(p.read_text())
    assert stats["observation.camera_poses"]["count"] == [3]


def test_copy_dataset_and_overwrite(tmp_path):
    src = tmp_path / "src"; (src / "meta").mkdir(parents=True)
    (src / "meta" / "info.json").write_text("{}")
    dst = tmp_path / "dst"
    copy_dataset(str(src), str(dst))
    assert os.path.exists(dst / "meta" / "info.json")
    with pytest.raises(FileExistsError):
        copy_dataset(str(src), str(dst))
    copy_dataset(str(src), str(dst), overwrite=True)  # no raise
```

- [ ] **Step 2: Run (expect fail)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_dataset.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement dataset.py**

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/dataset.py`:
```python
"""Copy a LeRobot dataset and inject a new vector feature + its metadata."""

import json
import os
import shutil

import numpy as np

_SUFFIX = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def copy_dataset(src, dst, overwrite=False):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"output exists: {dst} (use --overwrite)")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def feature_names(cameras):
    names = []
    for cam in cameras:
        name = cam[0]
        names += [f"{name}_{s}" for s in _SUFFIX]
    return names


def stats_for(arr):
    a = np.asarray(arr, dtype=np.float64)
    q = np.quantile(a, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "count": [int(a.shape[0])],
        "q01": q[0].tolist(), "q10": q[1].tolist(), "q50": q[2].tolist(),
        "q90": q[3].tolist(), "q99": q[4].tolist(),
    }


def add_feature_to_info(info_path, feature, dim, names):
    with open(info_path) as f:
        info = json.load(f)
    if feature in info["features"]:
        raise ValueError(f"dataset already has feature {feature!r}")
    info["features"][feature] = {"dtype": "float32", "shape": [dim], "names": names}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def add_feature_to_stats(stats_path, feature, all_values):
    with open(stats_path) as f:
        stats = json.load(f)
    stats[feature] = stats_for(all_values)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
```

- [ ] **Step 4: Run (expect pass)**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests/test_dataset.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/dataset.py tools/lerobot-cam-pose-adder/tests/test_dataset.py
git commit -m "feat(campose): add dataset copy + info/stats feature injection"
```

---

## Task 7: cli.py + launcher + __main__

**Files:** create `lerobot_cam_pose_adder/cli.py`, `lerobot_cam_pose_adder/__main__.py`, and the launcher `tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py`.

- [ ] **Step 1: Implement cli.py**

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/cli.py`:
```python
"""CLI: add observation.camera_poses to a copy of a LeRobot v3 dataset."""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from robotkit.config import load_config
from robotkit.kinematics import Kinematics
from robotkit.camera import extrinsic_matrix
from .fieldmap import load_fieldmap, state_index_lookup
from .poses import state_to_cfg, camera_poses_row
from .dataset import copy_dataset, feature_names, add_feature_to_info, add_feature_to_stats

CAMERA_ORDER = ["head", "hand_left", "hand_right"]
FEATURE = "observation.camera_poses"


def build_parser():
    p = argparse.ArgumentParser(
        prog="lerobot-cam-pose-adder",
        description="Add per-frame camera poses (base_link, pos+quat) to a LeRobot v3 dataset.")
    p.add_argument("--dataset", required=True, help="Input LeRobot v3 dataset dir")
    p.add_argument("--config", required=True, help="Robot config JSON (urdf, joint_mapping, cameras/extrinsic)")
    p.add_argument("--fieldmap", required=True, help="h5<->lerobot field map JSON")
    p.add_argument("--output", required=True, help="Output dataset dir (a copy)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite --output if it exists")
    return p


def build_cameras(cfg):
    cams = []
    for name in CAMERA_ORDER:
        cc = cfg["cameras"][name]
        cams.append((name, cc["mount_link"], extrinsic_matrix(cc["extrinsic"])))
    return cams


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        lookup = state_index_lookup(load_fieldmap(args.fieldmap), "observation.state")
        kin = Kinematics(cfg["urdf_resolved"], cfg["base_link"])
        cameras = build_cameras(cfg)
        actuated = kin.robot.actuated_joint_names

        copy_dataset(args.dataset, args.output, overwrite=args.overwrite)

        chunks = []
        for pq in sorted(glob.glob(os.path.join(args.output, "data", "**", "*.parquet"), recursive=True)):
            df = pd.read_parquet(pq)
            rows = [camera_poses_row(
                        state_to_cfg(np.asarray(s, float), cfg["joint_mapping"], lookup, actuated),
                        kin, cameras)
                    for s in df["observation.state"]]
            mat = np.stack(rows)
            df[FEATURE] = list(mat)
            df.to_parquet(pq, index=False)
            chunks.append(mat)
        if not chunks:
            raise ValueError(f"no data parquet files found under {args.output}/data")
        allv = np.concatenate(chunks)

        add_feature_to_info(os.path.join(args.output, "meta", "info.json"),
                            FEATURE, allv.shape[1], feature_names(cameras))
        add_feature_to_stats(os.path.join(args.output, "meta", "stats.json"), FEATURE, allv)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output} with {FEATURE} ({allv.shape[1]} dims, {allv.shape[0]} frames)")
    return 0
```

- [ ] **Step 2: Create __main__.py and launcher**

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/__main__.py`:
```python
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

`tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py`:
```python
#!/usr/bin/env python3
"""Standalone launcher: run from any directory.

    uv run python tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py \
        --dataset DIR --config tools/configs/a2d.json \
        --fieldmap tools/configs/a2d_h5_lerobot_map.json --output OUT
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "common"))
sys.path.insert(0, _here)

from lerobot_cam_pose_adder.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify CLI parses**

Run: `uv run python tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py --help`
Expected: usage with `--dataset --config --fieldmap --output --overwrite`.

- [ ] **Step 4: Commit**

```bash
git add tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/cli.py tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder/__main__.py tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py
git commit -m "feat(campose): add CLI orchestration + launcher"
```

---

## Task 8: Real-data integration, cross-check, README

**Files:** create `tools/lerobot-cam-pose-adder/README.md`.

- [ ] **Step 1: Run on the real a2d LeRobot dataset**

Run:
```bash
SP=/tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad
uv run python tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py \
  --dataset  example-dataset/guodi/a2d/lerobotV3 \
  --config   tools/configs/a2d.json \
  --fieldmap tools/configs/a2d_h5_lerobot_map.json \
  --output   $SP/lerobotV3_cam --overwrite
```
Expected: prints `wrote .../lerobotV3_cam with observation.camera_poses (21 dims, 855 frames)`.

- [ ] **Step 2: Verify feature, metadata, and cross-check vs the h5 path**

Run:
```bash
SP=/tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad
uv run python - <<'PY'
import json, glob, numpy as np, pandas as pd, os, sys
sys.path.insert(0, "tools/common")
from robotkit.config import load_config
from robotkit.kinematics import Kinematics, build_cfg
from robotkit.camera import extrinsic_matrix
from robotkit.h5io import open_h5, read_frame_values
sys.path.insert(0, "tools/lerobot-cam-pose-adder")
from lerobot_cam_pose_adder.poses import mat_to_pose

SP=os.environ.get("SP") or "/tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad"
out=f"{SP}/lerobotV3_cam"
info=json.load(open(f"{out}/meta/info.json"))
feat=info["features"]["observation.camera_poses"]
assert feat["shape"]==[21] and len(feat["names"])==21, feat
stats=json.load(open(f"{out}/meta/stats.json"))
assert stats["observation.camera_poses"]["count"][0]==855
df=pd.read_parquet(glob.glob(f"{out}/data/**/*.parquet", recursive=True)[0])
cp0=np.asarray(df["observation.camera_poses"].iloc[0], float)
assert cp0.shape==(21,)

# cross-check head pose (first 7) against the h5-derived FK path
cfg=load_config("tools/configs/a2d.json")
kin=Kinematics(cfg["urdf_resolved"], cfg["base_link"])
with open_h5("example-dataset/guodi/a2d/s1a1a435bba44e46a0abac239f78df74.h5") as h5:
    fv=read_frame_values(h5, cfg["joint_mapping"], 0)
jcfg=build_cfg(fv, cfg["joint_mapping"], kin.robot.actuated_joint_names)
E=extrinsic_matrix(cfg["cameras"]["head"]["extrinsic"])
expected=mat_to_pose(kin.link_transform(jcfg, cfg["cameras"]["head"]["mount_link"]) @ E)
assert np.allclose(cp0[:7], expected, atol=1e-4), (cp0[:7], expected)

# original dataset untouched (no camera_poses column there)
orig=pd.read_parquet(glob.glob("example-dataset/guodi/a2d/lerobotV3/data/**/*.parquet", recursive=True)[0])
assert "observation.camera_poses" not in orig.columns
print("OK: feature shape 21, stats count 855, head pose cross-check matches, original unchanged")
PY
```
Expected: prints the `OK: ...` line.

- [ ] **Step 3: Write README**

`tools/lerobot-cam-pose-adder/README.md`:
````markdown
# lerobot-cam-pose-adder

Add a per-frame `observation.camera_poses` feature (21-dim: head, hand_left,
hand_right; each `[x,y,z,qx,qy,qz,qw]` in **base_link**, quaternion xyzw) to a
**copy** of a LeRobot v3 dataset. The original dataset is never modified.

Poses are computed from the dataset's own `observation.state` joints (no h5
needed): joints are located via the field map, the URDF FK gives the mount-link
pose, and `T_base_cam = T_base_link @ E` uses the config extrinsic.

## Usage

```bash
uv run python tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py \
  --dataset  /path/to/lerobotV3 \
  --config   tools/configs/a2d.json \
  --fieldmap tools/configs/a2d_h5_lerobot_map.json \
  --output   /path/to/lerobotV3_with_poses \
  [--overwrite]
```

Inputs:
- `--config`: robot config (urdf, base_link, joint_mapping with signs, cameras with mount_link + extrinsic). Shared with extrinsic-checker (`tools/configs/`).
- `--fieldmap`: per-dimension h5<->lerobot field map (locates the state columns).

The output is a self-contained LeRobot v3 dataset with the new feature added to
`data/*.parquet`, `meta/info.json`, and `meta/stats.json`.

## Tests

```bash
uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests -v
```
````

- [ ] **Step 4: Run the full tool test suite**

Run: `uv run --with pytest python -m pytest tools/lerobot-cam-pose-adder/tests -v`
Expected: fieldmap (2) + poses (4) + dataset (6) = 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/lerobot-cam-pose-adder/README.md
git commit -m "docs(campose): add README + real-data integration verified"
```

---

## Self-Review Notes

- **Spec coverage:** robotkit extraction (Task 1) + extrinsic-checker refactor keeping tests green (Task 2); inputs from lerobot state + config + fieldmap, no h5 (Tasks 4/5/7); chain via (h5_path,h5_index) (fieldmap Task 4 + state_to_cfg Task 5); FK ∘ config-extrinsic → pos+quat xyzw (Task 5); 21-dim head/hand_left/hand_right feature (cli CAMERA_ORDER + feature_names); write to --output copy, original untouched, info+stats(incl. quantiles) updated (Task 6 + cli Task 7); cross-check vs extrinsic-checker T_base_cam (Task 8). All covered.
- **Type consistency:** `load_config→cfg["urdf_resolved"]`, `Kinematics(...).link_transform(cfg, link)`, `extrinsic_matrix(extr)`, `state_index_lookup(fieldmap, feature)→{(h5_path,h5_index):idx}`, `state_to_cfg(state_row, joint_mapping, index_lookup, actuated_joints)`, `camera_poses_row(cfg, kin, cameras)` with `cameras=[(name,mount_link,E)]`, `mat_to_pose(T)→7`, `feature_names(cameras)`, `stats_for(arr)`, `add_feature_to_info(path, feature, dim, names)`, `add_feature_to_stats(path, feature, all_values)`. Consistent across tasks.
- **No placeholders:** every code/test step is complete. Orchestration (`cli.main`) is exercised by the Task 8 real-data run; pure functions are unit-tested in Tasks 4–6.
