# extrinsic-checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a platform-agnostic tool at `tools/extrinsic-checker/` that verifies a dataset's camera extrinsics by reconstructing each camera into the robot base frame (depth cameras → point-cloud plane check; color cameras → known-point projection check), driven entirely by an external self-contained JSON config.

**Architecture:** A Python package `extrinsic_checker` split by responsibility — `loader` (config), `h5read` (h5 IO), `kinematics` (yourdfpy FK + joint-cfg assembly), `depth_check` and `projection_check` (the two methods + pure verdict functions), `report` (Verdict + printing), `cli`. An external `configs/a2d.json` carries all a2d specifics. Pure verdict/assembly functions are unit-tested; FK and the full pipeline are covered by an integration test and a real-data smoke test.

**Tech Stack:** Python 3.12 in the project's **uv** environment (has open3d, yourdfpy, h5py, opencv-python-headless, numpy, scipy). Run everything via `uv run` from the repo root.

**Conventions for this plan:**
- All `pytest`/run commands are executed **from the repo root** `/root/codes/data-normalizer` via `uv run`.
- Test command: `uv run python -m pytest tools/extrinsic-checker/tests -v`. The `tools/extrinsic-checker/conftest.py` file makes pytest prepend that directory to `sys.path`, so `import extrinsic_checker` resolves.

---

## File Structure

```
tools/extrinsic-checker/
  pyproject.toml                 # metadata + deps (documentation / standalone install)
  conftest.py                    # empty — makes pytest add this dir to sys.path
  extrinsic_check.py             # standalone launcher (any cwd)
  extrinsic_checker/
    __init__.py
    report.py                    # Verdict dataclass + print_report
    loader.py                    # load_config (+ validation, urdf path resolve)
    h5read.py                    # open_h5, read_intrinsic, read_extrinsic, decode_image, read_frame_values, has_modality
    kinematics.py                # build_cfg (pure) + Kinematics (yourdfpy wrapper)
    depth_check.py               # plane_verdict (pure) + run_depth_check + _topdown
    projection_check.py          # project_point (pure) + run_projection_check
    cli.py                       # argparse + orchestration + exit code
    __main__.py
  configs/
    a2d.json
  tests/
    conftest.py                  # tiny_a2d_h5 fixture
    test_report.py
    test_loader.py
    test_h5read.py
    test_kinematics.py
    test_depth_check.py
    test_projection_check.py
  README.md
```

---

## Task 1: Scaffold the package

**Files:**
- Create: `tools/extrinsic-checker/conftest.py`
- Create: `tools/extrinsic-checker/extrinsic_checker/__init__.py`
- Create: `tools/extrinsic-checker/pyproject.toml`
- Create: `tools/extrinsic-checker/configs/.gitkeep`

- [ ] **Step 1: Verify the uv env has the deps**

Run (from repo root):
```bash
uv run python -c "import open3d, yourdfpy, h5py, cv2, numpy; print('deps ok')"
```
Expected: prints `deps ok`.

- [ ] **Step 2: Create the empty top-level conftest (sys.path hook)**

Create `tools/extrinsic-checker/conftest.py`:
```python
# Presence of this file makes pytest add this directory to sys.path,
# so `import extrinsic_checker` works without installation.
```

- [ ] **Step 3: Create the package init**

Create `tools/extrinsic-checker/extrinsic_checker/__init__.py`:
```python
"""extrinsic_checker — verify camera extrinsics by reconstructing into the robot base frame."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create pyproject.toml**

Create `tools/extrinsic-checker/pyproject.toml`:
```toml
[project]
name = "extrinsic-checker"
version = "0.1.0"
description = "Verify dataset camera extrinsics via base-frame reconstruction (depth point-cloud + color projection)."
requires-python = ">=3.10"
dependencies = ["open3d", "yourdfpy", "h5py", "opencv-python-headless", "numpy"]

[project.scripts]
extrinsic-check = "extrinsic_checker.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["extrinsic_checker"]
```

- [ ] **Step 5: Create the configs dir placeholder**

Create `tools/extrinsic-checker/configs/.gitkeep` with empty content.

- [ ] **Step 6: Verify the package imports**

Run:
```bash
cd tools/extrinsic-checker && uv run --project /root/codes/data-normalizer python -c "import extrinsic_checker; print(extrinsic_checker.__version__)"; cd /root/codes/data-normalizer
```
Expected: prints `0.1.0`. (If the `cd` form is awkward, equivalently: `PYTHONPATH=tools/extrinsic-checker uv run python -c "import extrinsic_checker; print(extrinsic_checker.__version__)"`.)

- [ ] **Step 7: Commit**

```bash
git add tools/extrinsic-checker/conftest.py tools/extrinsic-checker/extrinsic_checker/__init__.py tools/extrinsic-checker/pyproject.toml tools/extrinsic-checker/configs/.gitkeep
git commit -m "feat(extcheck): scaffold extrinsic-checker package"
```

---

## Task 2: report.py (Verdict + printing)

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/report.py`
- Test: `tools/extrinsic-checker/tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tools/extrinsic-checker/tests/test_report.py`:
```python
from extrinsic_checker.report import Verdict, print_report


def test_print_report_all_pass(capsys):
    vs = [Verdict("head", "depth", True, {"plane_height": 0.84}, ["a.ply"]),
          Verdict("hand_left", "projection", True, {}, ["b.png"])]
    overall = print_report(vs)
    out = capsys.readouterr().out
    assert overall is True
    assert "head" in out and "PASS" in out


def test_print_report_one_fail(capsys):
    vs = [Verdict("head", "depth", True, {}, []),
          Verdict("hand_left", "projection", False, {}, [])]
    overall = print_report(vs)
    assert overall is False
    assert "FAIL" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.report'`.

- [ ] **Step 3: Implement report.py**

Create `tools/extrinsic-checker/extrinsic_checker/report.py`:
```python
"""Per-camera verdict structure, printing, and overall aggregation."""

from dataclasses import dataclass, field


@dataclass
class Verdict:
    camera: str
    method: str            # "depth" | "projection"
    passed: bool
    metrics: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)


def print_report(verdicts):
    """Print each verdict; return True iff all passed."""
    overall = True
    for v in verdicts:
        overall = overall and v.passed
        status = "PASS" if v.passed else "FAIL"
        print(f"[{status}] {v.camera} ({v.method})")
        for k, val in v.metrics.items():
            print(f"    {k}: {val}")
        for a in v.artifacts:
            print(f"    artifact: {a}")
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return overall
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_report.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/report.py tools/extrinsic-checker/tests/test_report.py
git commit -m "feat(extcheck): add Verdict + report printing"
```

---

## Task 3: loader.py (config loading + validation)

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/loader.py`
- Test: `tools/extrinsic-checker/tests/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/extrinsic-checker/tests/test_loader.py`:
```python
import json
import pytest

from extrinsic_checker.loader import load_config


def _write(tmp_path, cfg, urdf_name="r.urdf"):
    (tmp_path / urdf_name).write_text("<robot name='r'></robot>")
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _good(urdf_name="r.urdf"):
    return {
        "urdf": urdf_name,
        "base_link": "base_link",
        "base_forward_axis": "-x",
        "joint_mapping": {},
        "cameras": {"head": {"mount_link": "head_link2", "modality": "depth"}},
        "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]},
    }


def test_load_valid(tmp_path):
    cfg = load_config(_write(tmp_path, _good()))
    assert cfg["base_forward_axis"] == "-x"
    assert cfg["urdf_resolved"].endswith("r.urdf")


def test_missing_key_raises(tmp_path):
    bad = _good(); del bad["thresholds"]
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "thresholds" in str(e.value)


def test_bad_forward_axis_raises(tmp_path):
    bad = _good(); bad["base_forward_axis"] = "north"
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "base_forward_axis" in str(e.value)


def test_bad_modality_raises(tmp_path):
    bad = _good(); bad["cameras"]["head"]["modality"] = "thermal"
    with pytest.raises(ValueError) as e:
        load_config(_write(tmp_path, bad))
    assert "modality" in str(e.value)


def test_missing_urdf_raises(tmp_path):
    bad = _good(urdf_name="nope.urdf")
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(bad))  # no urdf file written
    with pytest.raises(FileNotFoundError):
        load_config(str(p))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.loader'`.

- [ ] **Step 3: Implement loader.py**

Create `tools/extrinsic-checker/extrinsic_checker/loader.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_loader.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/loader.py tools/extrinsic-checker/tests/test_loader.py
git commit -m "feat(extcheck): add config loader + validation"
```

---

## Task 4: test fixture + h5read.py

**Files:**
- Create: `tools/extrinsic-checker/tests/conftest.py`
- Create: `tools/extrinsic-checker/extrinsic_checker/h5read.py`
- Test: `tools/extrinsic-checker/tests/test_h5read.py`

- [ ] **Step 1: Create the synthetic h5 fixture**

Create `tools/extrinsic-checker/tests/conftest.py`:
```python
import json

import cv2
import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_a2d_h5(tmp_path):
    """Minimal a2d-like h5: head color+depth, head joint group, camera params."""
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

- [ ] **Step 2: Write the failing tests**

Create `tools/extrinsic-checker/tests/test_h5read.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_h5read.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.h5read'`.

- [ ] **Step 4: Implement h5read.py**

Create `tools/extrinsic-checker/extrinsic_checker/h5read.py`:
```python
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


def read_extrinsic(h5, cam):
    e = _params(h5, cam)["extrinsic"]
    E = np.eye(4)
    E[:3, :3] = np.array(e["rotation_matrix"], float)
    E[:3, 3] = np.array(e["translation_vector"], float)
    return E


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

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_h5read.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/extrinsic-checker/tests/conftest.py tools/extrinsic-checker/extrinsic_checker/h5read.py tools/extrinsic-checker/tests/test_h5read.py
git commit -m "feat(extcheck): add h5 readers + synthetic fixture"
```

---

## Task 5: kinematics.py (joint cfg assembly + FK)

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/kinematics.py`
- Test: `tools/extrinsic-checker/tests/test_kinematics.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/extrinsic-checker/tests/test_kinematics.py`:
```python
import os

import numpy as np
import pytest

from extrinsic_checker.kinematics import build_cfg, Kinematics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
G1_URDF = os.path.join(REPO, "example-dataset/guodi/a2d/g1/g1_flat.urdf")


def test_build_cfg_applies_sign_and_defaults():
    mapping = {"head": {"h5_path": "x", "entries": [
        {"h5_index": 0, "urdf_joint": "jA", "sign": -1},
        {"h5_index": 1, "urdf_joint": "jB"},
    ]}}
    fv = {"head": np.array([0.5, 0.2])}
    cfg = build_cfg(fv, mapping, ["jA", "jB", "jC"])
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
            {"h5_index": 1, "urdf_joint": "idx12_head_joint2", "sign": 1}]},
    }
    fv = {"waist": np.array([0.7083, 0.3885]), "head": np.array([0.0, 0.4363])}
    cfg = build_cfg(fv, mapping, kin.robot.actuated_joint_names)
    T = kin.link_transform(cfg, "head_link2")
    assert np.allclose(T[:3, 3], [0.495, 0.0, 1.385], atol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_kinematics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.kinematics'`.

- [ ] **Step 3: Implement kinematics.py**

Create `tools/extrinsic-checker/extrinsic_checker/kinematics.py`:
```python
"""Joint-config assembly (pure) and a thin yourdfpy FK wrapper."""

import yourdfpy


def build_cfg(frame_values, joint_mapping, actuated_joints):
    """Assemble a {urdf_joint: value} dict.

    frame_values: {group_name: 1D array already indexed at the frame}.
    Applies each entry's `sign` (default 1); unmapped joints default to 0.0.
    """
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
        """Return T_base_link (4x4) for `link` expressed in base_link, given cfg."""
        self.robot.update_cfg(cfg)
        return self.robot.get_transform(link, self.base_link)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_kinematics.py -v`
Expected: all 3 tests PASS (the FK test confirms head_link2 ≈ [0.495, 0, 1.385]).

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/kinematics.py tools/extrinsic-checker/tests/test_kinematics.py
git commit -m "feat(extcheck): add joint-cfg assembly + yourdfpy FK"
```

---

## Task 6: depth_check.py (plane verdict + pipeline)

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/depth_check.py`
- Test: `tools/extrinsic-checker/tests/test_depth_check.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/extrinsic-checker/tests/test_depth_check.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_depth_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.depth_check'`.

- [ ] **Step 3: Implement depth_check.py**

Create `tools/extrinsic-checker/extrinsic_checker/depth_check.py`:
```python
"""Depth-camera verification: point-cloud reprojection + dominant-plane verdict."""

import os

import cv2
import numpy as np
import open3d as o3d

from .h5read import read_intrinsic, read_extrinsic, decode_image
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


def run_depth_check(h5, cam, mount_link, kin, cfg, thresholds, forward_axis, frame, out_dir):
    I = read_intrinsic(h5, cam)
    E = read_extrinsic(h5, cam)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_depth_check.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/depth_check.py tools/extrinsic-checker/tests/test_depth_check.py
git commit -m "feat(extcheck): add depth point-cloud plane check"
```

---

## Task 7: projection_check.py (projection verdict + pipeline)

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/projection_check.py`
- Test: `tools/extrinsic-checker/tests/test_projection_check.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/extrinsic-checker/tests/test_projection_check.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_projection_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'extrinsic_checker.projection_check'`.

- [ ] **Step 3: Implement projection_check.py**

Create `tools/extrinsic-checker/extrinsic_checker/projection_check.py`:
```python
"""Color-camera verification: project known link points into the image."""

import os

import cv2
import numpy as np

from .h5read import read_intrinsic, read_extrinsic, decode_image
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


def run_projection_check(h5, cam, mount_link, targets, kin, cfg, frame, out_dir):
    I = read_intrinsic(h5, cam)
    E = read_extrinsic(h5, cam)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tools/extrinsic-checker/tests/test_projection_check.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/projection_check.py tools/extrinsic-checker/tests/test_projection_check.py
git commit -m "feat(extcheck): add color projection check"
```

---

## Task 8: cli.py + launcher

**Files:**
- Create: `tools/extrinsic-checker/extrinsic_checker/cli.py`
- Create: `tools/extrinsic-checker/extrinsic_checker/__main__.py`
- Create: `tools/extrinsic-checker/extrinsic_check.py`

- [ ] **Step 1: Implement cli.py**

Create `tools/extrinsic-checker/extrinsic_checker/cli.py`:
```python
"""Command-line entry point: orchestrate per-camera checks."""

import argparse
import sys

from .loader import load_config
from .h5read import open_h5, read_frame_values, has_modality
from .kinematics import Kinematics, build_cfg
from .depth_check import run_depth_check
from .projection_check import run_projection_check
from .report import print_report


def build_parser():
    p = argparse.ArgumentParser(
        prog="extrinsic-check",
        description="Verify dataset camera extrinsics via base-frame reconstruction.",
    )
    p.add_argument("--config", required=True, help="Path to the checker config JSON")
    p.add_argument("--input", required=True, help="Path to the .h5 recording")
    p.add_argument("--camera", nargs="+", default=None,
                   help="Cameras to check (default: all in config)")
    p.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    p.add_argument("--out-dir", default="extrinsic_check_out", help="Artifact output dir")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        kin = Kinematics(cfg["urdf_resolved"], cfg["base_link"])
        cameras = args.camera or list(cfg["cameras"].keys())
        with open_h5(args.input) as h5:
            frame_values = read_frame_values(h5, cfg["joint_mapping"], args.frame)
            jcfg = build_cfg(frame_values, cfg["joint_mapping"], kin.robot.actuated_joint_names)
            verdicts = []
            for cam in cameras:
                if cam not in cfg["cameras"]:
                    print(f"error: camera {cam!r} not in config; available {list(cfg['cameras'])}",
                          file=sys.stderr)
                    return 2
                cc = cfg["cameras"][cam]
                modality = cc["modality"]
                if not has_modality(h5, cam, modality):
                    print(f"error: camera {cam!r} modality {modality!r} absent in h5", file=sys.stderr)
                    return 2
                if modality == "depth":
                    v = run_depth_check(h5, cam, cc["mount_link"], kin, jcfg,
                                        cfg["thresholds"], cfg["base_forward_axis"],
                                        args.frame, args.out_dir)
                else:
                    v = run_projection_check(h5, cam, cc["mount_link"],
                                             cc.get("projection_targets", []), kin, jcfg,
                                             args.frame, args.out_dir)
                verdicts.append(v)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ok = print_report(verdicts)
    return 0 if ok else 1
```

- [ ] **Step 2: Create the module entry point**

Create `tools/extrinsic-checker/extrinsic_checker/__main__.py`:
```python
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Create the launcher**

Create `tools/extrinsic-checker/extrinsic_check.py`:
```python
#!/usr/bin/env python3
"""Standalone launcher: run extrinsic-checker from any directory without install.

    uv run python tools/extrinsic-checker/extrinsic_check.py \
        --config tools/extrinsic-checker/configs/a2d.json --input FILE.h5

Adds this directory to sys.path so the ``extrinsic_checker`` package imports
regardless of the current working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extrinsic_checker.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify the CLI parses and errors cleanly**

Run:
```bash
uv run python tools/extrinsic-checker/extrinsic_check.py --help
```
Expected: prints usage with `--config`, `--input`, `--camera`, `--frame`, `--out-dir`.

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/extrinsic_checker/cli.py tools/extrinsic-checker/extrinsic_checker/__main__.py tools/extrinsic-checker/extrinsic_check.py
git commit -m "feat(extcheck): add CLI orchestration + launcher"
```

---

## Task 9: a2d config, real-data smoke, README

**Files:**
- Create: `tools/extrinsic-checker/configs/a2d.json`
- Create: `tools/extrinsic-checker/README.md`

- [ ] **Step 1: Write the a2d config**

Create `tools/extrinsic-checker/configs/a2d.json`:
```json
{
  "urdf": "../../../example-dataset/guodi/a2d/g1/g1_flat.urdf",
  "base_link": "base_link",
  "base_forward_axis": "-x",
  "joint_mapping": {
    "waist": {
      "h5_path": "joints/state/waist/position",
      "entries": [
        {"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1},
        {"h5_index": 1, "urdf_joint": "idx01_body_joint1", "sign": 1}
      ]
    },
    "head": {
      "h5_path": "joints/state/head/position",
      "entries": [
        {"h5_index": 0, "urdf_joint": "idx11_head_joint1", "sign": 1},
        {"h5_index": 1, "urdf_joint": "idx12_head_joint2", "sign": 1}
      ]
    },
    "arm": {
      "h5_path": "joints/state/arm/position",
      "entries": [
        {"h5_index": 0, "urdf_joint": "idx21_arm_l_joint1", "sign": 1},
        {"h5_index": 1, "urdf_joint": "idx22_arm_l_joint2", "sign": 1},
        {"h5_index": 2, "urdf_joint": "idx23_arm_l_joint3", "sign": 1},
        {"h5_index": 3, "urdf_joint": "idx24_arm_l_joint4", "sign": 1},
        {"h5_index": 4, "urdf_joint": "idx25_arm_l_joint5", "sign": 1},
        {"h5_index": 5, "urdf_joint": "idx26_arm_l_joint6", "sign": 1},
        {"h5_index": 6, "urdf_joint": "idx27_arm_l_joint7", "sign": 1},
        {"h5_index": 7, "urdf_joint": "idx61_arm_r_joint1", "sign": 1},
        {"h5_index": 8, "urdf_joint": "idx62_arm_r_joint2", "sign": 1},
        {"h5_index": 9, "urdf_joint": "idx63_arm_r_joint3", "sign": 1},
        {"h5_index": 10, "urdf_joint": "idx64_arm_r_joint4", "sign": 1},
        {"h5_index": 11, "urdf_joint": "idx65_arm_r_joint5", "sign": 1},
        {"h5_index": 12, "urdf_joint": "idx66_arm_r_joint6", "sign": 1},
        {"h5_index": 13, "urdf_joint": "idx67_arm_r_joint7", "sign": 1}
      ]
    }
  },
  "cameras": {
    "head": {"mount_link": "head_link2", "modality": "depth"},
    "hand_left": {"mount_link": "arm_l_end_link", "modality": "color",
                  "projection_targets": ["gripper_left_center_link"]},
    "hand_right": {"mount_link": "arm_r_end_link", "modality": "color",
                   "projection_targets": ["gripper_right_center_link"]}
  },
  "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]}
}
```

- [ ] **Step 2: Smoke-test the head depth check on real a2d data**

Run (from repo root):
```bash
uv run python tools/extrinsic-checker/extrinsic_check.py \
  --config tools/extrinsic-checker/configs/a2d.json \
  --input  example-dataset/guodi/a2d/s1a1a435bba44e46a0abac239f78df74.h5 \
  --camera head \
  --out-dir /tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad/extcheck_out
```
Expected: report shows `[PASS] head (depth)` with `vertical_align` near 1.0, `plane_height` ≈ 0.84, `front_ok: True`; `OVERALL: PASS`; exit code 0. Artifacts `head_base.ply` and `head_topdown.png` written to the out-dir.

- [ ] **Step 3: Run the full test suite**

Run:
```bash
uv run python -m pytest tools/extrinsic-checker/tests -v
```
Expected: all unit + integration tests PASS (report, loader, h5read, kinematics incl. FK, depth_check, projection_check).

- [ ] **Step 4: Write the README**

Create `tools/extrinsic-checker/README.md`:
````markdown
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

Intrinsics and extrinsics are read from the h5 (`parameters/camera/<cam>.json`);
the config only names each camera's `mount_link`, `modality`, and (for color)
`projection_targets`.

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
uv run python -m pytest tools/extrinsic-checker/tests -v
```
````

- [ ] **Step 5: Commit**

```bash
git add tools/extrinsic-checker/configs/a2d.json tools/extrinsic-checker/README.md
git commit -m "docs(extcheck): add a2d config + README"
```

---

## Self-Review Notes

- **Spec coverage:** generic package + external config (Tasks 1,3,9); two methods auto-selected by modality (cli Task 8 → depth Task 6 / projection Task 7); intrinsics/extrinsics from h5 (h5read Task 4); FK with sign overrides (kinematics Task 5, validated by FK test = [0.495,0,1.385]); depth criteria — horizontal/height/front (plane_verdict Task 6); projection criteria — Z>0/in-image (Task 7); artifacts PLY+topdown / overlay (Tasks 6,7); CLI flags + exit code (Task 8); error handling for missing camera/modality/config/joint-range (loader Task 3, cli Task 8, build_cfg Task 5); a2d config + smoke (Task 9); TDD throughout. All covered.
- **Type consistency:** `Verdict(camera, method, passed, metrics, artifacts)`; `load_config→cfg["urdf_resolved"]`; `build_cfg(frame_values, joint_mapping, actuated_joints)`; `Kinematics(urdf_path, base_link).link_transform(cfg, link)`; `plane_verdict(points, thresholds, forward_axis)→(passed, metrics)`; `project_point(p_base, T_base_link, E, fx,fy,cx,cy,W,H)→dict`; `run_depth_check(h5, cam, mount_link, kin, cfg, thresholds, forward_axis, frame, out_dir)`; `run_projection_check(h5, cam, mount_link, targets, kin, cfg, frame, out_dir)`. Names consistent across tasks.
- **No placeholders:** every code/test step contains complete content.
- **Note:** `run_depth_check`/`run_projection_check` orchestration is exercised by the Task 9 real-data smoke test (pure functions `plane_verdict`/`project_point` are unit-tested in Tasks 6/7).
