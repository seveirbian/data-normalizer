# h5vid-exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tool at `tools/h5vid-exporter/` that renders one or more camera image-stream topics from an h5 file into a single side-by-side (1×N) mp4 video.

**Architecture:** A small Python package `h5vid_exporter` split by responsibility — `reader` (h5 access + topic validation), `render` (decode bytes → labeled RGB tile, color vs depth by decoded shape), `compose` (hstack tiles), `exporter` (orchestrate + encode with `cv2.VideoWriter`), and `cli` (argparse). Tests use a synthetic tiny h5 fixture.

**Tech Stack:** Python 3.10+, h5py, numpy, opencv-python (decode + colormap + encode). pytest for tests. All already installed in the environment; no network install required.

---

## File Structure

```
tools/h5vid-exporter/
  pyproject.toml              # package metadata + console script (h5vid-export)
  README.md                   # usage
  conftest.py                 # empty — makes pytest add this dir to sys.path
  h5vid_exporter/
    __init__.py
    reader.py                 # H5Reader: topic validation, frame counts, frame bytes
    render.py                 # decode + color/depth tile + label + resize
    compose.py                # compose_row (hstack)
    exporter.py               # export_video orchestration + cv2 encode
    cli.py                    # main(argv) argparse entry
    __main__.py               # python -m h5vid_exporter
  tests/
    conftest.py               # tiny_h5 / mismatch_h5 fixtures
    test_reader.py
    test_render.py
    test_compose.py
    test_exporter.py
    test_cli.py
```

All test/run commands assume current directory `tools/h5vid-exporter/`. The empty top-level `conftest.py` makes pytest treat that directory as rootdir and prepend it to `sys.path`, so `import h5vid_exporter` resolves without installation.

---

## Task 1: Scaffold the package

**Files:**
- Create: `tools/h5vid-exporter/conftest.py`
- Create: `tools/h5vid-exporter/h5vid_exporter/__init__.py`
- Create: `tools/h5vid-exporter/pyproject.toml`

- [ ] **Step 1: Verify dependencies are importable**

Run:
```bash
python3 -c "import h5py, numpy, cv2, pytest; print('deps ok', cv2.__version__)"
```
Expected: prints `deps ok 4.x.x` with no ImportError. (If anything is missing: `pip install h5py numpy opencv-python pytest`.)

- [ ] **Step 2: Create the empty top-level conftest (sys.path hook)**

Create `tools/h5vid-exporter/conftest.py` with exactly this content:

```python
# Presence of this file makes pytest use this directory as rootdir and
# prepend it to sys.path, so `import h5vid_exporter` works without install.
```

- [ ] **Step 3: Create the package init**

Create `tools/h5vid-exporter/h5vid_exporter/__init__.py`:

```python
"""h5vid_exporter — render h5 camera image-stream topics into a side-by-side mp4."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create pyproject.toml for the tool**

Create `tools/h5vid-exporter/pyproject.toml`:

```toml
[project]
name = "h5vid-exporter"
version = "0.1.0"
description = "Render h5 camera image-stream topics into a side-by-side mp4 video."
requires-python = ">=3.10"
dependencies = ["h5py", "numpy", "opencv-python"]

[project.scripts]
h5vid-export = "h5vid_exporter.cli:main"

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["h5vid_exporter"]
```

- [ ] **Step 5: Verify the package imports**

Run:
```bash
cd tools/h5vid-exporter && python3 -c "import h5vid_exporter; print(h5vid_exporter.__version__)"
```
Expected: prints `0.1.0`.

- [ ] **Step 6: Commit**

```bash
git add tools/h5vid-exporter/conftest.py tools/h5vid-exporter/h5vid_exporter/__init__.py tools/h5vid-exporter/pyproject.toml
git commit -m "feat(h5vid): scaffold h5vid-exporter package"
```

---

## Task 2: Test fixtures

**Files:**
- Create: `tools/h5vid-exporter/tests/conftest.py`

- [ ] **Step 1: Create the fixtures**

Create `tools/h5vid-exporter/tests/conftest.py`:

```python
import cv2
import h5py
import numpy as np
import pytest


def _encode_jpg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _encode_png16(depth):
    ok, buf = cv2.imencode(".png", depth)
    assert ok
    return buf.tobytes()


def _write_h5(path, topic_to_count):
    vlen = h5py.vlen_dtype(np.uint8)
    with h5py.File(path, "w") as f:
        for topic, (kind, n) in topic_to_count.items():
            ds = f.create_dataset(f"{topic}/data", (n,), dtype=vlen)
            for i in range(n):
                if kind == "color":
                    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
                    b = _encode_jpg(img)
                else:  # depth
                    depth = (np.arange(48 * 64).reshape(48, 64) % 1000).astype(np.uint16)
                    b = _encode_png16(depth)
                ds[i] = np.frombuffer(b, dtype=np.uint8)


@pytest.fixture
def tiny_h5(tmp_path):
    """h5 with one color + one depth topic, 3 aligned frames each."""
    path = tmp_path / "tiny.h5"
    _write_h5(path, {
        "cameras/head/color": ("color", 3),
        "cameras/head/depth": ("depth", 3),
    })
    return str(path)


@pytest.fixture
def mismatch_h5(tmp_path):
    """h5 where topics have differing frame counts (3 vs 2)."""
    path = tmp_path / "mismatch.h5"
    _write_h5(path, {
        "cameras/head/color": ("color", 3),
        "cameras/hand_left/color": ("color", 2),
    })
    return str(path)
```

- [ ] **Step 2: Verify fixtures construct a readable h5**

Run:
```bash
cd tools/h5vid-exporter && python3 -c "
import h5py, tempfile, os
from tests.conftest import _write_h5
d = tempfile.mkdtemp(); p = os.path.join(d, 't.h5')
_write_h5(p, {'cameras/head/color': ('color', 3), 'cameras/head/depth': ('depth', 3)})
with h5py.File(p) as f:
    print(sorted(k for k in [] ) or 'ok', f['cameras/head/color/data'].shape, f['cameras/head/depth/data'].shape)
"
```
Expected: prints `ok (3,) (3,)`.

- [ ] **Step 3: Commit**

```bash
git add tools/h5vid-exporter/tests/conftest.py
git commit -m "test(h5vid): add synthetic h5 fixtures"
```

---

## Task 3: reader.py

**Files:**
- Create: `tools/h5vid-exporter/h5vid_exporter/reader.py`
- Test: `tools/h5vid-exporter/tests/test_reader.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/h5vid-exporter/tests/test_reader.py`:

```python
import pytest

from h5vid_exporter.reader import H5Reader


def test_available_camera_topics(tiny_h5):
    with H5Reader(tiny_h5) as r:
        assert r.available_camera_topics() == [
            "cameras/head/color",
            "cameras/head/depth",
        ]


def test_validate_topic_ok(tiny_h5):
    with H5Reader(tiny_h5) as r:
        r.validate_topic("cameras/head/color")  # no raise


def test_validate_topic_missing_lists_available(tiny_h5):
    with H5Reader(tiny_h5) as r:
        with pytest.raises(ValueError) as exc:
            r.validate_topic("cameras/nope/color")
    msg = str(exc.value)
    assert "cameras/nope/color" in msg
    assert "cameras/head/color" in msg


def test_validate_topic_group_without_data(tiny_h5):
    with H5Reader(tiny_h5) as r:
        with pytest.raises(ValueError):
            r.validate_topic("cameras/head")  # group, but no /data child


def test_frame_count(tiny_h5):
    with H5Reader(tiny_h5) as r:
        assert r.frame_count("cameras/head/color") == 3


def test_frame_bytes_decodable(tiny_h5):
    import numpy as np
    import cv2
    with H5Reader(tiny_h5) as r:
        b = r.frame_bytes("cameras/head/color", 0)
    assert isinstance(b, bytes) and len(b) > 0
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape == (48, 64, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_reader.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'h5vid_exporter.reader'`.

- [ ] **Step 3: Implement reader.py**

Create `tools/h5vid-exporter/h5vid_exporter/reader.py`:

```python
"""Read camera image-stream topics from a guodi-style h5 file.

A "topic" is a group path (e.g. ``cameras/head/color``) that directly contains
a ``data`` dataset of shape ``(num_frames,)`` holding encoded jpg/png bytes.
"""

import h5py
import numpy as np


class H5Reader:
    def __init__(self, path):
        self._f = h5py.File(path, "r")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def available_camera_topics(self):
        """Sorted list of topic paths: groups that directly contain a 'data' dataset."""
        topics = []

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "data" in obj:
                child = obj["data"]
                if isinstance(child, h5py.Dataset):
                    topics.append(name)

        self._f.visititems(visit)
        return sorted(topics)

    def _is_topic(self, topic):
        grp = self._f.get(topic)
        return (
            isinstance(grp, h5py.Group)
            and "data" in grp
            and isinstance(grp["data"], h5py.Dataset)
        )

    def validate_topic(self, topic):
        if not self._is_topic(topic):
            raise ValueError(
                f"Topic {topic!r} is not a valid image-stream topic. "
                f"Available topics: {self.available_camera_topics()}"
            )

    def frame_count(self, topic):
        return int(self._f[topic]["data"].shape[0])

    def frame_bytes(self, topic, index):
        elem = self._f[topic]["data"][index]
        if isinstance(elem, bytes):
            return elem
        if isinstance(elem, np.ndarray):
            return elem.tobytes()
        return bytes(elem)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_reader.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/h5vid-exporter/h5vid_exporter/reader.py tools/h5vid-exporter/tests/test_reader.py
git commit -m "feat(h5vid): add H5Reader with topic validation"
```

---

## Task 4: render.py

**Files:**
- Create: `tools/h5vid-exporter/h5vid_exporter/render.py`
- Test: `tools/h5vid-exporter/tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/h5vid-exporter/tests/test_render.py`:

```python
import cv2
import numpy as np
import pytest

from h5vid_exporter.render import (
    decode_frame,
    to_rgb_tile,
    add_label,
    resize_to_height,
    render_tile,
)


def _jpg_bytes():
    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _png16_bytes():
    depth = (np.arange(48 * 64).reshape(48, 64) % 1000).astype(np.uint16)
    ok, buf = cv2.imencode(".png", depth)
    assert ok
    return buf.tobytes()


def test_decode_frame_color():
    img = decode_frame(_jpg_bytes())
    assert img.shape == (48, 64, 3)


def test_decode_frame_bad_bytes_raises():
    with pytest.raises(ValueError):
        decode_frame(b"not an image")


def test_to_rgb_tile_color_keeps_size_3ch():
    img = decode_frame(_jpg_bytes())
    tile = to_rgb_tile(img)
    assert tile.shape == (48, 64, 3) and tile.dtype == np.uint8


def test_to_rgb_tile_depth_2d_becomes_3ch():
    depth = decode_frame(_png16_bytes())
    assert depth.ndim == 2  # 16-bit png decodes single-channel
    tile = to_rgb_tile(depth)
    assert tile.shape == (48, 64, 3) and tile.dtype == np.uint8


def test_resize_to_height():
    tile = np.zeros((48, 64, 3), np.uint8)
    out = resize_to_height(tile, 240)
    assert out.shape[0] == 240 and out.shape[2] == 3


def test_add_label_keeps_shape():
    tile = np.zeros((100, 120, 3), np.uint8)
    out = add_label(tile, "cameras/head/color")
    assert out.shape == (100, 120, 3)


def test_render_tile_end_to_end():
    out = render_tile(_jpg_bytes(), "cameras/head/color", 120)
    assert out.shape[0] == 120 and out.shape[2] == 3 and out.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_render.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'h5vid_exporter.render'`.

- [ ] **Step 3: Implement render.py**

Create `tools/h5vid-exporter/h5vid_exporter/render.py`:

```python
"""Turn encoded frame bytes into a labeled RGB tile ready for compositing.

Color vs depth is decided by the *decoded array shape*, not by the topic name,
so it is robust to naming differences across robot platforms.
"""

import cv2
import numpy as np


def decode_frame(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode frame bytes as an image")
    return img


def _depth_to_rgb(depth):
    d = depth.astype(np.float32)
    valid = d[d > 0]
    if valid.size:
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        span = max(hi - lo, 1.0)
        norm = np.clip((d - lo) / span * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.clip(d, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_rgb_tile(img):
    """Decoded image array -> 3-channel uint8 RGB tile."""
    if img.ndim == 2:
        return _depth_to_rgb(img)
    if img.shape[2] == 1:
        return _depth_to_rgb(img[:, :, 0])
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def resize_to_height(tile, height):
    h, w = tile.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(tile, (new_w, height))


def add_label(tile, text):
    out = tile.copy()
    w = out.shape[1]
    cv2.rectangle(out, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_tile(buf, topic, height):
    """bytes -> resized, labeled RGB tile of the given height."""
    img = decode_frame(buf)
    rgb = to_rgb_tile(img)
    resized = resize_to_height(rgb, height)
    return add_label(resized, topic)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_render.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/h5vid-exporter/h5vid_exporter/render.py tools/h5vid-exporter/tests/test_render.py
git commit -m "feat(h5vid): add frame rendering (color/depth tiles)"
```

---

## Task 5: compose.py

**Files:**
- Create: `tools/h5vid-exporter/h5vid_exporter/compose.py`
- Test: `tools/h5vid-exporter/tests/test_compose.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/h5vid-exporter/tests/test_compose.py`:

```python
import numpy as np
import pytest

from h5vid_exporter.compose import compose_row


def test_compose_row_hstacks_same_height():
    a = np.zeros((120, 80, 3), np.uint8)
    b = np.zeros((120, 100, 3), np.uint8)
    out = compose_row([a, b])
    assert out.shape == (120, 180, 3)


def test_compose_row_single_tile():
    a = np.zeros((120, 80, 3), np.uint8)
    out = compose_row([a])
    assert out.shape == (120, 80, 3)


def test_compose_row_differing_heights_raises():
    a = np.zeros((120, 80, 3), np.uint8)
    b = np.zeros((100, 80, 3), np.uint8)
    with pytest.raises(ValueError):
        compose_row([a, b])


def test_compose_row_empty_raises():
    with pytest.raises(ValueError):
        compose_row([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_compose.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'h5vid_exporter.compose'`.

- [ ] **Step 3: Implement compose.py**

Create `tools/h5vid-exporter/h5vid_exporter/compose.py`:

```python
"""Composite same-frame tiles into a single horizontal-row frame (1xN)."""

import numpy as np


def compose_row(tiles):
    if not tiles:
        raise ValueError("No tiles to compose")
    heights = {t.shape[0] for t in tiles}
    if len(heights) != 1:
        raise ValueError(f"Tiles have differing heights: {sorted(heights)}")
    return np.hstack(tiles)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_compose.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/h5vid-exporter/h5vid_exporter/compose.py tools/h5vid-exporter/tests/test_compose.py
git commit -m "feat(h5vid): add row composition"
```

---

## Task 6: exporter.py

**Files:**
- Create: `tools/h5vid-exporter/h5vid_exporter/exporter.py`
- Test: `tools/h5vid-exporter/tests/test_exporter.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/h5vid-exporter/tests/test_exporter.py`:

```python
import os

import cv2
import pytest

from h5vid_exporter.exporter import export_video


def _read_video(path):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, w, h


def test_export_video_creates_readable_mp4(tiny_h5, tmp_path):
    out = str(tmp_path / "out.mp4")
    export_video(tiny_h5, ["cameras/head/color", "cameras/head/depth"], out,
                 fps=10, height=120)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    frames, w, h = _read_video(out)
    assert frames == 3
    assert h == 120          # height preserved (even)
    assert w >= 120          # two tiles side by side


def test_export_video_no_topics_raises(tiny_h5, tmp_path):
    with pytest.raises(ValueError):
        export_video(tiny_h5, [], str(tmp_path / "x.mp4"))


def test_export_video_invalid_topic_raises(tiny_h5, tmp_path):
    with pytest.raises(ValueError):
        export_video(tiny_h5, ["cameras/nope/color"], str(tmp_path / "x.mp4"))


def test_export_video_frame_count_mismatch_warns_and_uses_min(mismatch_h5, tmp_path):
    out = str(tmp_path / "out.mp4")
    with pytest.warns(UserWarning):
        export_video(mismatch_h5,
                     ["cameras/head/color", "cameras/hand_left/color"],
                     out, fps=10, height=120)
    frames, _, _ = _read_video(out)
    assert frames == 2       # minimum of (3, 2)


def test_export_video_creates_output_dir(tiny_h5, tmp_path):
    out = str(tmp_path / "nested" / "dir" / "out.mp4")
    export_video(tiny_h5, ["cameras/head/color"], out, fps=10, height=120)
    assert os.path.exists(out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_exporter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'h5vid_exporter.exporter'`.

- [ ] **Step 3: Implement exporter.py**

Create `tools/h5vid-exporter/h5vid_exporter/exporter.py`:

```python
"""Orchestrate reading, rendering, composing and encoding into an mp4.

Encoding uses cv2.VideoWriter with the mp4v codec (see the design spec for why
this is preferred over imageio/ffmpeg in this environment).
"""

import os
import warnings

import cv2
import numpy as np

from .compose import compose_row
from .reader import H5Reader
from .render import render_tile


def _pad_even(frame):
    h, w = frame.shape[:2]
    ph, pw = h % 2, w % 2
    if ph or pw:
        frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), constant_values=0)
    return frame


def export_video(input_path, topics, output_path, fps=30, height=480):
    if not topics:
        raise ValueError("At least one topic is required")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with H5Reader(input_path) as reader:
        for topic in topics:
            reader.validate_topic(topic)

        counts = {t: reader.frame_count(t) for t in topics}
        n = min(counts.values())
        if len(set(counts.values())) > 1:
            warnings.warn(
                f"Topics have differing frame counts {counts}; using minimum {n}"
            )

        writer = None
        try:
            for i in range(n):
                tiles = [render_tile(reader.frame_bytes(t, i), t, height)
                         for t in topics]
                frame = _pad_even(compose_row(tiles))  # RGB
                if writer is None:
                    fh, fw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (fw, fh))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer for {output_path!r}"
                        )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_exporter.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/h5vid-exporter/h5vid_exporter/exporter.py tools/h5vid-exporter/tests/test_exporter.py
git commit -m "feat(h5vid): add export_video orchestration + mp4 encoding"
```

---

## Task 7: cli.py + entry point

**Files:**
- Create: `tools/h5vid-exporter/h5vid_exporter/cli.py`
- Create: `tools/h5vid-exporter/h5vid_exporter/__main__.py`
- Test: `tools/h5vid-exporter/tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tools/h5vid-exporter/tests/test_cli.py`:

```python
import os

from h5vid_exporter.cli import main


def test_cli_main_creates_video(tiny_h5, tmp_path):
    out = str(tmp_path / "cli.mp4")
    rc = main([
        "--input", tiny_h5,
        "--topics", "cameras/head/color", "cameras/head/depth",
        "--output", out,
        "--fps", "10",
        "--height", "120",
    ])
    assert rc == 0
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_cli_main_invalid_topic_returns_nonzero(tiny_h5, tmp_path, capsys):
    out = str(tmp_path / "cli.mp4")
    rc = main([
        "--input", tiny_h5,
        "--topics", "cameras/nope/color",
        "--output", out,
    ])
    assert rc != 0
    assert not os.path.exists(out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_cli.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'h5vid_exporter.cli'`.

- [ ] **Step 3: Implement cli.py**

Create `tools/h5vid-exporter/h5vid_exporter/cli.py`:

```python
"""Command-line entry point for h5vid-exporter."""

import argparse
import sys

from .exporter import export_video


def build_parser():
    p = argparse.ArgumentParser(
        prog="h5vid-export",
        description="Render h5 camera image-stream topics into a side-by-side mp4.",
    )
    p.add_argument("--input", required=True, help="Path to the input .h5 file")
    p.add_argument("--topics", required=True, nargs="+",
                   help="One or more full topic paths, e.g. cameras/head/color")
    p.add_argument("--output", required=True, help="Output .mp4 path")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second (default 30)")
    p.add_argument("--height", type=int, default=480,
                   help="Uniform per-tile height in pixels (default 480)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        export_video(args.input, args.topics, args.output,
                     fps=args.fps, height=args.height)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0
```

- [ ] **Step 4: Create the module entry point**

Create `tools/h5vid-exporter/h5vid_exporter/__main__.py`:

```python
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest tests/test_cli.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Run the full test suite**

Run:
```bash
cd tools/h5vid-exporter && python3 -m pytest -v
```
Expected: all tests across the 5 test files PASS (24 total).

- [ ] **Step 7: Commit**

```bash
git add tools/h5vid-exporter/h5vid_exporter/cli.py tools/h5vid-exporter/h5vid_exporter/__main__.py tools/h5vid-exporter/tests/test_cli.py
git commit -m "feat(h5vid): add CLI entry point"
```

---

## Task 8: README + real-data smoke check

**Files:**
- Create: `tools/h5vid-exporter/README.md`

- [ ] **Step 1: Write the README**

Create `tools/h5vid-exporter/README.md`:

````markdown
# h5vid-exporter

Render one or more camera image-stream topics from a guodi-style `.h5` file into
a single side-by-side (1×N) mp4 video.

## Topics

A *topic* is the full group path of an image stream, e.g.:

- `cameras/head/color`
- `cameras/head/depth`
- `cameras/hand_left/color`

Color streams are rendered as-is; depth streams are rendered with a JET colormap
(2nd–98th percentile normalization). Color vs depth is detected from the decoded
frame shape, not the topic name.

## Usage

Run from this directory (`tools/h5vid-exporter/`):

```bash
python3 -m h5vid_exporter \
  --input /path/to/file.h5 \
  --topics cameras/head/color cameras/head/depth \
  --output out.mp4 \
  --fps 30 \
  --height 480
```

Options:

- `--input` (required): input `.h5` file.
- `--topics` (required, 1+): full topic paths.
- `--output` (required): output `.mp4` path.
- `--fps`: frames per second, default `30`.
- `--height`: uniform per-tile height in pixels, default `480`.

If a topic is invalid, the tool prints the list of available topics in that file.

## Tests

```bash
python3 -m pytest -v
```
````

- [ ] **Step 2: Smoke-test against a real dataset file**

Run (uses the real ur sample; writes to scratch):
```bash
cd tools/h5vid-exporter && python3 -m h5vid_exporter \
  --input /root/codes/data-normalizer/example-dataset/guodi/ur/s1a0fb41b8c1434d96b9d5ccf0356baa.h5 \
  --topics cameras/head/color cameras/head/depth cameras/hand_left/color cameras/hand_right/color \
  --output /tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad/ur_all.mp4 \
  --fps 30 --height 480
```
Expected: prints `wrote .../ur_all.mp4`. Then verify:
```bash
python3 -c "import cv2; c=cv2.VideoCapture('/tmp/claude-0/-root-codes-data-normalizer/f75b6c3c-447e-4d75-83e7-26f201f6d16f/scratchpad/ur_all.mp4'); print('frames', int(c.get(7)), 'w', int(c.get(3)), 'h', int(c.get(4)))"
```
Expected: `frames 3385 w <≈4*width> h 480`.

- [ ] **Step 3: Commit**

```bash
git add tools/h5vid-exporter/README.md
git commit -m "docs(h5vid): add README"
```

---

## Self-Review Notes

- **Spec coverage:** camera-only topics (Task 3/4), full-path topics + validation listing available (Task 3), 1×N horizontal layout (Task 5), per-tile label (Task 4), color/depth by decoded shape (Task 4), `--fps` default 30 not read from file (Task 7), min-frame-count alignment + warning (Task 6), cv2 mp4v encoding (Task 6), TDD with synthetic fixture (Tasks 2–7). All covered.
- **Type consistency:** `H5Reader.validate_topic/frame_count/frame_bytes`, `render_tile(buf, topic, height)`, `compose_row(tiles)`, `export_video(input_path, topics, output_path, fps, height)`, `main(argv)` — names are consistent across tasks.
- **No placeholders:** every code/test step contains complete content.
