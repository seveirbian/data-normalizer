# LeRobot v3 Format Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `convert_v3.py` CLI that converts A2D HDF5 robot data directly to LeRobot v3.0 format, with size-based multi-episode parquet and video file splitting.

**Architecture:** Three new files (`parquet_writer_v3.py`, `meta_builder_v3.py`, `convert_v3.py`) implement the v3 path while sharing `reader.py` and `video_writer.py` unchanged. Data and video files hold multiple episodes up to configurable size limits (100 MB / 200 MB) with ffmpeg concat for video assembly. Multi-batch append reads existing `tasks.parquet` and `meta/episodes/` parquet on startup.

**Tech Stack:** Python 3.10, h5py, pandas, pyarrow, numpy, opencv-python-headless, ffmpeg subprocess, pytest

**Run all tests with:** `/usr/bin/python3 -m pytest /root/codes/a2d-lerobot-converter/tests/ -v`

**Working directory for all commands:** `/root/codes/a2d-lerobot-converter`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `h5_lerobot_converter/parquet_writer_v3.py` | Create | Accumulates rows across episodes; flushes to `data/chunk-NNN/file-NNN.parquet` when size limit hit |
| `h5_lerobot_converter/meta_builder_v3.py` | Create | Writes `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`, `meta/info.json` (v3.0), `meta/stats.json` |
| `h5_lerobot_converter/convert_v3.py` | Create | `DatasetBuilderV3` orchestrator + CLI (`python3 -m h5_lerobot_converter.convert_v3`) |
| `tests/test_parquet_writer_v3.py` | Create | Unit tests for `ParquetWriterV3` |
| `tests/test_meta_builder_v3.py` | Create | Unit tests for `MetaBuilderV3` |
| `tests/test_convert_v3.py` | Create | Integration tests for full v3 pipeline |
| `h5_lerobot_converter/reader.py` | Unchanged | Shared HDF5 reader |
| `h5_lerobot_converter/video_writer.py` | Unchanged | Shared ffmpeg video writer |

---

## Task 1: `ParquetWriterV3` — size-based multi-episode parquet writer

**Files:**
- Create: `h5_lerobot_converter/parquet_writer_v3.py`
- Test: `tests/test_parquet_writer_v3.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_parquet_writer_v3.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.parquet_writer_v3 import ParquetWriterV3
from h5_lerobot_converter.reader import EpisodeData


def _make_episode(n_frames=10):
    return EpisodeData(
        state=np.zeros((n_frames, 27), dtype=np.float32),
        action=np.zeros((n_frames, 25), dtype=np.float32),
        timestamps=np.arange(n_frames, dtype=np.float32) / 30.0,
        images={},
        n_frames=n_frames,
    )


def test_single_episode_file_location(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    loc = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    writer.flush()

    assert loc == {"chunk_index": 0, "file_index": 0}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()


def test_single_episode_schema_and_values(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    ep = _make_episode(n_frames=5)
    writer.write_episode(ep, episode_index=0, global_start_index=0, task_index=2)
    writer.flush()

    df = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert set(df.columns) >= {
        "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "index", "task_index", "next.done"
    }
    assert len(df) == 5
    assert df["index"].tolist() == [0, 1, 2, 3, 4]
    assert df["episode_index"].tolist() == [0, 0, 0, 0, 0]
    assert df["task_index"].tolist() == [2, 2, 2, 2, 2]
    assert df["next.done"].tolist() == [False, False, False, False, True]
    assert len(df["observation.state"].iloc[0]) == 27
    assert len(df["action"].iloc[0]) == 25


def test_two_episodes_same_file(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == loc1 == {"chunk_index": 0, "file_index": 0}
    df = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert len(df) == 20
    assert df["index"].tolist() == list(range(20))


def test_size_limit_triggers_new_file(tmp_path):
    # Tiny size limit so ep1 triggers internal flush of ep0 and goes to file-001
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=1e-9)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    # Do NOT flush manually — ep0 stays buffered so the size check fires for ep1
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == {"chunk_index": 0, "file_index": 0}
    assert loc1 == {"chunk_index": 0, "file_index": 1}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "data" / "chunk-000" / "file-001.parquet").exists()


def test_chunk_boundary_resets_file_index(tmp_path):
    # chunk_size=1: episode 0 → chunk 0, episode 1 → chunk 1
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1, max_size_mb=100.0)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    writer.flush()
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == {"chunk_index": 0, "file_index": 0}
    assert loc1 == {"chunk_index": 1, "file_index": 0}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "data" / "chunk-001" / "file-000.parquet").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/usr/bin/python3 -m pytest tests/test_parquet_writer_v3.py -v
```

Expected: `ModuleNotFoundError: No module named 'h5_lerobot_converter.parquet_writer_v3'`

- [ ] **Step 3: Implement `parquet_writer_v3.py`**

Create `h5_lerobot_converter/parquet_writer_v3.py`:

```python
from pathlib import Path
import pandas as pd
from h5_lerobot_converter.reader import EpisodeData


class ParquetWriterV3:
    def __init__(self, data_dir: Path, chunk_size: int = 1000, max_size_mb: float = 100.0):
        self._data_dir = Path(data_dir)
        self._chunk_size = chunk_size
        self._max_bytes = max_size_mb * 1024 * 1024
        self._rows: list = []
        self._current_bytes: int = 0
        self._chunk_index: int = 0
        self._file_index: int = 0

    @staticmethod
    def _estimate_bytes(data: EpisodeData) -> int:
        # (27 + 25) float32 + 1 float32 timestamp + 5 int32 + 1 bool, per frame
        return data.n_frames * ((27 + 25 + 1) * 4 + 5 * 4 + 1)

    def write_episode(
        self,
        data: EpisodeData,
        episode_index: int,
        global_start_index: int,
        task_index: int,
    ) -> dict:
        ep_chunk = episode_index // self._chunk_size
        ep_bytes = self._estimate_bytes(data)

        if ep_chunk != self._chunk_index:
            if self._rows:
                self._flush()
            self._chunk_index = ep_chunk
            self._file_index = 0
        elif self._rows and self._current_bytes + ep_bytes > self._max_bytes:
            self._flush()
            self._file_index += 1

        n = data.n_frames
        done = [False] * (n - 1) + [True]
        for i in range(n):
            self._rows.append({
                "observation.state": data.state[i].tolist(),
                "action":            data.action[i].tolist(),
                "timestamp":         float(data.timestamps[i]),
                "frame_index":       i,
                "episode_index":     episode_index,
                "index":             global_start_index + i,
                "task_index":        task_index,
                "next.done":         done[i],
            })
        self._current_bytes += ep_bytes

        return {"chunk_index": self._chunk_index, "file_index": self._file_index}

    def flush(self) -> None:
        if self._rows:
            self._flush()

    def _flush(self) -> None:
        out = (
            self._data_dir
            / f"chunk-{self._chunk_index:03d}"
            / f"file-{self._file_index:03d}.parquet"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(out, index=False)
        self._rows = []
        self._current_bytes = 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/usr/bin/python3 -m pytest tests/test_parquet_writer_v3.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/a2d-lerobot-converter add h5_lerobot_converter/parquet_writer_v3.py tests/test_parquet_writer_v3.py
git -C /root/codes/a2d-lerobot-converter commit -m "feat: add ParquetWriterV3 with size-based multi-episode file splitting"
```

---

## Task 2: `MetaBuilderV3` — v3 metadata files

**Files:**
- Create: `h5_lerobot_converter/meta_builder_v3.py`
- Test: `tests/test_meta_builder_v3.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_meta_builder_v3.py`:

```python
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.meta_builder_v3 import MetaBuilderV3


@pytest.fixture
def builder(tmp_path):
    return MetaBuilderV3(tmp_path, fps=30.0, chunk_size=1000,
                         data_file_size_mb=100.0, video_file_size_mb=200.0)


def test_write_tasks_creates_parquet(builder, tmp_path):
    builder.write_tasks(["task A", "task B"])
    df = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
    assert set(df.columns) == {"task_index", "task"}
    assert df["task"].tolist() == ["task A", "task B"]
    assert df["task_index"].tolist() == [0, 1]


def test_write_episodes_creates_parquet(builder, tmp_path):
    episodes = [
        {
            "episode_index": 0, "tasks": ["task A"], "length": 10,
            "dataset_from_index": 0, "dataset_to_index": 10,
            "data/chunk_index": 0, "data/file_index": 0,
            "videos/observation.images.hand_left/chunk_index": 0,
            "videos/observation.images.hand_left/file_index": 0,
            "videos/observation.images.hand_right/chunk_index": 0,
            "videos/observation.images.hand_right/file_index": 0,
            "videos/observation.images.head/chunk_index": 0,
            "videos/observation.images.head/file_index": 0,
            "videos/observation.images.head_depth/chunk_index": 0,
            "videos/observation.images.head_depth/file_index": 0,
        }
    ]
    builder.write_episodes(episodes)
    ep_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    assert ep_path.exists()
    df = pd.read_parquet(ep_path)
    assert len(df) == 1
    assert df["episode_index"].iloc[0] == 0
    assert df["data/chunk_index"].iloc[0] == 0
    assert df["data/file_index"].iloc[0] == 0


def test_write_info_v3(builder, tmp_path):
    builder.write_info(
        total_episodes=2, total_frames=20, tasks=["task A"],
        image_shapes={"observation.images.head": (720, 1280, 3)}
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20
    assert info["data_files_size_in_mb"] == 100.0
    assert info["video_files_size_in_mb"] == 200.0
    assert "observation.images.head" in info["features"]


def test_write_stats_no_nan(builder, tmp_path):
    # Write a small data parquet manually, then compute stats from it
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    rows = []
    for i in range(5):
        rows.append({
            "observation.state": [float(i)] * 27,
            "action": [float(i)] * 25,
            "timestamp": i / 30.0,
            "frame_index": i,
            "episode_index": 0,
            "index": i,
            "task_index": 0,
            "next.done": i == 4,
        })
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    episodes = [{"data/chunk_index": 0, "data/file_index": 0, "length": 5}]
    builder.write_stats(episodes)

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    for key in ["mean", "std", "min", "max"]:
        vals = stats["observation.state"][key]
        assert not any(v != v or abs(v) == float("inf") for v in vals)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/usr/bin/python3 -m pytest tests/test_meta_builder_v3.py -v
```

Expected: `ModuleNotFoundError: No module named 'h5_lerobot_converter.meta_builder_v3'`

- [ ] **Step 3: Implement `meta_builder_v3.py`**

Create `h5_lerobot_converter/meta_builder_v3.py`:

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

from h5_lerobot_converter.meta_builder import STATE_NAMES, ACTION_NAMES

CAMERA_KEYS = [
    "observation.images.hand_left",
    "observation.images.hand_right",
    "observation.images.head",
    "observation.images.head_depth",
]


class MetaBuilderV3:
    def __init__(
        self,
        output_dir: Path,
        fps: float,
        chunk_size: int = 1000,
        data_file_size_mb: float = 100.0,
        video_file_size_mb: float = 200.0,
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_size = chunk_size
        self.data_file_size_mb = data_file_size_mb
        self.video_file_size_mb = video_file_size_mb
        self.meta_dir = self.output_dir / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def write_tasks(self, tasks: list) -> None:
        df = pd.DataFrame({"task_index": list(range(len(tasks))), "task": tasks})
        df.to_parquet(self.meta_dir / "tasks.parquet", index=False)

    def write_episodes(self, episodes: list) -> None:
        out_path = self.meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episodes).to_parquet(out_path, index=False)

    def write_info(
        self,
        total_episodes: int,
        total_frames: int,
        tasks: list,
        image_shapes: dict,
    ) -> None:
        features = {
            "observation.state": {"dtype": "float32", "shape": [27], "names": STATE_NAMES},
            "action":            {"dtype": "float32", "shape": [25], "names": ACTION_NAMES},
            "timestamp":         {"dtype": "float32", "shape": [1],  "names": None},
            "frame_index":       {"dtype": "int64",   "shape": [1],  "names": None},
            "episode_index":     {"dtype": "int64",   "shape": [1],  "names": None},
            "index":             {"dtype": "int64",   "shape": [1],  "names": None},
            "task_index":        {"dtype": "int64",   "shape": [1],  "names": None},
            "next.done":         {"dtype": "bool",    "shape": [1],  "names": None},
        }
        for cam_name, shape in image_shapes.items():
            features[cam_name] = {
                "dtype": "video",
                "shape": list(shape),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": self.fps,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": "depth" in cam_name,
                    "video.height": shape[0],
                    "video.width": shape[1],
                },
            }

        info = {
            "codebase_version": "v3.0",
            "robot_type": "a2d",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "data_files_size_in_mb": self.data_file_size_mb,
            "video_files_size_in_mb": self.video_file_size_mb,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def write_stats(self, episodes: list) -> None:
        seen = set()
        all_states, all_actions = [], []
        for ep in episodes:
            key = (ep["data/chunk_index"], ep["data/file_index"])
            if key in seen:
                continue
            seen.add(key)
            ci, fi = key
            pq = self.output_dir / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
            df = pd.read_parquet(pq, columns=["observation.state", "action"])
            all_states.extend(df["observation.state"].tolist())
            all_actions.extend(df["action"].tolist())

        states  = np.array(all_states,  dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)

        def _stats(arr):
            return {
                "mean": arr.mean(axis=0).tolist(),
                "std":  arr.std(axis=0).tolist(),
                "min":  arr.min(axis=0).tolist(),
                "max":  arr.max(axis=0).tolist(),
            }

        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump({"observation.state": _stats(states), "action": _stats(actions)}, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/usr/bin/python3 -m pytest tests/test_meta_builder_v3.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/a2d-lerobot-converter add h5_lerobot_converter/meta_builder_v3.py tests/test_meta_builder_v3.py
git -C /root/codes/a2d-lerobot-converter commit -m "feat: add MetaBuilderV3 for tasks.parquet, episodes parquet, v3 info.json"
```

---

## Task 3: `DatasetBuilderV3` and `convert_v3.py`

**Files:**
- Create: `h5_lerobot_converter/convert_v3.py`
- Test: `tests/test_convert_v3.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_convert_v3.py`:

```python
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.convert_v3 import DatasetBuilderV3


@pytest.fixture
def two_episodes(tmp_path, mini_h5) -> tuple:
    ep0 = tmp_path / "input" / "ep0.h5"
    ep1 = tmp_path / "input" / "ep1.h5"
    ep0.parent.mkdir()
    shutil.copy(mini_h5, ep0)
    shutil.copy(mini_h5, ep1)
    return [ep0, ep1], tmp_path / "output"


def test_output_structure_v3(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup", fps=30.0)

    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "tasks.parquet").exists()
    assert (out / "meta" / "stats.json").exists()
    assert (out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert (out / "data" / "chunk-000" / "file-000.parquet").exists()
    for cam in ["observation.images.hand_left", "observation.images.hand_right",
                "observation.images.head", "observation.images.head_depth"]:
        assert (out / "videos" / cam / "chunk-000" / "file-000.mp4").exists()


def test_info_json_v3(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20


def test_tasks_parquet(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    df = pd.read_parquet(out / "meta" / "tasks.parquet")
    assert df["task"].tolist() == ["grab cup"]
    assert df["task_index"].tolist() == [0]


def test_episodes_parquet_schema(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    df = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(df) == 2
    required = {
        "episode_index", "tasks", "length",
        "dataset_from_index", "dataset_to_index",
        "data/chunk_index", "data/file_index",
    }
    assert required <= set(df.columns)
    assert df["dataset_from_index"].tolist() == [0, 10]
    assert df["dataset_to_index"].tolist() == [10, 20]


def test_data_parquet_global_index(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    df = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
    assert len(df) == 20
    assert df["index"].tolist() == list(range(20))
    assert df["next.done"].iloc[9] == True   # last frame of ep0
    assert df["next.done"].iloc[10] == False  # first frame of ep1
    assert df["next.done"].iloc[19] == True   # last frame of ep1


def test_stats_no_nan(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    stats = json.loads((out / "meta" / "stats.json").read_text())
    for key in ["observation.state", "action"]:
        for field in ["mean", "std", "min", "max"]:
            vals = stats[key][field]
            assert not any(v != v or abs(v) == float("inf") for v in vals)


def test_video_files_nonempty(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="grab cup")
    for cam in ["observation.images.hand_left", "observation.images.hand_right",
                "observation.images.head", "observation.images.head_depth"]:
        vid = out / "videos" / cam / "chunk-000" / "file-000.mp4"
        assert vid.stat().st_size > 1000


def test_multi_batch_append(two_episodes, mini_h5, tmp_path):
    h5_files, out = two_episodes
    DatasetBuilderV3(out).build(h5_files, task="task A")

    ep2 = tmp_path / "batch2" / "ep2.h5"
    ep2.parent.mkdir()
    shutil.copy(mini_h5, ep2)
    DatasetBuilderV3(out).build([ep2], task="task B")

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_tasks"] == 2

    df_tasks = pd.read_parquet(out / "meta" / "tasks.parquet")
    assert df_tasks["task"].tolist() == ["task A", "task B"]

    df_ep = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(df_ep) == 3
    assert df_ep["dataset_from_index"].tolist() == [0, 10, 20]


def test_size_limit_creates_second_file(tmp_path, mini_h5):
    ep0 = tmp_path / "input" / "ep0.h5"
    ep1 = tmp_path / "input" / "ep1.h5"
    ep0.parent.mkdir()
    shutil.copy(mini_h5, ep0)
    shutil.copy(mini_h5, ep1)
    out = tmp_path / "output"

    # Limit so small every episode gets its own file
    DatasetBuilderV3(out, data_file_size_mb=1e-9, video_file_size_mb=1e-9).build(
        [ep0, ep1], task="task", fps=30.0
    )
    assert (out / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (out / "data" / "chunk-000" / "file-001.parquet").exists()
    for cam in ["observation.images.hand_left", "observation.images.hand_right",
                "observation.images.head", "observation.images.head_depth"]:
        assert (out / "videos" / cam / "chunk-000" / "file-000.mp4").exists()
        assert (out / "videos" / cam / "chunk-000" / "file-001.mp4").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/usr/bin/python3 -m pytest tests/test_convert_v3.py -v
```

Expected: `ModuleNotFoundError: No module named 'h5_lerobot_converter.convert_v3'`

- [ ] **Step 3: Implement `convert_v3.py`**

Create `h5_lerobot_converter/convert_v3.py`:

```python
import argparse
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from h5_lerobot_converter.reader import HDF5Reader
from h5_lerobot_converter.video_writer import VideoWriter
from h5_lerobot_converter.parquet_writer_v3 import ParquetWriterV3
from h5_lerobot_converter.meta_builder_v3 import MetaBuilderV3, CAMERA_KEYS


def _detect_image_shapes(data) -> dict:
    import cv2
    shapes = {}
    for key, frames in data.images.items():
        img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot decode image for {key}")
        shapes[key] = img.shape if img.ndim == 3 else (*img.shape, 1)
    return shapes


def _ffmpeg_concat(input_paths: list, output_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in input_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_file = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", str(output_path)],
            check=True, capture_output=True,
        )
    finally:
        Path(list_file).unlink(missing_ok=True)


def _flush_video_group(
    pending: dict,
    output_dir: Path,
    chunk_index: int,
    file_index: int,
) -> None:
    for cam_key, temps in pending.items():
        if not temps:
            continue
        out = (output_dir / "videos" / cam_key
               / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(temps) == 1:
            temps[0].rename(out)
        else:
            _ffmpeg_concat(temps, out)
            for p in temps:
                p.unlink(missing_ok=True)


class DatasetBuilderV3:
    def __init__(
        self,
        output_dir: Path,
        chunk_size: int = 1000,
        data_file_size_mb: float = 100.0,
        video_file_size_mb: float = 200.0,
    ):
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.data_file_size_mb = data_file_size_mb
        self.video_file_size_mb = video_file_size_mb
        self.reader = HDF5Reader()
        self.video_writer = VideoWriter()

    def _load_existing_state(self):
        tasks = []
        episodes = []
        global_frame_index = 0

        tasks_path = self.output_dir / "meta" / "tasks.parquet"
        episodes_dir = self.output_dir / "meta" / "episodes"

        if tasks_path.exists():
            df = pd.read_parquet(tasks_path)
            tasks = df.sort_values("task_index")["task"].tolist()

        if episodes_dir.exists():
            for ep_file in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
                df = pd.read_parquet(ep_file)
                for _, row in df.iterrows():
                    ep = row.to_dict()
                    episodes.append(ep)
                    global_frame_index += int(ep["length"])

        return tasks, episodes, global_frame_index

    def build(self, h5_files: list, task: str, fps: float = 30.0) -> None:
        tasks, existing_episodes, global_frame_index = self._load_existing_state()

        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)

        start_ep_idx = len(existing_episodes)

        parquet_writer = ParquetWriterV3(
            self.output_dir / "data",
            chunk_size=self.chunk_size,
            max_size_mb=self.data_file_size_mb,
        )

        tmp_dir = self.output_dir / "_tmp_videos"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Video file tracking — independent of data file tracking
        vid_chunk_index = 0
        vid_file_index = 0
        vid_file_bytes = 0
        vid_max_bytes = self.video_file_size_mb * 1024 * 1024
        pending_video_temps: dict = defaultdict(list)  # cam_key → [Path, ...]

        image_shapes = None
        new_episodes = []

        for i, h5_path in enumerate(h5_files):
            ep_idx = start_ep_idx + i
            print(f"[{i+1}/{len(h5_files)}] {h5_path.name} → episode_{ep_idx:06d}")

            data = self.reader.read(h5_path)
            if image_shapes is None:
                image_shapes = _detect_image_shapes(data)

            data_location = parquet_writer.write_episode(
                data, ep_idx, global_frame_index, task_index
            )

            # Flush video group if size limit exceeded (10 KB per frame, rough estimate)
            ep_video_bytes = data.n_frames * 10 * 1024
            if pending_video_temps[CAMERA_KEYS[0]] and \
               vid_file_bytes + ep_video_bytes > vid_max_bytes:
                _flush_video_group(pending_video_temps, self.output_dir,
                                   vid_chunk_index, vid_file_index)
                pending_video_temps = defaultdict(list)
                vid_file_index += 1
                vid_file_bytes = 0

            # Write episode videos to temp files
            for cam_key in CAMERA_KEYS:
                temp_path = tmp_dir / f"ep{ep_idx:06d}_{cam_key.replace('.', '_')}.mp4"
                self.video_writer.write(data.images[cam_key], temp_path, fps)
                pending_video_temps[cam_key].append(temp_path)
            vid_file_bytes += ep_video_bytes

            new_episodes.append({
                "episode_index": ep_idx,
                "tasks": [task],
                "length": data.n_frames,
                "dataset_from_index": global_frame_index,
                "dataset_to_index": global_frame_index + data.n_frames,
                "data/chunk_index": data_location["chunk_index"],
                "data/file_index": data_location["file_index"],
                **{f"videos/{cam}/chunk_index": vid_chunk_index for cam in CAMERA_KEYS},
                **{f"videos/{cam}/file_index": vid_file_index for cam in CAMERA_KEYS},
            })
            global_frame_index += data.n_frames

        parquet_writer.flush()
        if any(pending_video_temps.values()):
            _flush_video_group(pending_video_temps, self.output_dir,
                               vid_chunk_index, vid_file_index)

        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        all_episodes = existing_episodes + new_episodes
        total_frames = sum(ep["length"] for ep in all_episodes)
        total_episodes = len(all_episodes)

        meta = MetaBuilderV3(
            self.output_dir, fps=fps,
            chunk_size=self.chunk_size,
            data_file_size_mb=self.data_file_size_mb,
            video_file_size_mb=self.video_file_size_mb,
        )
        meta.write_tasks(tasks)
        meta.write_episodes(all_episodes)
        meta.write_info(total_episodes, total_frames, tasks, image_shapes)
        meta.write_stats(all_episodes)

        print(f"\nDone. {total_episodes} episodes / {total_frames} frames → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert A2D HDF5 dataset to LeRobot v3.0 format"
    )
    parser.add_argument("--input_dir",          required=True, type=Path)
    parser.add_argument("--output_dir",         required=True, type=Path)
    parser.add_argument("--task",               required=True, type=str)
    parser.add_argument("--fps",                default=30.0,  type=float)
    parser.add_argument("--chunk_size",         default=1000,  type=int)
    parser.add_argument("--data_file_size_mb",  default=100.0, type=float)
    parser.add_argument("--video_file_size_mb", default=200.0, type=float)
    args = parser.parse_args()

    h5_files = sorted(args.input_dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {args.input_dir}")

    print(f"Found {len(h5_files)} episode(s) in {args.input_dir}")
    DatasetBuilderV3(
        args.output_dir,
        chunk_size=args.chunk_size,
        data_file_size_mb=args.data_file_size_mb,
        video_file_size_mb=args.video_file_size_mb,
    ).build(h5_files, args.task, args.fps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/usr/bin/python3 -m pytest tests/test_convert_v3.py -v
```

Expected: 9 tests PASSED

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
/usr/bin/python3 -m pytest tests/ -v
```

Expected: All 35 tests PASSED (21 existing + 5 parquet_writer_v3 + 4 meta_builder_v3 + 9 convert_v3 = 39 total... adjust count based on actual results, all should pass)

- [ ] **Step 6: Commit**

```bash
git -C /root/codes/a2d-lerobot-converter add h5_lerobot_converter/convert_v3.py tests/test_convert_v3.py
git -C /root/codes/a2d-lerobot-converter commit -m "feat: add DatasetBuilderV3 and convert_v3 CLI for LeRobot v3.0 format"
```

---

## Task 4: Commit spec doc and verify CLI works end-to-end

**Files:**
- Modify: (verify only)

- [ ] **Step 1: Commit spec document**

```bash
git -C /root/codes/a2d-lerobot-converter add docs/
git -C /root/codes/a2d-lerobot-converter commit -m "docs: add LeRobot v3 design spec and implementation plan"
```

- [ ] **Step 2: Smoke test with real data (if available)**

```bash
/usr/bin/python3 -m h5_lerobot_converter.convert_v3 \
  --input_dir /root/datasets/guodi/a2d \
  --output_dir /tmp/lerobot_v3_test \
  --task "robot manipulation task" \
  --fps 30.0
```

Expected: `Done. N episodes / M frames → /tmp/lerobot_v3_test`

- [ ] **Step 3: Verify output structure**

```bash
find /tmp/lerobot_v3_test -type f | sort
```

Expected output includes:
```
/tmp/lerobot_v3_test/meta/info.json
/tmp/lerobot_v3_test/meta/tasks.parquet
/tmp/lerobot_v3_test/meta/stats.json
/tmp/lerobot_v3_test/meta/episodes/chunk-000/file-000.parquet
/tmp/lerobot_v3_test/data/chunk-000/file-000.parquet
/tmp/lerobot_v3_test/videos/observation.images.hand_left/chunk-000/file-000.mp4
...
```

- [ ] **Step 4: Check `codebase_version` in info.json**

```bash
python3 -c "import json; d=json.load(open('/tmp/lerobot_v3_test/meta/info.json')); print(d['codebase_version'], d['total_episodes'], d['total_frames'])"
```

Expected: `v3.0 <N> <M>`
