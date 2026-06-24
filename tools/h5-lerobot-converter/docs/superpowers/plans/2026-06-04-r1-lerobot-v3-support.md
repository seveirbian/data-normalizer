# R1 → LeRobot v3 格式支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增星海图 R1 机器人 HDF5 数据转 LeRobot v3.0 格式的 CLI，复用现有 v3 写入管线。

**Architecture:** `MetaBuilderV3.write_info()` 新增可选参数支持任意 robot_type 和 state/action 名称；`r1_reader.py` 读取 R1 HDF5 返回 `EpisodeData`（26 维 state/action，5 路摄像头）；`r1_convert_v3.py` 薄封装 CLI，复用 `ParquetWriterV3`、`MetaBuilderV3`、`VideoWriter`。

**Tech Stack:** Python 3.10, h5py, pandas, pyarrow, numpy, opencv-python-headless, ffmpeg subprocess, pytest

**Run all tests with:** `/usr/bin/python3 -m pytest /root/codes/a2d-lerobot-converter/tests/ -v`

**Working directory:** `/root/codes/a2d-lerobot-converter`

---

## File Map

| 文件 | 操作 | 职责 |
|---|---|---|
| `h5_lerobot_converter/meta_builder_v3.py` | 修改 | `write_info()` 新增 `robot_type`、`state_names`、`action_names` 可选参数 |
| `h5_lerobot_converter/r1_reader.py` | 新建 | 读取 R1 HDF5，返回 EpisodeData（26 维，5 路摄像头） |
| `h5_lerobot_converter/r1_convert_v3.py` | 新建 | DatasetBuilderR1V3 + CLI |
| `tests/conftest.py` | 修改 | 新增 `r1_mini_h5` fixture |
| `tests/test_r1_reader.py` | 新建 | R1Reader 单元测试 |
| `tests/test_r1_convert_v3.py` | 新建 | R1 → LeRobot v3 集成测试 |

---

## Task 1: 修改 MetaBuilderV3 支持可变 robot_type 和 state/action 名称

**Files:**
- Modify: `h5_lerobot_converter/meta_builder_v3.py`
- Test: `tests/test_meta_builder_v3.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_meta_builder_v3.py` 末尾追加：

```python
def test_write_info_custom_robot_type(builder, tmp_path):
    custom_state = ["s" + str(i) for i in range(26)]
    custom_action = ["a" + str(i) for i in range(26)]
    builder.write_info(
        total_episodes=1, total_frames=10, tasks=["t"],
        image_shapes={},
        robot_type="r1",
        state_names=custom_state,
        action_names=custom_action,
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["robot_type"] == "r1"
    assert info["features"]["observation.state"]["shape"] == [26]
    assert info["features"]["action"]["shape"] == [26]
    assert info["features"]["observation.state"]["names"] == custom_state
```

- [ ] **Step 2: 运行测试确认失败**

```bash
/usr/bin/python3 -m pytest tests/test_meta_builder_v3.py::test_write_info_custom_robot_type -v
```

Expected: `TypeError: write_info() got an unexpected keyword argument 'robot_type'`

- [ ] **Step 3: 修改 `meta_builder_v3.py`**

将 `write_info` 签名和函数体修改为：

```python
def write_info(
    self,
    total_episodes: int,
    total_frames: int,
    tasks: list,
    image_shapes: dict,
    robot_type: str = "a2d",
    state_names: list = None,
    action_names: list = None,
) -> None:
    if state_names is None:
        state_names = STATE_NAMES
    if action_names is None:
        action_names = ACTION_NAMES
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": state_names},
        "action":            {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
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
        "robot_type": robot_type,
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
```

- [ ] **Step 4: 运行全部 MetaBuilderV3 测试**

```bash
/usr/bin/python3 -m pytest tests/test_meta_builder_v3.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add h5_lerobot_converter/meta_builder_v3.py tests/test_meta_builder_v3.py
git commit -m "feat: make MetaBuilderV3.write_info() support variable robot_type and state/action names"
```

---

## Task 2: R1Reader

**Files:**
- Modify: `tests/conftest.py`
- Create: `h5_lerobot_converter/r1_reader.py`
- Create: `tests/test_r1_reader.py`

- [ ] **Step 1: 在 conftest.py 末尾追加 r1_mini_h5 fixture**

```python
@pytest.fixture
def r1_mini_h5(tmp_path):
    """10帧最小合法 R1 HDF5 文件。"""
    n = 10
    path = tmp_path / "test_r1_ep.h5"

    with h5py.File(path, "w") as f:
        vlen = h5py.vlen_dtype(np.uint8)
        for cam_path in [
            "cameras/hand_left/color/data",
            "cameras/hand_left/depth/data",
            "cameras/hand_right/color/data",
            "cameras/hand_right/depth/data",
            "cameras/head/color/data",
        ]:
            ds = f.create_dataset(cam_path, (n,), dtype=vlen)
            for i in range(n):
                raw = make_jpeg_bytes()
                ds[i] = np.frombuffer(raw, dtype=np.uint8)

        f.create_dataset("joints/state/arm/position",       data=np.random.rand(n, 12))
        f.create_dataset("joints/state/arm/velocity",       data=np.random.rand(n, 12))
        f.create_dataset("joints/state/effector/position",  data=np.random.rand(n, 2))
        f.create_dataset("joints/action/arm/position",      data=np.random.rand(n, 12))
        f.create_dataset("joints/action/arm/velocity",      data=np.random.rand(n, 12))
        f.create_dataset("joints/action/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("timestamp", data=np.arange(n, dtype=np.int64) * 33_333_333)

        meta = {"ver": "2.1.0", "equipment_info": {"manufacturer": "星海图", "model": "R1"},
                "time_align_info": {"fps": 30.0, "frame_count": n}}
        f.create_dataset("metadata.json", data=json.dumps(meta))

    return path
```

- [ ] **Step 2: 写失败测试 `tests/test_r1_reader.py`**

```python
import numpy as np
from h5_lerobot_converter.r1_reader import R1Reader


def test_read_state_shape(r1_mini_h5):
    data = R1Reader().read(r1_mini_h5)
    assert data.state.shape == (10, 26), f"got {data.state.shape}"
    assert data.state.dtype == np.float32


def test_read_action_shape(r1_mini_h5):
    data = R1Reader().read(r1_mini_h5)
    assert data.action.shape == (10, 26), f"got {data.action.shape}"
    assert data.action.dtype == np.float32


def test_read_timestamps_seconds(r1_mini_h5):
    data = R1Reader().read(r1_mini_h5)
    assert data.timestamps.shape == (10,)
    assert abs(data.timestamps[0]) < 1e-6
    assert abs(data.timestamps[1] - 1 / 30) < 1e-4


def test_read_images_five_cameras(r1_mini_h5):
    data = R1Reader().read(r1_mini_h5)
    expected_keys = {
        "observation.images.hand_left",
        "observation.images.hand_left_depth",
        "observation.images.hand_right",
        "observation.images.hand_right_depth",
        "observation.images.head",
    }
    assert set(data.images.keys()) == expected_keys
    for key, frames in data.images.items():
        assert len(frames) == 10, f"{key}: expected 10 frames"
        assert isinstance(frames[0], bytes), f"{key}: frame should be bytes"
        assert len(frames[0]) > 100, f"{key}: frame too small"
```

- [ ] **Step 3: 运行测试确认失败**

```bash
/usr/bin/python3 -m pytest tests/test_r1_reader.py -v
```

Expected: `ModuleNotFoundError: No module named 'h5_lerobot_converter.r1_reader'`

- [ ] **Step 4: 实现 `h5_lerobot_converter/r1_reader.py`**

```python
from pathlib import Path
import h5py
import numpy as np

from h5_lerobot_converter.reader import EpisodeData

R1_STATE_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "left_arm_vel1", "left_arm_vel2", "left_arm_vel3",
    "left_arm_vel4", "left_arm_vel5", "left_arm_vel6",
    "right_arm_vel1", "right_arm_vel2", "right_arm_vel3",
    "right_arm_vel4", "right_arm_vel5", "right_arm_vel6",
    "left_gripper", "right_gripper",
]

R1_ACTION_NAMES = R1_STATE_NAMES[:]

CAMERA_KEYS_R1 = [
    "observation.images.hand_left",
    "observation.images.hand_left_depth",
    "observation.images.hand_right",
    "observation.images.hand_right_depth",
    "observation.images.head",
]


class R1Reader:
    _STATE_PATHS = [
        "joints/state/arm/position",       # 12
        "joints/state/arm/velocity",       # 12
        "joints/state/effector/position",  # 2  → total 26
    ]
    _ACTION_PATHS = [
        "joints/action/arm/position",       # 12
        "joints/action/arm/velocity",       # 12
        "joints/action/effector/position",  # 2  → total 26
    ]
    _CAMERAS = {
        "observation.images.hand_left":        "cameras/hand_left/color/data",
        "observation.images.hand_left_depth":  "cameras/hand_left/depth/data",
        "observation.images.hand_right":       "cameras/hand_right/color/data",
        "observation.images.hand_right_depth": "cameras/hand_right/depth/data",
        "observation.images.head":             "cameras/head/color/data",
    }

    def read(self, path: Path) -> EpisodeData:
        with h5py.File(path, "r") as f:
            state = np.concatenate(
                [f[p][:] for p in self._STATE_PATHS], axis=1
            ).astype(np.float32)
            action = np.concatenate(
                [f[p][:] for p in self._ACTION_PATHS], axis=1
            ).astype(np.float32)
            timestamps = (f["timestamp"][:] / 1e9).astype(np.float32)
            n = state.shape[0]
            images = {
                key: [bytes(f[hdf_path][i]) for i in range(n)]
                for key, hdf_path in self._CAMERAS.items()
            }
        return EpisodeData(
            state=state, action=action,
            timestamps=timestamps, images=images, n_frames=n,
        )
```

- [ ] **Step 5: 运行测试确认通过**

```bash
/usr/bin/python3 -m pytest tests/test_r1_reader.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py h5_lerobot_converter/r1_reader.py tests/test_r1_reader.py
git commit -m "feat: add R1Reader for R1 HDF5 format (26-dim state/action, 5 cameras)"
```

---

## Task 3: DatasetBuilderR1V3 + CLI

**Files:**
- Create: `h5_lerobot_converter/r1_convert_v3.py`
- Create: `tests/test_r1_convert_v3.py`

- [ ] **Step 1: 写失败集成测试 `tests/test_r1_convert_v3.py`**

```python
import json
import shutil
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.r1_convert_v3 import DatasetBuilderR1V3


@pytest.fixture
def two_r1_episodes(tmp_path, r1_mini_h5):
    ep0 = tmp_path / "input" / "ep0.h5"
    ep1 = tmp_path / "input" / "ep1.h5"
    ep0.parent.mkdir()
    shutil.copy(r1_mini_h5, ep0)
    shutil.copy(r1_mini_h5, ep1)
    return [ep0, ep1], tmp_path / "output"


def test_output_structure_r1(two_r1_episodes):
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="grab cup", fps=30.0)

    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "tasks.parquet").exists()
    assert (out / "meta" / "stats.json").exists()
    assert (out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert (out / "data" / "chunk-000" / "file-000.parquet").exists()
    for cam in [
        "observation.images.hand_left",
        "observation.images.hand_left_depth",
        "observation.images.hand_right",
        "observation.images.hand_right_depth",
        "observation.images.head",
    ]:
        assert (out / "videos" / cam / "chunk-000" / "file-000.mp4").exists()


def test_info_json_r1(two_r1_episodes):
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="grab cup")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "r1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20
    assert info["features"]["observation.state"]["shape"] == [26]
    assert info["features"]["action"]["shape"] == [26]


def test_data_parquet_schema_r1(two_r1_episodes):
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="grab cup")
    df = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
    assert len(df) == 20
    assert df["index"].tolist() == list(range(20))
    assert len(df["observation.state"].iloc[0]) == 26
    assert len(df["action"].iloc[0]) == 26
    assert df["next.done"].iloc[9] == True
    assert df["next.done"].iloc[19] == True


def test_stats_no_nan_r1(two_r1_episodes):
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="grab cup")
    stats = json.loads((out / "meta" / "stats.json").read_text())
    for key in ["observation.state", "action"]:
        for field in ["mean", "std", "min", "max"]:
            vals = stats[key][field]
            assert not any(v != v or abs(v) == float("inf") for v in vals)


def test_multi_batch_append_r1(two_r1_episodes, r1_mini_h5, tmp_path):
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="task A")

    ep2 = tmp_path / "batch2" / "ep2.h5"
    ep2.parent.mkdir()
    shutil.copy(r1_mini_h5, ep2)
    DatasetBuilderR1V3(out).build([ep2], task="task B")

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_tasks"] == 2
```

- [ ] **Step 2: 运行测试确认失败**

```bash
/usr/bin/python3 -m pytest tests/test_r1_convert_v3.py -v
```

Expected: `ModuleNotFoundError: No module named 'h5_lerobot_converter.r1_convert_v3'`

- [ ] **Step 3: 实现 `h5_lerobot_converter/r1_convert_v3.py`**

```python
import argparse
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from h5_lerobot_converter.r1_reader import R1Reader, CAMERA_KEYS_R1, R1_STATE_NAMES, R1_ACTION_NAMES
from h5_lerobot_converter.video_writer import VideoWriter
from h5_lerobot_converter.parquet_writer_v3 import ParquetWriterV3
from h5_lerobot_converter.meta_builder_v3 import MetaBuilderV3


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


def _flush_video_group(pending: dict, output_dir: Path, chunk_index: int, file_index: int) -> None:
    for cam_key, temps in pending.items():
        if not temps:
            continue
        out = (output_dir / "videos" / cam_key
               / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(temps) == 1:
            Path(temps[0]).rename(out)
        else:
            _ffmpeg_concat(temps, out)
            for p in temps:
                Path(p).unlink(missing_ok=True)


class DatasetBuilderR1V3:
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
        self.reader = R1Reader()
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

        vid_chunk_index = 0
        vid_file_index = 0
        vid_file_bytes = 0
        vid_max_bytes = self.video_file_size_mb * 1024 * 1024
        pending_video_temps: dict = defaultdict(list)

        image_shapes = None
        new_episodes = []

        for i, h5_path in enumerate(h5_files):
            ep_idx = start_ep_idx + i
            print(f"[{i+1}/{len(h5_files)}] {Path(h5_path).name} → episode_{ep_idx:06d}")

            data = self.reader.read(h5_path)
            if image_shapes is None:
                image_shapes = _detect_image_shapes(data)

            data_location = parquet_writer.write_episode(
                data, ep_idx, global_frame_index, task_index
            )

            ep_video_bytes = data.n_frames * 10 * 1024
            if pending_video_temps[CAMERA_KEYS_R1[0]] and \
               vid_file_bytes + ep_video_bytes > vid_max_bytes:
                _flush_video_group(pending_video_temps, self.output_dir,
                                   vid_chunk_index, vid_file_index)
                pending_video_temps = defaultdict(list)
                vid_file_index += 1
                vid_file_bytes = 0

            for cam_key in CAMERA_KEYS_R1:
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
                **{f"videos/{cam}/chunk_index": vid_chunk_index for cam in CAMERA_KEYS_R1},
                **{f"videos/{cam}/file_index": vid_file_index for cam in CAMERA_KEYS_R1},
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
        meta.write_info(
            total_episodes, total_frames, tasks, image_shapes,
            robot_type="r1",
            state_names=R1_STATE_NAMES,
            action_names=R1_ACTION_NAMES,
        )
        meta.write_stats(all_episodes)

        print(f"\nDone. {total_episodes} episodes / {total_frames} frames → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert R1 HDF5 dataset to LeRobot v3.0 format"
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
    DatasetBuilderR1V3(
        args.output_dir,
        chunk_size=args.chunk_size,
        data_file_size_mb=args.data_file_size_mb,
        video_file_size_mb=args.video_file_size_mb,
    ).build(h5_files, args.task, args.fps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行集成测试**

```bash
/usr/bin/python3 -m pytest tests/test_r1_convert_v3.py -v
```

Expected: 5 passed

- [ ] **Step 5: 运行全量测试**

```bash
/usr/bin/python3 -m pytest tests/ -v
```

Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add h5_lerobot_converter/r1_convert_v3.py tests/test_r1_convert_v3.py
git commit -m "feat: add DatasetBuilderR1V3 and r1_convert_v3 CLI for R1 → LeRobot v3"
```

---

## Task 4: 更新 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 README 的"使用方法"下新增 R1 章节**

在 `### LeRobot v2.0 格式（旧版）` 小节之前插入：

```markdown
### 星海图 R1 → LeRobot v3.0 格式

```bash
python3 -m h5_lerobot_converter.r1_convert_v3 \
  --input_dir datasets/r1 \
  --output_dir /path/to/lerobot_dataset \
  --task "robot manipulation task" \
  --fps 30.0
```

参数与 `convert_v3` 完全相同。R1 的 `observation.state` 和 `action` 均为 26 维（arm_position 12 + arm_velocity 12 + gripper 2），摄像头 5 路（双手 RGB+深度 + 头部 RGB）。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add R1 converter usage to README"
```
