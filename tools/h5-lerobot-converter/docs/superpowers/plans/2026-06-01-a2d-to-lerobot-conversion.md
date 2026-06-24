# A2D → LeRobot 3.0 转换脚本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将智元 A2D 机器人的 HDF5 数据集批量转换为 LeRobot 3.0 格式（parquet + mp4 + meta）

**Architecture:** 四个职责单一的模块（reader/video_writer/parquet_writer/meta_builder）+ 协调层 DatasetBuilder + CLI 入口。支持多批次追加（不同 task 多次运行到同一 output_dir）。TDD 驱动开发，每个模块先写失败测试再实现。

**Tech Stack:** Python 3.10, h5py, numpy, pandas, pyarrow, ffmpeg (subprocess pipe), cv2 (test fixture)

---

## 文件结构

```
/root/codes/playground/
  h5_lerobot_converter/
    __init__.py
    reader.py          # HDF5Reader: 读取单个 .h5 → EpisodeData
    video_writer.py    # VideoWriter: jpg bytes 列表 → mp4 (ffmpeg pipe)
    parquet_writer.py  # ParquetWriter: EpisodeData → episode parquet
    meta_builder.py    # MetaBuilder: 写 tasks/episodes/info/stats json
    convert.py         # DatasetBuilder + argparse CLI 入口
  tests/
    conftest.py        # 生成 mini HDF5 fixture (10帧)
    test_reader.py
    test_video_writer.py
    test_parquet_writer.py
    test_meta_builder.py
    test_integration.py
```

---

## Task 1: 项目脚手架 + 测试 Fixture

**Files:**
- Create: `h5_lerobot_converter/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 写 conftest.py（生成 mini HDF5 fixture）**

```python
# tests/conftest.py
import io
import json
import numpy as np
import pytest
import h5py
import cv2


def make_jpeg_bytes(h: int = 32, w: int = 48) -> bytes:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


@pytest.fixture
def mini_h5(tmp_path) -> "pathlib.Path":
    """10帧的最小合法 A2D HDF5 文件。"""
    n = 10
    path = tmp_path / "test_ep.h5"

    with h5py.File(path, "w") as f:
        # cameras — variable-length bytes
        vlen = h5py.vlen_dtype(np.uint8)
        for cam_path in [
            "cameras/hand_left/color/data",
            "cameras/hand_right/color/data",
            "cameras/head/color/data",
            "cameras/head/depth/data",
        ]:
            ds = f.create_dataset(cam_path, (n,), dtype=vlen)
            for i in range(n):
                raw = make_jpeg_bytes()
                ds[i] = np.frombuffer(raw, dtype=np.uint8)

        # joints/state
        f.create_dataset("joints/state/arm/position",      data=np.random.rand(n, 14))
        f.create_dataset("joints/state/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("joints/state/head/position",     data=np.random.rand(n, 2))
        f.create_dataset("joints/state/waist/position",    data=np.random.rand(n, 2))
        f.create_dataset("joints/state/robot/position",    data=np.random.rand(n, 3))
        f.create_dataset("joints/state/robot/orientation", data=np.random.rand(n, 4))

        # joints/action
        f.create_dataset("joints/action/arm/position",      data=np.random.rand(n, 14))
        f.create_dataset("joints/action/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("joints/action/head/position",     data=np.random.rand(n, 2))
        f.create_dataset("joints/action/waist/position",    data=np.random.rand(n, 2))
        f.create_dataset("joints/action/robot/position",    data=np.random.rand(n, 3))
        f.create_dataset("joints/action/robot/velocity",    data=np.random.rand(n, 2))

        # timestamp: nanoseconds at 30fps
        f.create_dataset("timestamp", data=np.arange(n, dtype=np.int64) * 33_333_333)

        meta = {"ver": "2.1.0", "time_align_info": {"fps": 30.0, "frame_count": n}}
        f.create_dataset("metadata.json", data=json.dumps(meta))

    return path
```

- [ ] **Step 2: 创建空 `__init__.py`**

```python
# h5_lerobot_converter/__init__.py
```

- [ ] **Step 3: 验证 fixture 可正常加载**

```bash
cd /root/codes/playground
python3 -m pytest tests/conftest.py --collect-only
```
Expected: `0 errors`, fixture `mini_h5` 可见

- [ ] **Step 4: Commit**

```bash
git -C /root/codes/playground init 2>/dev/null || true
git -C /root/codes/playground add h5_lerobot_converter/__init__.py tests/conftest.py
git -C /root/codes/playground commit -m "chore: project scaffold and test fixture"
```

---

## Task 2: HDF5Reader

**Files:**
- Create: `h5_lerobot_converter/reader.py`
- Create: `tests/test_reader.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_reader.py
import numpy as np
from h5_lerobot_converter.reader import HDF5Reader


def test_read_state_shape(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.state.shape == (10, 27), f"got {data.state.shape}"
    assert data.state.dtype == np.float32


def test_read_action_shape(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.action.shape == (10, 25), f"got {data.action.shape}"
    assert data.action.dtype == np.float32


def test_read_timestamps_seconds(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.timestamps.shape == (10,)
    # first timestamp = 0, second ≈ 0.0333s
    assert abs(data.timestamps[0]) < 1e-6
    assert abs(data.timestamps[1] - 1 / 30) < 1e-4


def test_read_images_four_cameras(mini_h5):
    data = HDF5Reader().read(mini_h5)
    expected_keys = {
        "observation.images.hand_left",
        "observation.images.hand_right",
        "observation.images.head",
        "observation.images.head_depth",
    }
    assert set(data.images.keys()) == expected_keys
    for key, frames in data.images.items():
        assert len(frames) == 10, f"{key}: expected 10 frames"
        assert isinstance(frames[0], bytes), f"{key}: frame should be bytes"
        assert len(frames[0]) > 100, f"{key}: frame too small"
```

- [ ] **Step 2: 确认测试失败**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_reader.py -v
```
Expected: `ImportError` 或 `ModuleNotFoundError`

- [ ] **Step 3: 实现 reader.py**

```python
# h5_lerobot_converter/reader.py
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np


@dataclass
class EpisodeData:
    state: np.ndarray        # (N, 27) float32
    action: np.ndarray       # (N, 25) float32
    timestamps: np.ndarray   # (N,)    float32, seconds
    images: dict             # str -> list[bytes], N jpeg frames
    n_frames: int


class HDF5Reader:
    _STATE_PATHS = [
        "joints/state/arm/position",       # 14
        "joints/state/effector/position",  # 2
        "joints/state/head/position",      # 2
        "joints/state/waist/position",     # 2
        "joints/state/robot/position",     # 3
        "joints/state/robot/orientation",  # 4  → total 27
    ]
    _ACTION_PATHS = [
        "joints/action/arm/position",       # 14
        "joints/action/effector/position",  # 2
        "joints/action/head/position",      # 2
        "joints/action/waist/position",     # 2
        "joints/action/robot/position",     # 3
        "joints/action/robot/velocity",     # 2  → total 25
    ]
    _CAMERAS = {
        "observation.images.hand_left":  "cameras/hand_left/color/data",
        "observation.images.hand_right": "cameras/hand_right/color/data",
        "observation.images.head":       "cameras/head/color/data",
        "observation.images.head_depth": "cameras/head/depth/data",
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
            state=state,
            action=action,
            timestamps=timestamps,
            images=images,
            n_frames=n,
        )
```

- [ ] **Step 4: 确认测试通过**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_reader.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/playground add h5_lerobot_converter/reader.py tests/test_reader.py
git -C /root/codes/playground commit -m "feat: HDF5Reader reads state/action/timestamps/images"
```

---

## Task 3: VideoWriter

**Files:**
- Create: `h5_lerobot_converter/video_writer.py`
- Create: `tests/test_video_writer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_video_writer.py
import subprocess
import numpy as np
import cv2
import pytest
from h5_lerobot_converter.video_writer import VideoWriter


def make_frames(n: int = 5, h: int = 32, w: int = 48) -> list:
    frames = []
    for _ in range(n):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        frames.append(buf.tobytes())
    return frames


def test_video_created(tmp_path):
    out = tmp_path / "test.mp4"
    VideoWriter().write(make_frames(), out, fps=5.0)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_video_duration(tmp_path):
    out = tmp_path / "test.mp4"
    VideoWriter().write(make_frames(n=30), out, fps=30.0)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    assert abs(duration - 1.0) < 0.2, f"expected ~1s, got {duration}s"


def test_parent_dirs_created(tmp_path):
    out = tmp_path / "deep" / "nested" / "ep.mp4"
    VideoWriter().write(make_frames(), out, fps=5.0)
    assert out.exists()
```

- [ ] **Step 2: 确认测试失败**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_video_writer.py -v
```
Expected: `ImportError`

- [ ] **Step 3: 实现 video_writer.py**

```python
# h5_lerobot_converter/video_writer.py
import subprocess
from pathlib import Path


class VideoWriter:
    def write(self, frames: list, output_path: Path, fps: float = 30.0) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames:
            proc.stdin.write(frame)
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (returncode={proc.returncode}) for {output_path}")
```

- [ ] **Step 4: 确认测试通过**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_video_writer.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/playground add h5_lerobot_converter/video_writer.py tests/test_video_writer.py
git -C /root/codes/playground commit -m "feat: VideoWriter encodes jpg frames to mp4 via ffmpeg"
```

---

## Task 4: ParquetWriter

**Files:**
- Create: `h5_lerobot_converter/parquet_writer.py`
- Create: `tests/test_parquet_writer.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_parquet_writer.py
import numpy as np
import pandas as pd
import pytest
from h5_lerobot_converter.reader import HDF5Reader, EpisodeData
from h5_lerobot_converter.parquet_writer import ParquetWriter


def make_episode(n: int = 5) -> EpisodeData:
    return EpisodeData(
        state=np.random.rand(n, 27).astype(np.float32),
        action=np.random.rand(n, 25).astype(np.float32),
        timestamps=np.arange(n, dtype=np.float32) / 30.0,
        images={},
        n_frames=n,
    )


def test_parquet_schema(tmp_path):
    out = tmp_path / "chunk-000" / "episode_000000.parquet"
    ParquetWriter().write(make_episode(), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    expected_cols = {
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index", "next.done",
    }
    assert set(df.columns) == expected_cols


def test_parquet_row_count(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=7), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert len(df) == 7


def test_parquet_global_index(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=5), episode_index=2, global_start_index=100,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert df["episode_index"].unique().tolist() == [2]
    assert df["index"].tolist() == list(range(100, 105))


def test_next_done_last_frame(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=5), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert df["next.done"].tolist() == [False, False, False, False, True]


def test_state_values_preserved(tmp_path):
    ep = make_episode(n=3)
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(ep, episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    recovered = np.array(df["observation.state"].tolist(), dtype=np.float32)
    np.testing.assert_allclose(recovered, ep.state, rtol=1e-5)
```

- [ ] **Step 2: 确认测试失败**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_parquet_writer.py -v
```
Expected: `ImportError`

- [ ] **Step 3: 实现 parquet_writer.py**

```python
# h5_lerobot_converter/parquet_writer.py
from pathlib import Path
import numpy as np
import pandas as pd
from h5_lerobot_converter.reader import EpisodeData


class ParquetWriter:
    def write(
        self,
        data: EpisodeData,
        episode_index: int,
        global_start_index: int,
        task_index: int,
        output_path: Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = data.n_frames

        done = [False] * (n - 1) + [True]
        df = pd.DataFrame({
            "observation.state": [data.state[i].tolist() for i in range(n)],
            "action":            [data.action[i].tolist() for i in range(n)],
            "timestamp":         data.timestamps.tolist(),
            "frame_index":       list(range(n)),
            "episode_index":     [episode_index] * n,
            "index":             list(range(global_start_index, global_start_index + n)),
            "task_index":        [task_index] * n,
            "next.done":         done,
        })
        df.to_parquet(output_path, index=False)
```

- [ ] **Step 4: 确认测试通过**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_parquet_writer.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/playground add h5_lerobot_converter/parquet_writer.py tests/test_parquet_writer.py
git -C /root/codes/playground commit -m "feat: ParquetWriter writes LeRobot 3.0 episode parquet"
```

---

## Task 5: MetaBuilder

**Files:**
- Create: `h5_lerobot_converter/meta_builder.py`
- Create: `tests/test_meta_builder.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_meta_builder.py
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.meta_builder import MetaBuilder, STATE_NAMES, ACTION_NAMES
from h5_lerobot_converter.parquet_writer import ParquetWriter
from h5_lerobot_converter.reader import EpisodeData


def make_parquet(tmp_path, ep_idx: int, global_start: int, n: int = 5) -> Path:
    ep = EpisodeData(
        state=np.ones((n, 27), dtype=np.float32) * ep_idx,
        action=np.ones((n, 25), dtype=np.float32) * ep_idx,
        timestamps=np.arange(n, dtype=np.float32) / 30.0,
        images={}, n_frames=n,
    )
    out = tmp_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    ParquetWriter().write(ep, ep_idx, global_start, task_index=0, output_path=out)
    return out


def test_write_tasks(tmp_path):
    mb = MetaBuilder(tmp_path, fps=30.0)
    mb.write_tasks(["task one", "task two"])
    lines = (tmp_path / "meta" / "tasks.jsonl").read_text().strip().split("\n")
    records = [json.loads(l) for l in lines]
    assert records[0] == {"task_index": 0, "task": "task one"}
    assert records[1] == {"task_index": 1, "task": "task two"}


def test_write_episodes(tmp_path):
    mb = MetaBuilder(tmp_path, fps=30.0)
    episodes = [
        {"episode_index": 0, "task": "grab cup", "length": 10},
        {"episode_index": 1, "task": "grab cup", "length": 8},
    ]
    mb.write_episodes(episodes)
    lines = (tmp_path / "meta" / "episodes.jsonl").read_text().strip().split("\n")
    r0 = json.loads(lines[0])
    assert r0["episode_index"] == 0
    assert r0["tasks"] == ["grab cup"]
    assert r0["length"] == 10


def test_write_info_features(tmp_path):
    mb = MetaBuilder(tmp_path, fps=30.0)
    mb.write_info(
        total_episodes=2, total_frames=10, tasks=["t"],
        image_shapes={
            "observation.images.hand_left": (480, 848, 3),
            "observation.images.head_depth": (480, 848, 3),
        },
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["fps"] == 30.0
    assert info["total_episodes"] == 2
    assert info["features"]["observation.state"]["shape"] == [27]
    assert info["features"]["action"]["shape"] == [25]
    assert info["features"]["observation.images.hand_left"]["dtype"] == "video"
    assert info["features"]["observation.images.head_depth"]["info"]["video.is_depth_map"] is True


def test_state_action_names_length():
    assert len(STATE_NAMES) == 27
    assert len(ACTION_NAMES) == 25


def test_write_stats(tmp_path):
    make_parquet(tmp_path, ep_idx=0, global_start=0, n=5)
    make_parquet(tmp_path, ep_idx=1, global_start=5, n=5)
    mb = MetaBuilder(tmp_path, fps=30.0)
    mb.write_stats(tmp_path / "data", total_episodes=2)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    assert len(stats["observation.state"]["mean"]) == 27
    assert len(stats["action"]["mean"]) == 25
```

- [ ] **Step 2: 确认测试失败**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_meta_builder.py -v
```
Expected: `ImportError`

- [ ] **Step 3: 实现 meta_builder.py**

```python
# h5_lerobot_converter/meta_builder.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

STATE_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
    "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
    "left_gripper_joint1", "right_gripper_joint1",
    "joint_head_yaw", "joint_head_pitch",
    "joint_body_pitch", "joint_lift_body",
    "base_pos_x", "base_pos_y", "base_pos_z",
    "base_ori_x", "base_ori_y", "base_ori_z", "base_ori_w",
]

ACTION_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
    "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
    "left_gripper_joint1", "right_gripper_joint1",
    "joint_head_yaw", "joint_head_pitch",
    "joint_body_pitch", "joint_lift_body",
    "base_cmd_pos_x", "base_cmd_pos_y", "base_cmd_pos_z",
    "base_cmd_vel_x", "base_cmd_vel_y",
]


class MetaBuilder:
    def __init__(self, output_dir: Path, fps: float, chunk_size: int = 1000):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_size = chunk_size
        self.meta_dir = self.output_dir / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def write_tasks(self, tasks: list) -> None:
        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            for i, task in enumerate(tasks):
                f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    def write_episodes(self, episodes: list) -> None:
        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps({
                    "episode_index": ep["episode_index"],
                    "tasks": [ep["task"]],
                    "length": ep["length"],
                }) + "\n")

    def write_info(self, total_episodes: int, total_frames: int,
                   tasks: list, image_shapes: dict) -> None:
        n_chunks = max(1, (total_episodes + self.chunk_size - 1) // self.chunk_size)

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
            "codebase_version": "v2.0",
            "robot_type": "a2d",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_chunks": n_chunks,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def write_stats(self, data_dir: Path, total_episodes: int) -> None:
        all_states, all_actions = [], []
        for ep_idx in range(total_episodes):
            chunk = ep_idx // self.chunk_size
            pq_path = Path(data_dir) / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(pq_path, columns=["observation.state", "action"])
            all_states.extend(df["observation.state"].tolist())
            all_actions.extend(df["action"].tolist())

        states  = np.array(all_states,  dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)

        def _feat_stats(arr):
            return {
                "mean": arr.mean(axis=0).tolist(),
                "std":  arr.std(axis=0).tolist(),
                "min":  arr.min(axis=0).tolist(),
                "max":  arr.max(axis=0).tolist(),
            }

        stats = {
            "observation.state": _feat_stats(states),
            "action":            _feat_stats(actions),
        }
        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
```

- [ ] **Step 4: 确认测试通过**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_meta_builder.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git -C /root/codes/playground add h5_lerobot_converter/meta_builder.py tests/test_meta_builder.py
git -C /root/codes/playground commit -m "feat: MetaBuilder writes tasks/episodes/info/stats json"
```

---

## Task 6: DatasetBuilder + CLI（含多批次支持）

**Files:**
- Create: `h5_lerobot_converter/convert.py`

- [ ] **Step 1: 实现 convert.py**

```python
# h5_lerobot_converter/convert.py
import argparse
import json
from pathlib import Path

from h5_lerobot_converter.reader import HDF5Reader
from h5_lerobot_converter.video_writer import VideoWriter
from h5_lerobot_converter.parquet_writer import ParquetWriter
from h5_lerobot_converter.meta_builder import MetaBuilder

CAMERA_KEYS = [
    "observation.images.hand_left",
    "observation.images.hand_right",
    "observation.images.head",
    "observation.images.head_depth",
]
IMAGE_SHAPES = {k: (480, 848, 3) for k in CAMERA_KEYS}


class DatasetBuilder:
    def __init__(self, output_dir: Path, chunk_size: int = 1000):
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.reader = HDF5Reader()
        self.video_writer = VideoWriter()
        self.parquet_writer = ParquetWriter()

    def _load_existing_state(self):
        """读取已有 episodes.jsonl / tasks.jsonl，支持多批次追加。"""
        tasks = []
        episodes = []
        total_frames = 0

        tasks_path = self.output_dir / "meta" / "tasks.jsonl"
        episodes_path = self.output_dir / "meta" / "episodes.jsonl"

        if tasks_path.exists():
            tasks = [json.loads(l)["task"] for l in tasks_path.read_text().splitlines() if l]
        if episodes_path.exists():
            for line in episodes_path.read_text().splitlines():
                if line:
                    ep = json.loads(line)
                    # 归一化：jsonl 里存的是 "tasks" list，内部统一用 "task" string
                    episodes.append({
                        "episode_index": ep["episode_index"],
                        "task": ep["tasks"][0],
                        "length": ep["length"],
                    })
                    total_frames += ep["length"]
        return tasks, episodes, total_frames

    def build(self, h5_files: list, task: str, fps: float = 30.0) -> None:
        tasks, existing_episodes, global_frame_index = self._load_existing_state()

        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)

        start_ep_idx = len(existing_episodes)
        new_episodes = []

        for i, h5_path in enumerate(h5_files):
            ep_idx = start_ep_idx + i
            chunk = ep_idx // self.chunk_size
            print(f"[{i+1}/{len(h5_files)}] {h5_path.name} → episode_{ep_idx:06d}")

            data = self.reader.read(h5_path)

            pq_path = (self.output_dir / "data"
                       / f"chunk-{chunk:03d}"
                       / f"episode_{ep_idx:06d}.parquet")
            self.parquet_writer.write(data, ep_idx, global_frame_index, task_index, pq_path)

            for cam_key in CAMERA_KEYS:
                vid_path = (self.output_dir / "videos"
                            / f"chunk-{chunk:03d}"
                            / cam_key
                            / f"episode_{ep_idx:06d}.mp4")
                self.video_writer.write(data.images[cam_key], vid_path, fps)

            new_episodes.append({
                "episode_index": ep_idx,
                "task": task,
                "length": data.n_frames,
            })
            global_frame_index += data.n_frames

        all_episodes = existing_episodes + new_episodes
        total_frames = sum(ep["length"] for ep in all_episodes)
        total_episodes = len(all_episodes)

        meta = MetaBuilder(self.output_dir, fps=fps, chunk_size=self.chunk_size)
        meta.write_tasks(tasks)
        meta.write_episodes(all_episodes)
        meta.write_info(total_episodes, total_frames, tasks, IMAGE_SHAPES)
        meta.write_stats(self.output_dir / "data", total_episodes)

        print(f"\nDone. {total_episodes} episodes / {total_frames} frames → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert A2D HDF5 dataset to LeRobot 3.0 format")
    parser.add_argument("--input_dir",  required=True, type=Path, help="目录，含 *.h5 文件")
    parser.add_argument("--output_dir", required=True, type=Path, help="LeRobot 输出目录")
    parser.add_argument("--task",       required=True, type=str,  help="任务描述文字")
    parser.add_argument("--chunk_size", default=1000,  type=int,  help="每个 chunk 包含的 episode 数")
    parser.add_argument("--fps",        default=30.0,  type=float)
    args = parser.parse_args()

    h5_files = sorted(args.input_dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {args.input_dir}")

    print(f"Found {len(h5_files)} episode(s) in {args.input_dir}")
    DatasetBuilder(args.output_dir, args.chunk_size).build(h5_files, args.task, args.fps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认脚本可导入**

```bash
cd /root/codes/playground && python3 -c "from h5_lerobot_converter.convert import DatasetBuilder; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git -C /root/codes/playground add h5_lerobot_converter/convert.py
git -C /root/codes/playground commit -m "feat: DatasetBuilder + CLI with multi-batch append support"
```

---

## Task 7: 集成测试 + 真实数据冒烟测试

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 写集成测试（用 mini_h5 fixture 做 2 条 episode）**

```python
# tests/test_integration.py
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.convert import DatasetBuilder


@pytest.fixture
def two_episodes(tmp_path, mini_h5) -> tuple:
    """复制 mini_h5 两次，模拟两条 episode。"""
    ep0 = tmp_path / "input" / "ep0.h5"
    ep1 = tmp_path / "input" / "ep1.h5"
    ep0.parent.mkdir()
    shutil.copy(mini_h5, ep0)
    shutil.copy(mini_h5, ep1)
    return [ep0, ep1], tmp_path / "output"


def test_output_structure(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilder(out).build(h5_files, task="grab cup", fps=30.0)

    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "episodes.jsonl").exists()
    assert (out / "meta" / "tasks.jsonl").exists()
    assert (out / "meta" / "stats.json").exists()
    assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (out / "data" / "chunk-000" / "episode_000001.parquet").exists()
    for cam in ["observation.images.hand_left", "observation.images.hand_right",
                "observation.images.head", "observation.images.head_depth"]:
        assert (out / "videos" / "chunk-000" / cam / "episode_000000.mp4").exists()


def test_info_episode_count(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilder(out).build(h5_files, task="grab cup")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20  # 10 frames * 2


def test_parquet_global_index_continuous(two_episodes):
    h5_files, out = two_episodes
    DatasetBuilder(out).build(h5_files, task="grab cup")
    df0 = pd.read_parquet(out / "data" / "chunk-000" / "episode_000000.parquet")
    df1 = pd.read_parquet(out / "data" / "chunk-000" / "episode_000001.parquet")
    assert df0["index"].tolist() == list(range(0, 10))
    assert df1["index"].tolist() == list(range(10, 20))


def test_multi_batch_append(two_episodes, mini_h5, tmp_path):
    h5_files, out = two_episodes
    DatasetBuilder(out).build(h5_files, task="task A")

    # 第二批：不同 task
    ep2 = tmp_path / "batch2" / "ep2.h5"
    ep2.parent.mkdir()
    shutil.copy(mini_h5, ep2)
    DatasetBuilder(out).build([ep2], task="task B")

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_tasks"] == 2

    tasks = [(json.loads(l)["task"]) for l in
             (out / "meta" / "tasks.jsonl").read_text().splitlines() if l]
    assert tasks == ["task A", "task B"]
```

- [ ] **Step 2: 确认集成测试通过**

```bash
cd /root/codes/playground && python3 -m pytest tests/test_integration.py -v
```
Expected: `4 passed`（约 30-60 秒，视频编码较慢）

- [ ] **Step 3: 用真实数据冒烟测试**

```bash
cd /root/codes/playground && python3 -m h5_lerobot_converter.convert \
  --input_dir /root/datasets/guodi/a2d \
  --output_dir /tmp/lerobot_test \
  --task "real robot manipulation task"
```
Expected: 打印进度，生成 `/tmp/lerobot_test/meta/info.json`，`total_episodes: 1`

- [ ] **Step 4: 验证输出结构**

```bash
find /tmp/lerobot_test -type f | sort
python3 -c "
import json
info = json.load(open('/tmp/lerobot_test/meta/info.json'))
print('episodes:', info['total_episodes'])
print('frames:  ', info['total_frames'])
print('features:', list(info['features'].keys()))
"
```

- [ ] **Step 5: 全量测试**

```bash
cd /root/codes/playground && python3 -m pytest tests/ -v
```
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git -C /root/codes/playground add tests/test_integration.py
git -C /root/codes/playground commit -m "test: integration tests including multi-batch append"
```
