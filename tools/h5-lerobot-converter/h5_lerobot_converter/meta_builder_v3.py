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
        robot_type: str = "a2d",
        state_names: list | None = None,
        action_names: list | None = None,
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

    def write_stats(self, episodes: list) -> None:
        _VECTOR_COLS  = ["observation.state", "action"]
        _SCALAR_COLS  = ["timestamp", "frame_index", "episode_index",
                         "index", "task_index", "next.done"]
        _ALL_COLS     = _VECTOR_COLS + _SCALAR_COLS

        seen: set = set()
        accum: dict = {col: [] for col in _ALL_COLS}

        for ep in episodes:
            key = (ep["data/chunk_index"], ep["data/file_index"])
            if key in seen:
                continue
            seen.add(key)
            ci, fi = key
            pq = self.output_dir / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
            df = pd.read_parquet(pq, columns=_ALL_COLS)
            for col in _VECTOR_COLS:
                accum[col].extend(df[col].tolist())
            for col in _SCALAR_COLS:
                accum[col].extend(df[col].astype(np.float32).tolist())

        def _stats(arr: np.ndarray) -> dict:
            n = arr.shape[0]
            return {
                "mean":  arr.mean(axis=0).tolist(),
                "std":   arr.std(axis=0).tolist(),
                "min":   arr.min(axis=0).tolist(),
                "max":   arr.max(axis=0).tolist(),
                "count": [int(n)],
                "q01":   np.percentile(arr,  1, axis=0).tolist(),
                "q10":   np.percentile(arr, 10, axis=0).tolist(),
                "q50":   np.percentile(arr, 50, axis=0).tolist(),
                "q90":   np.percentile(arr, 90, axis=0).tolist(),
                "q99":   np.percentile(arr, 99, axis=0).tolist(),
            }

        stats: dict = {}
        for col in _VECTOR_COLS:
            stats[col] = _stats(np.array(accum[col], dtype=np.float32))
        for col in _SCALAR_COLS:
            stats[col] = _stats(np.array(accum[col], dtype=np.float32).reshape(-1, 1))

        # Add placeholder stats for video/image features so that factory.py can
        # overwrite them with ImageNet stats without KeyError.
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as _f:
                _info = json.load(_f)
            n_frames = stats["observation.state"]["count"][0]
            # LeRobot _validate_stat_value hardcodes shape (3,1,1) for ALL "image"
            # features regardless of actual channel count. factory.py also writes
            # (3,1,1) ImageNet stats to every camera key. Always use 3 channels.
            _mid   = [[[0.5]], [[0.5]], [[0.5]]]   # (3,1,1)
            _zeros = [[[0.0]], [[0.0]], [[0.0]]]
            _ones  = [[[1.0]], [[1.0]], [[1.0]]]
            for feat_name, feat in _info.get("features", {}).items():
                if feat["dtype"] in ("video", "image"):
                    stats[feat_name] = {
                        "mean": _mid, "std": _mid,
                        "min": _zeros, "max": _ones,
                        "count": [n_frames],
                        "q01": _mid, "q10": _mid, "q50": _mid, "q90": _mid, "q99": _mid,
                    }

        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
