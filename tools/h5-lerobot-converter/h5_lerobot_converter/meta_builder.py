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
