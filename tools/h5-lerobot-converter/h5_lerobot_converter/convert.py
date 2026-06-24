import argparse
import json
from pathlib import Path
import numpy as np

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


def _detect_image_shapes(data) -> dict:
    import cv2
    shapes = {}
    for key, frames in data.images.items():
        img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot decode image for {key}")
        shapes[key] = img.shape if img.ndim == 3 else (*img.shape, 1)
    return shapes


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
        image_shapes = None

        for i, h5_path in enumerate(h5_files):
            ep_idx = start_ep_idx + i
            chunk = ep_idx // self.chunk_size
            print(f"[{i+1}/{len(h5_files)}] {h5_path.name} → episode_{ep_idx:06d}")

            data = self.reader.read(h5_path)
            if image_shapes is None:
                image_shapes = _detect_image_shapes(data)

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
        meta.write_info(total_episodes, total_frames, tasks, image_shapes)
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
