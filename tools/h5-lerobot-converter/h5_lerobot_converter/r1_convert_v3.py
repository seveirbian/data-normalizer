import argparse
import shutil
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

    def _next_data_file_indices(self) -> tuple[int, int]:
        data_dir = self.output_dir / "data"
        if not data_dir.exists():
            return 0, 0
        files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
        if not files:
            return 0, 0
        last = files[-1]
        chunk = int(last.parent.name.split("-")[1])
        file = int(last.stem.split("-")[1])
        return chunk, file + 1

    def _next_video_file_index(self) -> int:
        ref_cam = CAMERA_KEYS_R1[0]
        vid_dir = self.output_dir / "videos" / ref_cam / "chunk-000"
        if not vid_dir.exists():
            return 0
        files = sorted(vid_dir.glob("file-*.mp4"))
        if not files:
            return 0
        return int(files[-1].stem.split("-")[1]) + 1

    def build(self, h5_files: list, task: str, fps: float = 30.0) -> None:
        tasks, existing_episodes, global_frame_index = self._load_existing_state()

        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)

        start_ep_idx = len(existing_episodes)

        start_chunk, start_file = self._next_data_file_indices()
        parquet_writer = ParquetWriterV3(
            self.output_dir / "data",
            chunk_size=self.chunk_size,
            max_size_mb=self.data_file_size_mb,
            start_chunk_index=start_chunk,
            start_file_index=start_file,
        )

        tmp_dir = self.output_dir / "_tmp_videos"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        vid_chunk_index = 0
        vid_file_index = self._next_video_file_index()
        vid_file_bytes = 0
        vid_max_bytes = self.video_file_size_mb * 1024 * 1024
        vid_cumulative_s: float = 0.0  # cumulative seconds written to current video file
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

            # rough estimate: ~10 KB per frame (empirical; actual varies with resolution/quality)
            ep_video_bytes = data.n_frames * 10 * 1024
            if any(pending_video_temps.values()) and \
               vid_file_bytes + ep_video_bytes > vid_max_bytes:
                _flush_video_group(pending_video_temps, self.output_dir,
                                   vid_chunk_index, vid_file_index)
                pending_video_temps = defaultdict(list)
                vid_file_index += 1
                vid_file_bytes = 0
                vid_cumulative_s = 0.0

            ep_from_ts = vid_cumulative_s
            ep_to_ts = vid_cumulative_s + data.n_frames / fps

            for cam_key in CAMERA_KEYS_R1:
                temp_path = tmp_dir / f"ep{ep_idx:06d}_{cam_key.replace('.', '_')}.mp4"
                self.video_writer.write(data.images[cam_key], temp_path, fps)
                pending_video_temps[cam_key].append(temp_path)
            vid_file_bytes += ep_video_bytes
            vid_cumulative_s = ep_to_ts

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
                **{f"videos/{cam}/from_timestamp": ep_from_ts for cam in CAMERA_KEYS_R1},
                **{f"videos/{cam}/to_timestamp": ep_to_ts for cam in CAMERA_KEYS_R1},
            })
            global_frame_index += data.n_frames

        parquet_writer.flush()
        if any(pending_video_temps.values()):
            _flush_video_group(pending_video_temps, self.output_dir,
                               vid_chunk_index, vid_file_index)

        shutil.rmtree(tmp_dir, ignore_errors=True)

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
