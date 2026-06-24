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
