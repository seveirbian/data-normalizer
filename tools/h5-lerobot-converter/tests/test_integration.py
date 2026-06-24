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

    ep2 = tmp_path / "batch2" / "ep2.h5"
    ep2.parent.mkdir()
    shutil.copy(mini_h5, ep2)
    DatasetBuilder(out).build([ep2], task="task B")

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_tasks"] == 2

    tasks = [json.loads(l)["task"] for l in
             (out / "meta" / "tasks.jsonl").read_text().splitlines() if l]
    assert tasks == ["task A", "task B"]
