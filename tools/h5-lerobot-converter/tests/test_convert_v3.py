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
