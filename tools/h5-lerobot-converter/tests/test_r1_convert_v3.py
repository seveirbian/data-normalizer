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


def test_episodes_parquet_has_video_timestamps(two_r1_episodes):
    """episodes parquet 必须含有 from_timestamp / to_timestamp，供 _query_videos 定位帧。"""
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="grab cup", fps=30.0)

    df = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    cam = "observation.images.hand_left"
    assert f"videos/{cam}/from_timestamp" in df.columns, "缺少 from_timestamp"
    assert f"videos/{cam}/to_timestamp" in df.columns, "缺少 to_timestamp"

    # 两个 episode 在同一 video 文件里，时间戳应连续
    assert df.iloc[0][f"videos/{cam}/from_timestamp"] == pytest.approx(0.0)
    assert df.iloc[0][f"videos/{cam}/to_timestamp"] == pytest.approx(10 / 30.0)
    assert df.iloc[1][f"videos/{cam}/from_timestamp"] == pytest.approx(10 / 30.0)
    assert df.iloc[1][f"videos/{cam}/to_timestamp"] == pytest.approx(20 / 30.0)


def test_multi_batch_data_integrity_r1(two_r1_episodes, r1_mini_h5, tmp_path):
    """第二批次写入不得覆盖第一批次的 parquet 数据。"""
    h5_files, out = two_r1_episodes
    DatasetBuilderR1V3(out).build(h5_files, task="task A")  # episodes 0, 1

    ep2 = tmp_path / "batch2" / "ep2.h5"
    ep2.parent.mkdir()
    shutil.copy(r1_mini_h5, ep2)
    DatasetBuilderR1V3(out).build([ep2], task="task B")  # episode 2

    all_dfs = []
    for pq in sorted((out / "data").glob("chunk-*/file-*.parquet")):
        all_dfs.append(pd.read_parquet(pq, columns=["episode_index", "index"]))
    df = pd.concat(all_dfs)

    assert set(df["episode_index"].unique()) == {0, 1, 2}, \
        f"数据缺失！现有 episode_index: {sorted(df['episode_index'].unique())}"
    assert len(df) == 30, f"期望 30 帧，实际 {len(df)}"
