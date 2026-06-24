import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.meta_builder_v3 import MetaBuilderV3


@pytest.fixture
def builder(tmp_path):
    return MetaBuilderV3(tmp_path, fps=30.0, chunk_size=1000,
                         data_file_size_mb=100.0, video_file_size_mb=200.0)


def test_write_tasks_creates_parquet(builder, tmp_path):
    builder.write_tasks(["task A", "task B"])
    df = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
    assert set(df.columns) == {"task_index", "task"}
    assert df["task"].tolist() == ["task A", "task B"]
    assert df["task_index"].tolist() == [0, 1]


def test_write_episodes_creates_parquet(builder, tmp_path):
    episodes = [
        {
            "episode_index": 0, "tasks": ["task A"], "length": 10,
            "dataset_from_index": 0, "dataset_to_index": 10,
            "data/chunk_index": 0, "data/file_index": 0,
            "videos/observation.images.hand_left/chunk_index": 0,
            "videos/observation.images.hand_left/file_index": 0,
            "videos/observation.images.hand_right/chunk_index": 0,
            "videos/observation.images.hand_right/file_index": 0,
            "videos/observation.images.head/chunk_index": 0,
            "videos/observation.images.head/file_index": 0,
            "videos/observation.images.head_depth/chunk_index": 0,
            "videos/observation.images.head_depth/file_index": 0,
        }
    ]
    builder.write_episodes(episodes)
    ep_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    assert ep_path.exists()
    df = pd.read_parquet(ep_path)
    assert len(df) == 1
    assert df["episode_index"].iloc[0] == 0
    assert df["data/chunk_index"].iloc[0] == 0
    assert df["data/file_index"].iloc[0] == 0


def test_write_info_v3(builder, tmp_path):
    builder.write_info(
        total_episodes=2, total_frames=20, tasks=["task A"],
        image_shapes={"observation.images.head": (720, 1280, 3)}
    )
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20
    assert info["data_files_size_in_mb"] == 100.0
    assert info["video_files_size_in_mb"] == 200.0
    assert "observation.images.head" in info["features"]


@pytest.fixture
def stats_parquet(tmp_path):
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(5):
        rows.append({
            "observation.state": [float(i)] * 27,
            "action": [float(i)] * 25,
            "timestamp": i / 30.0,
            "frame_index": i,
            "episode_index": 0,
            "index": i,
            "task_index": 0,
            "next.done": i == 4,
        })
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    return [{"data/chunk_index": 0, "data/file_index": 0, "length": 5}]


def test_write_stats_no_nan(builder, tmp_path, stats_parquet):
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    for key in ["mean", "std", "min", "max"]:
        vals = stats["observation.state"][key]
        assert not any(v != v or abs(v) == float("inf") for v in vals)


def test_write_stats_has_count_field(builder, tmp_path, stats_parquet):
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert stats["observation.state"]["count"] == [5]
    assert stats["action"]["count"] == [5]


def test_write_stats_has_quantile_fields(builder, tmp_path, stats_parquet):
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    for q in ["q01", "q10", "q50", "q90", "q99"]:
        assert q in stats["observation.state"], f"{q} missing from observation.state"
        assert q in stats["action"], f"{q} missing from action"
        assert len(stats["observation.state"][q]) == 27
        assert len(stats["action"][q]) == 25


def test_write_stats_includes_all_parquet_features(builder, tmp_path, stats_parquet):
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    expected = {"observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index", "next.done"}
    assert set(stats.keys()) == expected


def test_write_stats_scalar_features_correct_shape(builder, tmp_path, stats_parquet):
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    scalar_feats = ["timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"]
    for feat in scalar_feats:
        for stat_key in ["mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"]:
            val = stats[feat][stat_key]
            assert isinstance(val, list) and len(val) == 1, \
                f"{feat}.{stat_key} should be list of 1 element, got {val}"
        assert stats[feat]["count"] == [5], f"{feat}.count should be [5]"


def test_write_stats_video_features_shape_3_1_1(builder, tmp_path, stats_parquet):
    """所有 video/image 特征的 stats 形状必须是 (3, 1, 1)。
    LeRobot _validate_stat_value 硬编码此要求，深度图（1通道）也不例外。
    factory.py 的 use_imagenet_stats 只覆写 mean/std，其余字段也必须是 (3,1,1)。
    """
    builder.write_info(
        total_episodes=1, total_frames=5, tasks=["t"],
        image_shapes={
            "observation.images.head": (720, 1280, 3),
            "observation.images.head_depth": (720, 1280, 1),  # 1-channel depth
        }
    )
    builder.write_stats(stats_parquet)
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())

    assert "observation.images.head" in stats
    assert "observation.images.head_depth" in stats

    for cam_key in ["observation.images.head", "observation.images.head_depth"]:
        cam_stats = stats[cam_key]
        for field in ["mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"]:
            val = cam_stats[field]
            assert len(val) == 3, \
                f"{cam_key}.{field} 应为 (3,1,1)，LeRobot 硬编码此要求；got len={len(val)}"
            assert len(val[0]) == 1 and len(val[0][0]) == 1, \
                f"{cam_key}.{field} 内层应为 [[v]]，got {val[0]}"
        assert cam_stats["count"] == [5]


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
