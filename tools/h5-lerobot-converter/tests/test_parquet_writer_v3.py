import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from h5_lerobot_converter.parquet_writer_v3 import ParquetWriterV3
from h5_lerobot_converter.reader import EpisodeData


def _make_episode(n_frames=10):
    return EpisodeData(
        state=np.zeros((n_frames, 27), dtype=np.float32),
        action=np.zeros((n_frames, 25), dtype=np.float32),
        timestamps=np.arange(n_frames, dtype=np.float32) / 30.0,
        images={},
        n_frames=n_frames,
    )


def test_single_episode_file_location(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    loc = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    writer.flush()

    assert loc == {"chunk_index": 0, "file_index": 0}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()


def test_single_episode_schema_and_values(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    ep = _make_episode(n_frames=5)
    writer.write_episode(ep, episode_index=0, global_start_index=0, task_index=2)
    writer.flush()

    df = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert set(df.columns) >= {
        "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "index", "task_index", "next.done"
    }
    assert len(df) == 5
    assert df["index"].tolist() == [0, 1, 2, 3, 4]
    assert df["episode_index"].tolist() == [0, 0, 0, 0, 0]
    assert df["task_index"].tolist() == [2, 2, 2, 2, 2]
    assert df["next.done"].tolist() == [False, False, False, False, True]
    assert len(df["observation.state"].iloc[0]) == 27
    assert len(df["action"].iloc[0]) == 25


def test_two_episodes_same_file(tmp_path):
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=100.0)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == loc1 == {"chunk_index": 0, "file_index": 0}
    df = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert len(df) == 20
    assert df["index"].tolist() == list(range(20))


def test_size_limit_triggers_new_file(tmp_path):
    # Tiny size limit so ep1 triggers internal flush of ep0 and goes to file-001
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1000, max_size_mb=1e-9)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    # Do NOT flush manually — ep0 stays buffered so the size check fires for ep1
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == {"chunk_index": 0, "file_index": 0}
    assert loc1 == {"chunk_index": 0, "file_index": 1}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "data" / "chunk-000" / "file-001.parquet").exists()


def test_chunk_boundary_resets_file_index(tmp_path):
    # chunk_size=1: episode 0 → chunk 0, episode 1 → chunk 1
    writer = ParquetWriterV3(tmp_path / "data", chunk_size=1, max_size_mb=100.0)
    loc0 = writer.write_episode(_make_episode(), episode_index=0, global_start_index=0, task_index=0)
    writer.flush()
    loc1 = writer.write_episode(_make_episode(), episode_index=1, global_start_index=10, task_index=0)
    writer.flush()

    assert loc0 == {"chunk_index": 0, "file_index": 0}
    assert loc1 == {"chunk_index": 1, "file_index": 0}
    assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()
    assert (tmp_path / "data" / "chunk-001" / "file-000.parquet").exists()
