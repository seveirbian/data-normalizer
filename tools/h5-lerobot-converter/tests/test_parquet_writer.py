import numpy as np
import pandas as pd
import pytest
from h5_lerobot_converter.reader import EpisodeData
from h5_lerobot_converter.parquet_writer import ParquetWriter


def make_episode(n: int = 5) -> EpisodeData:
    return EpisodeData(
        state=np.random.rand(n, 27).astype(np.float32),
        action=np.random.rand(n, 25).astype(np.float32),
        timestamps=np.arange(n, dtype=np.float32) / 30.0,
        images={},
        n_frames=n,
    )


def test_parquet_schema(tmp_path):
    out = tmp_path / "chunk-000" / "episode_000000.parquet"
    ParquetWriter().write(make_episode(), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    expected_cols = {
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index", "next.done",
    }
    assert set(df.columns) == expected_cols


def test_parquet_row_count(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=7), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert len(df) == 7


def test_parquet_global_index(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=5), episode_index=2, global_start_index=100,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert df["episode_index"].unique().tolist() == [2]
    assert df["index"].tolist() == list(range(100, 105))


def test_next_done_last_frame(tmp_path):
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(make_episode(n=5), episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    assert df["next.done"].tolist() == [False, False, False, False, True]


def test_state_values_preserved(tmp_path):
    ep = make_episode(n=3)
    out = tmp_path / "ep.parquet"
    ParquetWriter().write(ep, episode_index=0, global_start_index=0,
                          task_index=0, output_path=out)
    df = pd.read_parquet(out)
    recovered = np.array(df["observation.state"].tolist(), dtype=np.float32)
    np.testing.assert_allclose(recovered, ep.state, rtol=1e-5)
