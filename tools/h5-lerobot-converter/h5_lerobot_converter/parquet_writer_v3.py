from pathlib import Path
import pandas as pd
from h5_lerobot_converter.reader import EpisodeData


class ParquetWriterV3:
    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = 1000,
        max_size_mb: float = 100.0,
        start_chunk_index: int = 0,
        start_file_index: int = 0,
    ):
        self._data_dir = Path(data_dir)
        self._chunk_size = chunk_size
        self._max_bytes = max_size_mb * 1024 * 1024
        self._rows: list = []
        self._current_bytes: int = 0
        self._chunk_index: int = start_chunk_index
        self._file_index: int = start_file_index

    @staticmethod
    def _estimate_bytes(data: EpisodeData) -> int:
        # (27 + 25 + 1) float32 + 5 int32 + 1 bool per frame
        return data.n_frames * ((27 + 25 + 1) * 4 + 5 * 4 + 1)

    def write_episode(
        self,
        data: EpisodeData,
        episode_index: int,
        global_start_index: int,
        task_index: int,
    ) -> dict:
        ep_chunk = episode_index // self._chunk_size
        ep_bytes = self._estimate_bytes(data)

        if ep_chunk != self._chunk_index:
            if self._rows:
                self._flush()
            self._chunk_index = ep_chunk
            self._file_index = 0
        elif self._rows and self._current_bytes + ep_bytes > self._max_bytes:
            self._flush()
            self._file_index += 1

        n = data.n_frames
        done = [False] * (n - 1) + [True]
        for i in range(n):
            self._rows.append({
                "observation.state": data.state[i].tolist(),
                "action":            data.action[i].tolist(),
                "timestamp":         float(data.timestamps[i]),
                "frame_index":       i,
                "episode_index":     episode_index,
                "index":             global_start_index + i,
                "task_index":        task_index,
                "next.done":         done[i],
            })
        self._current_bytes += ep_bytes

        return {"chunk_index": self._chunk_index, "file_index": self._file_index}

    def flush(self) -> None:
        if self._rows:
            self._flush()

    def _flush(self) -> None:
        out = (
            self._data_dir
            / f"chunk-{self._chunk_index:03d}"
            / f"file-{self._file_index:03d}.parquet"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(out, index=False)
        self._rows = []
        self._current_bytes = 0
