from pathlib import Path
import numpy as np
import pandas as pd
from h5_lerobot_converter.reader import EpisodeData


class ParquetWriter:
    def write(
        self,
        data: EpisodeData,
        episode_index: int,
        global_start_index: int,
        task_index: int,
        output_path: Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = data.n_frames

        done = [False] * (n - 1) + [True]
        df = pd.DataFrame({
            "observation.state": [data.state[i].tolist() for i in range(n)],
            "action":            [data.action[i].tolist() for i in range(n)],
            "timestamp":         data.timestamps.tolist(),
            "frame_index":       list(range(n)),
            "episode_index":     [episode_index] * n,
            "index":             list(range(global_start_index, global_start_index + n)),
            "task_index":        [task_index] * n,
            "next.done":         done,
        })
        df.to_parquet(output_path, index=False)
