from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np


@dataclass
class EpisodeData:
    state: np.ndarray        # (N, 27) float32
    action: np.ndarray       # (N, 25) float32
    timestamps: np.ndarray   # (N,)    float32, seconds
    images: dict             # str -> list[bytes], N jpeg frames
    n_frames: int


class HDF5Reader:
    _STATE_PATHS = [
        "joints/state/arm/position",       # 14
        "joints/state/effector/position",  # 2
        "joints/state/head/position",      # 2
        "joints/state/waist/position",     # 2
        "joints/state/robot/position",     # 3
        "joints/state/robot/orientation",  # 4  → total 27
    ]
    _ACTION_PATHS = [
        "joints/action/arm/position",       # 14
        "joints/action/effector/position",  # 2
        "joints/action/head/position",      # 2
        "joints/action/waist/position",     # 2
        "joints/action/robot/position",     # 3
        "joints/action/robot/velocity",     # 2  → total 25
    ]
    _CAMERAS = {
        "observation.images.hand_left":  "cameras/hand_left/color/data",
        "observation.images.hand_right": "cameras/hand_right/color/data",
        "observation.images.head":       "cameras/head/color/data",
        "observation.images.head_depth": "cameras/head/depth/data",
    }

    def read(self, path: Path) -> EpisodeData:
        with h5py.File(path, "r") as f:
            state = np.concatenate(
                [f[p][:] for p in self._STATE_PATHS], axis=1
            ).astype(np.float32)

            action = np.concatenate(
                [f[p][:] for p in self._ACTION_PATHS], axis=1
            ).astype(np.float32)

            timestamps = (f["timestamp"][:] / 1e9).astype(np.float32)
            n = state.shape[0]

            images = {
                key: [bytes(f[hdf_path][i]) for i in range(n)]
                for key, hdf_path in self._CAMERAS.items()
            }

        return EpisodeData(
            state=state,
            action=action,
            timestamps=timestamps,
            images=images,
            n_frames=n,
        )
