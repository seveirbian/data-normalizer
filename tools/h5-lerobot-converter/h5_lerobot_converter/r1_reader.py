from pathlib import Path
import h5py
import numpy as np

from h5_lerobot_converter.reader import EpisodeData

# Public API: these constants are imported by r1_convert_v3 and other callers
R1_STATE_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "left_arm_vel1", "left_arm_vel2", "left_arm_vel3",
    "left_arm_vel4", "left_arm_vel5", "left_arm_vel6",
    "right_arm_vel1", "right_arm_vel2", "right_arm_vel3",
    "right_arm_vel4", "right_arm_vel5", "right_arm_vel6",
    "left_gripper", "right_gripper",
]

R1_ACTION_NAMES = R1_STATE_NAMES[:]


class R1Reader:
    _STATE_PATHS = [
        "joints/state/arm/position",       # 12
        "joints/state/arm/velocity",       # 12
        "joints/state/effector/position",  # 2  → total 26
    ]
    _ACTION_PATHS = [
        "joints/action/arm/position",       # 12
        "joints/action/arm/velocity",       # 12
        "joints/action/effector/position",  # 2  → total 26
    ]
    _CAMERAS = {
        "observation.images.hand_left":        "cameras/hand_left/color/data",
        "observation.images.hand_left_depth":  "cameras/hand_left/depth/data",
        "observation.images.hand_right":       "cameras/hand_right/color/data",
        "observation.images.hand_right_depth": "cameras/hand_right/depth/data",
        "observation.images.head":             "cameras/head/color/data",
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
            state=state, action=action,
            timestamps=timestamps, images=images, n_frames=n,
        )


# Public API: these constants are imported by r1_convert_v3 and other callers
CAMERA_KEYS_R1 = list(R1Reader._CAMERAS)
