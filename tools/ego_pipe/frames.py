"""pipeline 处理的共享领域对象:io 层负责构造它,steps 就地往上面填字段。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frames:
    # io.read_video 写入
    frames: np.ndarray  # (T, H, W, 3), RGB
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_sec: float
    fourcc: str

    # steps.detect_hands 写入;跑之前保持 None
    hand_landmarks: np.ndarray | None = None  # (T, 2, 21, 3) 归一化图像坐标,0=Left 1=Right,缺失为 NaN
    hand_world_landmarks: np.ndarray | None = None  # (T, 2, 21, 3) 米制 3D 坐标(以手腕为原点),缺失为 NaN
    handedness: np.ndarray | None = None  # (T, 2) 'Left'/'Right'/'',缺失为 ''
    handedness_score: np.ndarray | None = None  # (T, 2) float,缺失为 NaN

    # steps.detect_objects 写入;跑之前保持 None
    # 长度为 T 的 list,每帧一个 list[dict],每个 dict: {"box": (x1,y1,x2,y2) 像素坐标, "label": str, "score": float}
    detected_objects: list[list[dict]] | None = None
