"""用 MediaPipe HandLandmarker 检测每帧里的左右手 21 个关键点,就地写回 Frames。

MediaPipe HandLandmarker 的能力与限制:
- 和 detect_objects 的 Grounding DINO 一样,纯逐帧检测,不带跨帧 ID/跟踪能力:
  同一只手在相邻帧之间没有跨帧一致的身份,只靠 handedness(Left/Right)分到
  两个固定槽位,不是"第一只检测到的手"这种顺序。
- 最多同时输出 2 只手(num_hands=2);某帧只检测到一只手时,另一槽位留空
  (NaN/空字符串),不会补位到另一只手上。
- 21 个关键点是 MediaPipe 固定的手部拓扑,索引不可配置:
    0        WRIST
    1-4      THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP
    5-8      INDEX_FINGER_MCP, PIP, DIP, TIP
    9-12     MIDDLE_FINGER_MCP, PIP, DIP, TIP
    13-16    RING_FINGER_MCP, PIP, DIP, TIP
    17-20    PINKY_MCP, PIP, DIP, TIP

标准输出格式(写回 Frames 的字段,详见 frames.py 里的字段注释;风格上对应
detect_objects 里 detected_objects 的 {"box", "label", "score"} 字典格式):
- hand_landmarks: (T, 2, 21, 3) 图像归一化坐标 (x, y, z),x/y 相对图像宽高
  ∈ [0,1],z 是相对手腕的深度(与 x 同数量级,越小越靠近相机)。
- hand_world_landmarks: (T, 2, 21, 3) 米制 3D 坐标,以手腕为原点,与相机
  坐标系无关,不能直接做图像投影。
- handedness: (T, 2) 'Left'/'Right'/''(未检测到)。
- handedness_score: (T, 2) MediaPipe 给出的左右手分类置信度。
- 以上 4 个数组第 2 维都是固定槽位 0=Left 1=Right;某手未检测到时该槽位
  在 landmarks/world_landmarks 里为 NaN,handedness 为空字符串,
  handedness_score 为 NaN。

不含旋转信息:
- HandLandmarkerResult 只有关键点位置(上面 4 个字段),没有 wrist/手掌的朝向
  (旋转矩阵/四元数/axis-angle),也没有每个手指关节各自的局部旋转,这是
  MediaPipe HandLandmarker 这个模型本身的输出就不含 pose,不是本实现遗漏。
- 如果下游需要旋转信息,两种思路:
  1. 从现有 landmarks 几何推导:用 wrist->index_MCP(5)、wrist->pinky_MCP(17)
     两个向量叉乘,构造手掌局部坐标系,得到近似的整手朝向(3x3 旋转矩阵)。
     实现简单、复用现有数据,但只能给整手一个粗略朝向,覆盖不了手指弯曲这种
     每个关节各自的旋转。
  2. 换用输出参数化手部模型(MANO)的方法,比如仓库里已有的 HaWoR(Hand World
     Reconstruction,/root/codes/ego-hoi-gen/HaWoR),直接回归 global orientation
     + 每个关节的 axis-angle 旋转,信息更完整,适合做 retarget/重建,但计算量
     远大于 MediaPipe 这种逐帧关键点检测。
"""

import logging
import os
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from tqdm import tqdm

from tools.ego_pipe.frames import Frames

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "hand_landmarker.task"
)
_SLOTS = {"Left": 0, "Right": 1}


def _ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def detect_hands(frames: Frames, show_progress: bool = True) -> None:
    _ensure_model()

    n = frames.n_frames
    hand_landmarks = np.full((n, 2, 21, 3), np.nan, dtype=np.float64)
    hand_world_landmarks = np.full((n, 2, 21, 3), np.nan, dtype=np.float64)
    handedness = np.full((n, 2), "", dtype=object)
    handedness_score = np.full((n, 2), np.nan, dtype=np.float64)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for i in tqdm(range(n), desc="detect_hands", unit="frame", disable=not show_progress):
            timestamp_ms = int(i / frames.fps * 1000)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames.frames[i])
            result = landmarker.detect_for_video(image, timestamp_ms)

            for lm, world_lm, hd in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness
            ):
                label = hd[0].category_name
                slot = _SLOTS.get(label)
                if slot is None:
                    continue
                hand_landmarks[i, slot] = [[p.x, p.y, p.z] for p in lm]
                hand_world_landmarks[i, slot] = [[p.x, p.y, p.z] for p in world_lm]
                handedness[i, slot] = label
                handedness_score[i, slot] = hd[0].score

    frames.hand_landmarks = hand_landmarks
    frames.hand_world_landmarks = hand_world_landmarks
    frames.handedness = handedness
    frames.handedness_score = handedness_score

    n_left = int((handedness[:, 0] == "Left").sum())
    n_right = int((handedness[:, 1] == "Right").sum())
    logger.info("%d/%d 帧检测到左手, %d/%d 帧检测到右手", n_left, n, n_right, n)
