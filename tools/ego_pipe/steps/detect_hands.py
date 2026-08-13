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
- hand_world_landmarks: (T, 2, 21, 3) 米制 3D 坐标,原点是手掌几何中心
  (MediaPipe 官方定义:"origin at the hand's geometric center",不是手腕)。
- handedness: (T, 2) 'Left'/'Right'/''(未检测到)。
- handedness_score: (T, 2) MediaPipe 给出的左右手分类置信度。
- 以上 4 个数组第 2 维都是固定槽位 0=Left 1=Right;某手未检测到时该槽位
  在 landmarks/world_landmarks 里为 NaN,handedness 为空字符串,
  handedness_score 为 NaN。

hand_landmarks vs hand_world_landmarks——两者回答的是完全不同的问题:
- hand_landmarks 回答"手在画面里哪个位置":x/y 是手在**当前这一帧图像**里的
  归一化像素位置,手在画面里左右移动、远近变化,会直接反映在 x/y 的变化上
  (z 只是相对手腕的粗略深度,不是米制,不能反推真实距离)。但这是"手在 2D
  画面里的位置",这段视频是移动的 ego 相机拍的,相机自己也在动,所以从这组
  数据没法区分"是手在动"还是"是相机在动"。
- hand_world_landmarks 回答"手指相对手掌怎么摆的"(articulation/shape),
  不回答"手在空间里挪到哪了"(position/trajectory)。关键在于 MediaPipe 每一帧
  都独立做一次"以这一帧检测到的手掌几何中心为原点"的坐标系重建——这个坐标系
  是跟着手走的,不是固定在场景里的。举例:手指形状不变、整只手在真实空间里
  平移了 30cm,frame 1 和 frame 100 里 hand_world_landmarks 会几乎完全一样,
  因为两帧都各自把手掌重新归零到 (0,0,0) 了,平移这个信息在归零这一步就被
  丢掉了,只剩下手指相对手掌的相对形状。
- 所以拿 hand_world_landmarks 做可视化,只能看"某一瞬间手的姿态长什么样",
  不能连起来看"手在场景里怎么移动的";真要后者,需要相机位姿(SLAM/VO)或者
  换用 HaWoR 这类做全局重建的方法,不是这份数据能提供的。

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
