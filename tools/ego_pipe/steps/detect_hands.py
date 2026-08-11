"""检测每帧里的左右手 21 个关键点,就地写回 Frames。"""

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
