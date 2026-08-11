import os
import socket
import urllib.request

import cv2
import numpy as np
import pytest

from tools.ego_pipe.frames import Frames

# MediaPipe 官方 hand_landmarker 示例图(来自其官方 Colab notebook),已知含双手,专为该任务准备
HAND_IMAGE_URL = "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
HAND_IMAGE_PATH = os.path.join(FIXTURES_DIR, "woman_hands.jpg")

# COCO val2017 常用示例图(HuggingFace Grounding DINO 官方文档同款),已知含猫和遥控器
OBJECT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
OBJECT_IMAGE_PATH = os.path.join(FIXTURES_DIR, "cats_remote.jpg")

N_FRAMES = 3
FPS = 10.0


def _network_available() -> bool:
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(), reason="需要网络下载 MediaPipe 模型/测试图片"
)


@pytest.fixture
def hand_frames() -> Frames:
    """同一张真实手部照片复制 N_FRAMES 帧,构造成 Frames,用于 detect_hands 集成测试。"""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    if not os.path.exists(HAND_IMAGE_PATH):
        try:
            urllib.request.urlretrieve(HAND_IMAGE_URL, HAND_IMAGE_PATH)
        except OSError as e:
            pytest.skip(f"下载测试图片失败: {e}")

    img_bgr = cv2.imread(HAND_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    return Frames(
        frames=np.stack([img_rgb] * N_FRAMES),
        path=HAND_IMAGE_PATH,
        fps=FPS,
        width=w,
        height=h,
        n_frames=N_FRAMES,
        duration_sec=N_FRAMES / FPS,
        fourcc="",
    )


@pytest.fixture
def object_frames() -> Frames:
    """同一张真实猫+遥控器照片复制 N_FRAMES 帧,构造成 Frames,用于 detect_objects 集成测试。"""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    if not os.path.exists(OBJECT_IMAGE_PATH):
        try:
            urllib.request.urlretrieve(OBJECT_IMAGE_URL, OBJECT_IMAGE_PATH)
        except OSError as e:
            pytest.skip(f"下载测试图片失败: {e}")

    img_bgr = cv2.imread(OBJECT_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    return Frames(
        frames=np.stack([img_rgb] * N_FRAMES),
        path=OBJECT_IMAGE_PATH,
        fps=FPS,
        width=w,
        height=h,
        n_frames=N_FRAMES,
        duration_sec=N_FRAMES / FPS,
        fourcc="",
    )
