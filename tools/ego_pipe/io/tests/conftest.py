import cv2
import numpy as np
import pytest

N_FRAMES = 5
WIDTH, HEIGHT = 64, 48
FPS = 10.0
# BGR 纯色帧,用于验证读回后确实转成了 RGB(通道顺序可辨,不依赖压缩后精确像素值)
FRAME_COLORS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]


@pytest.fixture
def tiny_video(tmp_path) -> str:
    """合成一个 5 帧 64x48 @10fps 的纯色 mp4,写到 tmp_path 下,返回路径。"""
    path = str(tmp_path / "tiny.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))
    for color in FRAME_COLORS_BGR:
        frame = np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
