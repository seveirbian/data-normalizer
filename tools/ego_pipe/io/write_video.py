"""把检测到的手部关键点叠加到原始帧上,写成一个视频文件。"""

import logging
import os
from fractions import Fraction

import av
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from tqdm import tqdm

from tools.ego_pipe.frames import Frames

logger = logging.getLogger(__name__)

_COLORS_RGB = {"Left": (0, 200, 0), "Right": (220, 220, 0)}  # Left=绿, Right=黄
_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
_LEGEND_BG_RGB = (128, 128, 128)
_LEGEND_TEXT_RGB = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def write_video(frames: Frames, path: str, show_progress: bool = True) -> None:
    if frames.hand_landmarks is None:
        raise ValueError("frames.hand_landmarks is None; 需要先跑 detect_hands")

    logger.info("开始写入 %d 帧 -> %s", frames.n_frames, path)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=Fraction(frames.fps).limit_denominator(1000))
    stream.width = frames.width
    stream.height = frames.height
    stream.pix_fmt = "yuv420p"

    line_thickness, point_radius = _line_thickness_and_point_radius(frames.width)

    for i in tqdm(
        range(frames.n_frames), desc="write_video", unit="frame", disable=not show_progress
    ):
        frame_rgb = frames.frames[i].copy()
        for slot, label in enumerate(("Left", "Right")):
            landmarks = frames.hand_landmarks[i, slot]
            if np.isnan(landmarks).any():
                continue
            wrist_px = _draw_hand(
                frame_rgb,
                landmarks,
                frames.width,
                frames.height,
                _COLORS_RGB[label],
                line_thickness,
                point_radius,
            )
            _draw_score(
                frame_rgb, wrist_px, frames.width, _COLORS_RGB[label],
                frames.handedness_score[i, slot],
            )
        _draw_legend(frame_rgb, frames.width)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    logger.info("写入 %d 帧 -> %s", frames.n_frames, path)


def _line_thickness_and_point_radius(width: int) -> tuple[int, int]:
    line_thickness = max(2, width // 400)
    point_radius = line_thickness * 2
    return line_thickness, point_radius


def _font_scale_and_thickness(width: int) -> tuple[float, int]:
    font_scale = max(0.4, width / 1600)
    text_thickness = max(2, width // 800)  # 太细的文字会被 h264 压缩糊掉,设个下限
    return font_scale, text_thickness


def _draw_hand(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    width: int,
    height: int,
    color: tuple,
    line_thickness: int,
    point_radius: int,
) -> tuple[int, int]:
    points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
    for conn in _CONNECTIONS:
        cv2.line(frame_rgb, points[conn.start], points[conn.end], color, line_thickness)
    for p in points:
        cv2.circle(frame_rgb, p, point_radius, color, -1)
    return points[0]  # 手腕


def _draw_score(
    frame_rgb: np.ndarray, wrist_px: tuple[int, int], width: int, color: tuple, score: float
) -> None:
    font_scale, text_thickness = _font_scale_and_thickness(width)
    pad = max(2, width // 300)
    text = f"{score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, font_scale, text_thickness)

    cx, cy = wrist_px
    top_left = (cx - text_w // 2 - pad, cy + pad)
    bottom_right = (cx + text_w // 2 + pad, cy + text_h + pad * 3)

    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, top_left, bottom_right, _LEGEND_BG_RGB, -1)
    cv2.addWeighted(overlay, 0.5, frame_rgb, 0.5, 0, dst=frame_rgb)

    text_origin = (cx - text_w // 2, cy + pad + text_h + pad)
    cv2.putText(
        frame_rgb, text, text_origin, _FONT, font_scale, color, text_thickness, cv2.LINE_AA
    )


def _draw_legend(frame_rgb: np.ndarray, width: int) -> None:
    font_scale, text_thickness = _font_scale_and_thickness(width)
    margin = max(8, width // 100)
    swatch = max(10, width // 90)
    gap = max(4, width // 300)

    labels = ("Left", "Right")
    line_h = max(swatch, max(cv2.getTextSize(t, _FONT, font_scale, text_thickness)[0][1] for t in labels))
    text_w = max(cv2.getTextSize(t, _FONT, font_scale, text_thickness)[0][0] for t in labels)
    block_w = swatch + gap * 2 + text_w
    block_h = line_h * 2 + gap * 3

    overlay = frame_rgb.copy()
    cv2.rectangle(
        overlay, (margin, margin), (margin + block_w, margin + block_h), _LEGEND_BG_RGB, -1
    )
    cv2.addWeighted(overlay, 0.5, frame_rgb, 0.5, 0, dst=frame_rgb)

    y = margin + gap + line_h
    for label in labels:
        x = margin + gap
        cv2.rectangle(frame_rgb, (x, y - swatch), (x + swatch, y), _COLORS_RGB[label], -1)
        cv2.putText(
            frame_rgb,
            label,
            (x + swatch + gap, y),
            _FONT,
            font_scale,
            _LEGEND_TEXT_RGB,
            text_thickness,
            cv2.LINE_AA,
        )
        y += line_h + gap
