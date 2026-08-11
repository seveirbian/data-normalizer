import os

import cv2
import numpy as np
import pytest

from tools.ego_pipe.frames import Frames
from tools.ego_pipe.io.write_video import (
    _color_for_label,
    _line_thickness_and_point_radius,
    write_video,
)

WIDTH, HEIGHT = 640, 480  # 帧太小时 h264 压缩会把细小文字/图例压花,像素级断言会不稳定
N_FRAMES = 2
FPS = 10.0


RIGHT_PX = (int(0.2 * HEIGHT), int(0.2 * WIDTH))  # (row, col)

OBJECT_BOX = (100.0, 100.0, 300.0, 250.0)  # (x1, y1, x2, y2)
OBJECT_LABEL = "cup"
OBJECT_SCORE = 0.8


def _blank_frames_with_one_hand() -> Frames:
    """frame 0 有一只完整的左手(手腕落在图像正中心)+一只完整的右手(左上角)+一个物体框,frame 1 什么都没检测到。"""
    frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)

    hand_landmarks = np.full((N_FRAMES, 2, 21, 3), np.nan, dtype=np.float64)
    handedness = np.full((N_FRAMES, 2), "", dtype=object)
    for i in range(21):
        hand_landmarks[0, 0, i] = [0.5 + i * 0.005, 0.5, 0.0]  # Left wrist(0) 在正中心
        hand_landmarks[0, 1, i] = [0.2 + i * 0.005, 0.2, 0.0]  # Right wrist(0) 在左上角
    handedness[0, 0] = "Left"
    handedness[0, 1] = "Right"

    return Frames(
        frames=frames,
        path="synthetic",
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        n_frames=N_FRAMES,
        duration_sec=N_FRAMES / FPS,
        fourcc="",
        hand_landmarks=hand_landmarks,
        hand_world_landmarks=hand_landmarks.copy(),
        handedness=handedness,
        handedness_score=np.where(handedness == "", np.nan, 0.9),
        detected_objects=[
            [{"box": OBJECT_BOX, "label": OBJECT_LABEL, "score": OBJECT_SCORE}],
            [],
        ],
    )


def _read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame  # BGR


def test_write_video_creates_readable_file_with_correct_shape(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    assert os.path.exists(out) and os.path.getsize(out) > 0
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert n == N_FRAMES
    assert (w, h) == (WIDTH, HEIGHT)


def test_write_video_draws_left_hand_green_at_wrist_pixel(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_hands=True, show_progress=False)

    wrist_px = _read_frame(out, 0)[HEIGHT // 2, WIDTH // 2]  # BGR
    assert wrist_px[1] > 150  # G 通道应该是主导(Left=绿色)
    assert int(wrist_px[1]) > int(wrist_px[2]) + 50  # 明显比 R 通道亮


def test_write_video_draws_right_hand_yellow_at_wrist_pixel(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_hands=True, show_progress=False)

    row, col = RIGHT_PX
    wrist_px = _read_frame(out, 0)[row, col]  # BGR
    assert wrist_px[1] > 150  # G 通道高
    assert wrist_px[2] > 150  # R 通道高(Right=黄色,R+G 都高)
    assert int(wrist_px[0]) < int(wrist_px[1]) - 50  # B 通道明显更暗


def test_write_video_frame_without_hand_stays_blank(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_hands=True, show_progress=False)

    frame1 = _read_frame(out, 1)
    assert frame1[HEIGHT // 2, WIDTH // 2].max() < 30  # 仍是接近黑色的背景


def test_write_video_missing_hand_data_raises_value_error(tmp_path) -> None:
    frames = _blank_frames_with_one_hand()
    frames.hand_landmarks = None

    with pytest.raises(ValueError):
        write_video(frames, str(tmp_path / "out.mp4"), draw_hands=True, show_progress=False)


def test_write_video_draw_hands_false_ignores_missing_hand_data(tmp_path) -> None:
    frames = _blank_frames_with_one_hand()
    frames.hand_landmarks = None

    write_video(frames, str(tmp_path / "out.mp4"), draw_hands=False, show_progress=False)


def test_write_video_draw_hands_and_draw_objects_false_by_default_leaves_frame_blank(
    tmp_path,
) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    frame0 = _read_frame(out, 0)
    assert frame0.max() < 30


def test_write_video_creates_output_dir(tmp_path) -> None:
    out = str(tmp_path / "nested" / "dir" / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    assert os.path.exists(out)


def test_write_video_logs_summary(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    out = str(tmp_path / "out.mp4")
    with caplog.at_level("INFO"):
        write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    assert out in caplog.text
    assert str(N_FRAMES) in caplog.text


def test_write_video_logs_entry_before_encoding(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    out = str(tmp_path / "out.mp4")
    with caplog.at_level("INFO"):
        write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    assert len(caplog.records) >= 2
    assert out in caplog.records[0].getMessage()


def test_line_thickness_and_point_radius_scale_with_width() -> None:
    small_line, small_radius = _line_thickness_and_point_radius(64)
    large_line, large_radius = _line_thickness_and_point_radius(1920)

    assert small_line >= 2  # 有下限,不会细到 1px
    assert large_line > small_line  # 大分辨率下明显更粗
    assert small_radius == small_line * 2
    assert large_radius == large_line * 2


def _region_has_color_close_to(region_bgr: np.ndarray, target_rgb: tuple, tol: int = 40) -> bool:
    target_bgr = np.array(target_rgb[::-1])
    diff = np.abs(region_bgr.astype(int) - target_bgr).sum(axis=-1)
    return bool((diff < tol).any())


def _region_has_grayish_pixel(region_bgr: np.ndarray) -> bool:
    grayish = np.all(
        np.abs(region_bgr.astype(int) - region_bgr.astype(int).mean(axis=-1, keepdims=True)) < 15,
        axis=-1,
    )
    not_black = region_bgr.max(axis=-1) > 20
    return bool((grayish & not_black).any())


def test_write_video_draws_legend_with_both_colors_and_gray_background(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_hands=True, show_progress=False)

    # 图例画在左上角,取一个足够大的角落区域来扫描,不依赖具体像素坐标
    corner = _read_frame(out, 1)[: HEIGHT // 2, : WIDTH // 2]  # frame 1 没有手,只有图例

    assert _region_has_color_close_to(corner, (0, 200, 0))  # Left 绿色色块
    assert _region_has_color_close_to(corner, (220, 220, 0))  # Right 黄色色块
    assert _region_has_grayish_pixel(corner)  # 半透明灰色底框


def test_write_video_draws_handedness_score_below_each_wrist(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(
        _blank_frames_with_one_hand(), out, draw_hands=True, show_progress=False
    )  # 两只手的 handedness_score 都是 0.9

    # fixture 里 21 个关键点是一条水平线(y 恒定),skeleton 本身贴着手腕那一行(含 point_radius
    # 圈住的范围);分数标签紧贴在这条线正下方,所以扫描区域从 point_radius 之后开始,取足够高
    # 的窗口(100px)以覆盖标签本身,不依赖标签内部具体的 pad/字高排布细节
    _, point_radius = _line_thickness_and_point_radius(WIDTH)

    frame0 = _read_frame(out, 0)
    left_row, left_col = HEIGHT // 2, WIDTH // 2
    below_left = frame0[
        left_row + point_radius : left_row + point_radius + 100,
        max(0, left_col - 60) : left_col + 60,
    ]
    assert _region_has_grayish_pixel(below_left)
    assert _region_has_color_close_to(below_left, (0, 200, 0))  # Left 绿色文字

    right_row, right_col = RIGHT_PX
    below_right = frame0[
        right_row + point_radius : right_row + point_radius + 100,
        max(0, right_col - 60) : right_col + 60,
    ]
    assert _region_has_grayish_pixel(below_right)
    assert _region_has_color_close_to(below_right, (220, 220, 0))  # Right 黄色文字


def test_write_video_show_progress_true_prints_progress_bar(tmp_path, capsys) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, show_progress=True)

    assert "write_video" in capsys.readouterr().err  # tqdm 默认写到 stderr


def test_write_video_show_progress_false_prints_nothing(tmp_path, capsys) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, show_progress=False)

    assert capsys.readouterr().err == ""


def test_color_for_label_is_deterministic_and_distinguishes_labels() -> None:
    assert _color_for_label("cup") == _color_for_label("cup")
    assert _color_for_label("cup") != _color_for_label("bottle")


def test_write_video_draws_object_corner_brackets_not_full_rectangle(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_objects=True, show_progress=False)

    frame0 = _read_frame(out, 0)
    color = _color_for_label(OBJECT_LABEL)

    x1, y1, x2, y2 = (int(v) for v in OBJECT_BOX)
    top_left_corner = frame0[y1 : y1 + 40, x1 : x1 + 40]
    assert _region_has_color_close_to(top_left_corner, color)

    # 中点不应该被画到,证明画的是四角括号而不是完整矩形边框
    mid_top_edge = frame0[y1 : y1 + 10, (x1 + x2) // 2 - 10 : (x1 + x2) // 2 + 10]
    assert not _region_has_color_close_to(mid_top_edge, color)


def test_write_video_draws_object_label_above_box(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_objects=True, show_progress=False)

    frame0 = _read_frame(out, 0)
    color = _color_for_label(OBJECT_LABEL)

    x1, y1, _, _ = (int(v) for v in OBJECT_BOX)
    above_box = frame0[max(0, y1 - 40) : y1, x1 : x1 + 150]
    assert _region_has_color_close_to(above_box, color)


def test_write_video_frame_without_objects_stays_blank(tmp_path) -> None:
    out = str(tmp_path / "out.mp4")
    write_video(_blank_frames_with_one_hand(), out, draw_objects=True, show_progress=False)

    frame1 = _read_frame(out, 1)
    assert frame1.max() < 30


def test_write_video_missing_object_data_raises_value_error(tmp_path) -> None:
    frames = _blank_frames_with_one_hand()
    frames.detected_objects = None

    with pytest.raises(ValueError):
        write_video(frames, str(tmp_path / "out.mp4"), draw_objects=True, show_progress=False)


def test_write_video_draw_objects_false_ignores_missing_object_data(tmp_path) -> None:
    frames = _blank_frames_with_one_hand()
    frames.detected_objects = None

    write_video(frames, str(tmp_path / "out.mp4"), draw_objects=False, show_progress=False)
