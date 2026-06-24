import subprocess
import numpy as np
import cv2
import pytest
from h5_lerobot_converter.video_writer import VideoWriter


def make_frames(n: int = 5, h: int = 32, w: int = 48) -> list:
    frames = []
    for _ in range(n):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        frames.append(buf.tobytes())
    return frames


def test_video_created(tmp_path):
    out = tmp_path / "test.mp4"
    VideoWriter().write(make_frames(), out, fps=5.0)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_video_duration(tmp_path):
    out = tmp_path / "test.mp4"
    VideoWriter().write(make_frames(n=30), out, fps=30.0)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    assert abs(duration - 1.0) < 0.2, f"expected ~1s, got {duration}s"


def test_parent_dirs_created(tmp_path):
    out = tmp_path / "deep" / "nested" / "ep.mp4"
    VideoWriter().write(make_frames(), out, fps=5.0)
    assert out.exists()


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0])
def test_frame_timestamp_precision(tmp_path, fps):
    """帧 PTS 误差必须 < 0.0001s，满足 LeRobot 0.5.1 FrameTimestampError 容忍度。
    覆盖 25/30/60fps，确保不绑死单一帧率。
    """
    out = tmp_path / f"test_{int(fps)}fps.mp4"
    VideoWriter().write(make_frames(n=int(fps * 2)), out, fps=fps)

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v",
         "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )
    actual_pts = []
    for t in result.stdout.strip().splitlines():
        try:
            actual_pts.append(float(t))
        except ValueError:
            pass
    assert len(actual_pts) > 0, "ffprobe 未返回任何帧时间戳"
    tolerance = 0.0001  # LeRobot 0.5.1 的容忍度
    for pts in actual_pts:
        nearest_frame = round(pts * fps)
        expected = nearest_frame / fps
        error = abs(pts - expected)
        assert error < tolerance, (
            f"fps={fps}: PTS {pts:.6f}s 离最近帧边界 {expected:.6f}s 误差 {error:.6f}s > {tolerance}s\n"
            f"（LeRobot 0.5.1 会抛出 FrameTimestampError）"
        )


def test_no_b_frames(tmp_path):
    """视频必须用 -bf 0 编码，否则 torchcodec 随机帧访问极慢（B-frame 解码需要前后参考帧）。"""
    out = tmp_path / "test.mp4"
    VideoWriter().write(make_frames(n=30), out, fps=30.0)
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(out)],
        capture_output=True, text=True,
    )
    import json
    streams = json.loads(result.stdout)["streams"]
    video_stream = next(s for s in streams if s.get("codec_type") == "video")
    has_b = int(video_stream.get("has_b_frames", 1))
    assert has_b == 0, f"视频不应有 B-frames (has_b_frames={has_b})，会导致 torchcodec 随机访问极慢"
