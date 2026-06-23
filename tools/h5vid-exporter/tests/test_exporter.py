import os

import cv2
import pytest

from h5vid_exporter.exporter import export_video


def _read_video(path):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, w, h


def test_export_video_creates_readable_mp4(tiny_h5, tmp_path):
    out = str(tmp_path / "out.mp4")
    export_video(tiny_h5, ["cameras/head/color", "cameras/head/depth"], out,
                 fps=10, height=120)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    frames, w, h = _read_video(out)
    assert frames == 3
    assert h == 120          # height preserved (even)
    assert w >= 120          # two tiles side by side


def test_export_video_no_topics_raises(tiny_h5, tmp_path):
    with pytest.raises(ValueError):
        export_video(tiny_h5, [], str(tmp_path / "x.mp4"))


def test_export_video_invalid_topic_raises(tiny_h5, tmp_path):
    with pytest.raises(ValueError):
        export_video(tiny_h5, ["cameras/nope/color"], str(tmp_path / "x.mp4"))


def test_export_video_frame_count_mismatch_warns_and_uses_min(mismatch_h5, tmp_path):
    out = str(tmp_path / "out.mp4")
    with pytest.warns(UserWarning):
        export_video(mismatch_h5,
                     ["cameras/head/color", "cameras/hand_left/color"],
                     out, fps=10, height=120)
    frames, _, _ = _read_video(out)
    assert frames == 2       # minimum of (3, 2)


def test_export_video_creates_output_dir(tiny_h5, tmp_path):
    out = str(tmp_path / "nested" / "dir" / "out.mp4")
    export_video(tiny_h5, ["cameras/head/color"], out, fps=10, height=120)
    assert os.path.exists(out)
