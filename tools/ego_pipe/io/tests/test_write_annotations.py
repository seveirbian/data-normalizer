import json
import os

import numpy as np
import pytest

from tools.ego_pipe.frames import Frames
from tools.ego_pipe.io.write_annotations import write_annotations

WIDTH, HEIGHT = 64, 48
N_FRAMES = 2
FPS = 10.0


def _frames_with_detections() -> Frames:
    hand_landmarks = np.full((N_FRAMES, 2, 21, 3), np.nan, dtype=np.float64)
    hand_landmarks[0, 0, 0] = [0.5, 0.5, 0.0]
    handedness = np.full((N_FRAMES, 2), "", dtype=object)
    handedness[0, 0] = "Left"
    handedness_score = np.full((N_FRAMES, 2), np.nan, dtype=np.float64)
    handedness_score[0, 0] = 0.9

    return Frames(
        frames=np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8),
        path="synthetic.mp4",
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        n_frames=N_FRAMES,
        duration_sec=N_FRAMES / FPS,
        fourcc="avc1",
        hand_landmarks=hand_landmarks,
        hand_world_landmarks=hand_landmarks.copy(),
        handedness=handedness,
        handedness_score=handedness_score,
        detected_objects=[
            [{"box": (10.0, 10.0, 30.0, 30.0), "label": "cup", "score": 0.8}],
            [],
        ],
    )


def test_write_annotations_creates_file(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_write_annotations_excludes_frames_field(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    with open(out) as f:
        data = json.load(f)
    assert "frames" not in data


def test_write_annotations_includes_metadata_fields(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    with open(out) as f:
        data = json.load(f)
    assert data["path"] == "synthetic.mp4"
    assert data["fps"] == FPS
    assert data["width"] == WIDTH
    assert data["height"] == HEIGHT
    assert data["n_frames"] == N_FRAMES
    assert data["duration_sec"] == N_FRAMES / FPS
    assert data["fourcc"] == "avc1"


def test_write_annotations_converts_nan_to_null(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    with open(out) as f:
        data = json.load(f)
    # frame 1, slot 0 的 handedness_score 是 NaN
    assert data["handedness_score"][1][0] is None
    # frame 0, slot 0, landmark 0 的 z 分量是 0.0(非 NaN),frame 0 slot 1 全是 NaN
    assert data["hand_landmarks"][0][1][0][0] is None


def test_write_annotations_preserves_real_values(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    with open(out) as f:
        data = json.load(f)
    assert data["hand_landmarks"][0][0][0] == [0.5, 0.5, 0.0]
    assert data["handedness"][0][0] == "Left"
    assert data["handedness_score"][0][0] == 0.9


def test_write_annotations_serializes_detected_objects(tmp_path) -> None:
    out = str(tmp_path / "out.json")
    write_annotations(_frames_with_detections(), out)

    with open(out) as f:
        data = json.load(f)
    assert data["detected_objects"][0] == [
        {"box": [10.0, 10.0, 30.0, 30.0], "label": "cup", "score": 0.8}
    ]
    assert data["detected_objects"][1] == []


def test_write_annotations_creates_output_dir(tmp_path) -> None:
    out = str(tmp_path / "nested" / "dir" / "out.json")
    write_annotations(_frames_with_detections(), out)

    assert os.path.exists(out)


def test_write_annotations_logs_summary(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    out = str(tmp_path / "out.json")
    with caplog.at_level("INFO"):
        write_annotations(_frames_with_detections(), out)

    assert out in caplog.text
