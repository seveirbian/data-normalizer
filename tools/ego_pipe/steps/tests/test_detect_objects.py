import pytest

from tools.ego_pipe.frames import Frames
from tools.ego_pipe.steps.detect_objects import DEFAULT_LABELS, detect_objects
from tools.ego_pipe.steps.tests.conftest import N_FRAMES, requires_network

LABELS = ["cat", "remote control"]


@requires_network
def test_detect_objects_fills_one_list_per_frame(object_frames: Frames) -> None:
    detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=False)

    assert len(object_frames.detected_objects) == N_FRAMES


@requires_network
def test_detect_objects_finds_cat_in_known_image(object_frames: Frames) -> None:
    detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=False)

    labels_found = {obj["label"] for obj in object_frames.detected_objects[0]}
    assert "cat" in labels_found


@requires_network
def test_detect_objects_boxes_within_image_bounds(object_frames: Frames) -> None:
    detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=False)

    for obj in object_frames.detected_objects[0]:
        x1, y1, x2, y2 = obj["box"]
        assert 0 <= x1 < x2 <= object_frames.width
        assert 0 <= y1 < y2 <= object_frames.height
        assert 0.0 <= obj["score"] <= 1.0


@requires_network
def test_detect_objects_uses_default_labels_when_omitted(object_frames: Frames) -> None:
    detect_objects(object_frames, device="cpu", show_progress=False)

    assert len(object_frames.detected_objects) == N_FRAMES


def test_default_labels_is_generic_object_list() -> None:
    assert DEFAULT_LABELS == ["hand", "tool", "part", "box", "wire", "bottle", "cup"]


@requires_network
def test_detect_objects_logs_summary(
    object_frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=False)

    assert str(N_FRAMES) in caplog.text


@requires_network
def test_detect_objects_show_progress_true_prints_progress_bar(
    object_frames: Frames, capsys
) -> None:
    detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=True)

    assert "detect_objects" in capsys.readouterr().err  # tqdm 默认写到 stderr


@requires_network
def test_detect_objects_show_progress_false_prints_nothing(
    object_frames: Frames, capsys
) -> None:
    detect_objects(object_frames, labels=LABELS, device="cpu", show_progress=False)

    assert capsys.readouterr().err == ""
