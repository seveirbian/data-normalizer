import pytest

from tools.ego_pipe.frames import Frames
from tools.ego_pipe.steps.detect_hands import detect_hands
from tools.ego_pipe.steps.tests.conftest import N_FRAMES, requires_network


@requires_network
def test_detect_hands_fills_fixed_shape_fields(hand_frames: Frames) -> None:
    detect_hands(hand_frames, show_progress=False)

    assert hand_frames.hand_landmarks.shape == (N_FRAMES, 2, 21, 3)
    assert hand_frames.hand_world_landmarks.shape == (N_FRAMES, 2, 21, 3)
    assert hand_frames.handedness.shape == (N_FRAMES, 2)
    assert hand_frames.handedness_score.shape == (N_FRAMES, 2)


@requires_network
def test_detect_hands_finds_both_hands_in_known_image(hand_frames: Frames) -> None:
    detect_hands(hand_frames, show_progress=False)

    # 官方示例图确定含双手,每一帧的 Left/Right 槽位都应该有检测结果
    assert (hand_frames.handedness[:, 0] == "Left").all()
    assert (hand_frames.handedness[:, 1] == "Right").all()


@requires_network
def test_detect_hands_slot_matches_handedness_label(hand_frames: Frames) -> None:
    detect_hands(hand_frames, show_progress=False)

    for label, slot in (("Left", 0), ("Right", 1)):
        detected = hand_frames.handedness[:, slot]
        assert all(v in (label, "") for v in detected)


@requires_network
def test_detect_hands_logs_summary(
    hand_frames: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        detect_hands(hand_frames, show_progress=False)

    assert f"{N_FRAMES}/{N_FRAMES}" in caplog.text  # 每帧都检测到左右手,应该是满分统计


@requires_network
def test_detect_hands_show_progress_true_prints_progress_bar(hand_frames: Frames, capsys) -> None:
    detect_hands(hand_frames, show_progress=True)

    assert "detect_hands" in capsys.readouterr().err  # tqdm 默认写到 stderr


@requires_network
def test_detect_hands_show_progress_false_prints_nothing(hand_frames: Frames, capsys) -> None:
    detect_hands(hand_frames, show_progress=False)

    assert capsys.readouterr().err == ""
