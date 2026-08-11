import pytest

from tools.ego_pipe.frames import Frames
from tools.ego_pipe.io.read_video import read_video
from tools.ego_pipe.io.tests.conftest import FPS, HEIGHT, N_FRAMES, WIDTH


def test_read_video_returns_correct_shape_and_metadata(tiny_video: str) -> None:
    result: Frames = read_video(tiny_video, show_progress=False)

    assert result.frames.shape == (N_FRAMES, HEIGHT, WIDTH, 3)
    assert result.width == WIDTH
    assert result.height == HEIGHT
    assert result.fps == FPS
    assert result.path == tiny_video


def test_read_video_converts_bgr_to_rgb(tiny_video: str) -> None:
    result = read_video(tiny_video, show_progress=False)

    first_frame = result.frames[0]  # 源帧是 BGR (0, 0, 255) = 纯红
    assert first_frame[..., 0].mean() > 200  # R 通道应该是主导通道
    assert first_frame[..., 2].mean() < 50  # B 通道应该很暗


def test_read_video_n_frames_matches_actual_decoded_count(tiny_video: str) -> None:
    result = read_video(tiny_video, show_progress=False)

    assert result.n_frames == N_FRAMES
    assert result.n_frames == len(result.frames)
    assert result.duration_sec == N_FRAMES / FPS


def test_read_video_missing_file_raises_value_error() -> None:
    with pytest.raises(ValueError, match="does-not-exist.mp4"):
        read_video("/tmp/does-not-exist.mp4", show_progress=False)


def test_read_video_reports_fourcc(tiny_video: str) -> None:
    result = read_video(tiny_video, show_progress=False)

    assert len(result.fourcc) == 4
    assert result.fourcc.strip("\x00").isprintable()


def test_read_video_max_frames_truncates(tiny_video: str) -> None:
    result = read_video(tiny_video, max_frames=3, show_progress=False)

    assert result.n_frames == 3
    assert result.frames.shape == (3, HEIGHT, WIDTH, 3)
    assert result.duration_sec == 3 / FPS


def test_read_video_max_frames_none_reads_all(tiny_video: str) -> None:
    result = read_video(tiny_video, max_frames=None, show_progress=False)

    assert result.n_frames == N_FRAMES


def test_read_video_max_frames_larger_than_video_reads_all(tiny_video: str) -> None:
    result = read_video(tiny_video, max_frames=1000, show_progress=False)

    assert result.n_frames == N_FRAMES


def test_read_video_logs_summary(tiny_video: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        result = read_video(tiny_video, show_progress=False)

    assert str(result.n_frames) in caplog.text
    assert str(result.width) in caplog.text
    assert str(result.height) in caplog.text


def test_read_video_logs_entry_before_decoding(
    tiny_video: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        read_video(tiny_video, show_progress=False)

    assert len(caplog.records) >= 2
    assert tiny_video in caplog.records[0].getMessage()


def test_read_video_show_progress_true_prints_progress_bar(tiny_video: str, capsys) -> None:
    read_video(tiny_video, show_progress=True)

    assert "read_video" in capsys.readouterr().err  # tqdm 默认写到 stderr


def test_read_video_show_progress_false_prints_nothing(tiny_video: str, capsys) -> None:
    read_video(tiny_video, show_progress=False)

    assert capsys.readouterr().err == ""
