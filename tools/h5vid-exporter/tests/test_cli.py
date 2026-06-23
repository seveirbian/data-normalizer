import os

from h5vid_exporter.cli import main


def test_cli_main_creates_video(tiny_h5, tmp_path):
    out = str(tmp_path / "cli.mp4")
    rc = main([
        "--input", tiny_h5,
        "--topics", "cameras/head/color", "cameras/head/depth",
        "--output", out,
        "--fps", "10",
        "--height", "120",
    ])
    assert rc == 0
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_cli_main_invalid_topic_returns_nonzero(tiny_h5, tmp_path, capsys):
    out = str(tmp_path / "cli.mp4")
    rc = main([
        "--input", tiny_h5,
        "--topics", "cameras/nope/color",
        "--output", out,
    ])
    assert rc != 0
    assert not os.path.exists(out)
