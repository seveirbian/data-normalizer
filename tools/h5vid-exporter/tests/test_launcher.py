import os
import subprocess
import sys

LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "h5vid-export.py")


def test_launcher_runs_from_unrelated_cwd(tiny_h5, tmp_path):
    """run.py must work from any cwd without PYTHONPATH or an install."""
    out = str(tmp_path / "launch.mp4")
    result = subprocess.run(
        [sys.executable, LAUNCHER,
         "--input", tiny_h5,
         "--topics", "cameras/head/color",
         "--output", out,
         "--fps", "10", "--height", "120"],
        cwd=str(tmp_path),  # deliberately unrelated working directory
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out) and os.path.getsize(out) > 0
