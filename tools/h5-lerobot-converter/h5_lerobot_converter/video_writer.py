import subprocess
from pathlib import Path
import numpy as np

_JPEG_MAGIC = b'\xff\xd8\xff'


class VideoWriter:
    def write(self, frames: list, output_path: Path, fps: float = 30.0) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            return

        if frames[0][:3] == _JPEG_MAGIC:
            self._write_jpeg_pipe(frames, output_path, fps)
        else:
            self._write_decoded_pipe(frames, output_path, fps)

    def _write_jpeg_pipe(self, frames: list, output_path: Path, fps: float) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg",
            "-r", str(fps), "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-bf", "0", "-g", "1",
            str(output_path),
        ]
        self._run_ffmpeg(cmd, frames)

    def _write_decoded_pipe(self, frames: list, output_path: Path, fps: float) -> None:
        import cv2

        first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise RuntimeError(f"Cannot decode first frame for {output_path}")

        h, w = first.shape[:2]
        is_depth = first.ndim == 2

        if is_depth:
            pix_fmt = "gray"
            def to_raw(raw_bytes):
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                return (img.astype(np.uint32) >> 8).astype(np.uint8).tobytes()
        else:
            pix_fmt = "rgb24"
            def to_raw(raw_bytes):
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", pix_fmt,
            "-r", str(fps), "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-bf", "0", "-g", "1",
            str(output_path),
        ]
        self._run_ffmpeg(cmd, frames, encoder=to_raw)

    def _run_ffmpeg(self, cmd: list, frames: list, encoder=None) -> None:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames:
            proc.stdin.write(encoder(frame) if encoder else frame)
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (returncode={proc.returncode}): {cmd[-1]}")
