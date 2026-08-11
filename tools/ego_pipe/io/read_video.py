"""读取一个视频文件(mp4 等,含 AV1 等 opencv 自带 ffmpeg 解不了的编码),解码全部帧到内存,产出 Frames。

用 pyav(直接绑定 ffmpeg 库)而不是 cv2.VideoCapture,因为 opencv-python 自带编译的 ffmpeg
不支持 AV1 解码,而 pyav 的 wheel 打包了更完整的 ffmpeg。
"""

import logging

import av
import numpy as np
from tqdm import tqdm

from tools.ego_pipe.frames import Frames

logger = logging.getLogger(__name__)


def read_video(path: str, max_frames: int | None = None, show_progress: bool = True) -> Frames:
    logger.info("开始读取: %s", path)

    try:
        container = av.open(path)
    except OSError as e:
        raise ValueError(f"cannot open video: {path}") from e

    stream = container.streams.video[0]
    width = stream.width
    height = stream.height
    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    fourcc = stream.codec_context.codec_tag

    frames = []
    total = max_frames if max_frames is not None else stream.frames or None
    progress = tqdm(desc="read_video", total=total, unit="frame", disable=not show_progress)
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        progress.update(1)
        if max_frames is not None and len(frames) >= max_frames:
            break
    progress.close()
    container.close()

    n_frames = len(frames)
    result = Frames(
        frames=np.stack(frames) if frames else np.empty((0, height, width, 3), dtype=np.uint8),
        path=path,
        fps=fps,
        width=width,
        height=height,
        n_frames=n_frames,
        duration_sec=n_frames / fps if fps else 0.0,
        fourcc=fourcc,
    )
    logger.info("%s -> %d frames, %dx%d, %.2f fps", path, n_frames, width, height, fps)
    return result
