"""示例:读一段真实 ego 视频,跑物体检测,把结果叠加输出成新视频,同时保存 annotations。

运行(从仓库根): uv run -m tools.ego_pipe.examples.demo_hand_overlay
"""

import os

from tools.ego_pipe.io.read_video import read_video
from tools.ego_pipe.io.write_annotations import write_annotations
from tools.ego_pipe.io.write_video import write_video
from tools.ego_pipe.log import configure_logging
from tools.ego_pipe.pipe.pipeline import Pipeline
from tools.ego_pipe.steps.detect_objects import detect_objects
from tools.ego_pipe.steps.detect_hands import detect_hands

VIDEO_PATH = (
    "/mnt/sfs_turbo/datasets-00573080/raw/Kupas-lerobot/paxini/Ego单目/Ego单目/"
    "Ego-Monocular/Factory/MonocularInspection/Task_001_LED_透镜灯泡安装/videos/"
    "observation.images.head/chunk-000/file-000.mp4"
)
MAX_FRAMES = 125  # 25fps 下约 5 秒
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "hand_overlay.mp4")
ANNOTATIONS_PATH = os.path.join(os.path.dirname(__file__), "output", "hand_overlay.json")


def main() -> None:
    configure_logging()

    frames = read_video(VIDEO_PATH)
    Pipeline([detect_hands, detect_objects]).run(frames)
    write_video(frames, OUTPUT_PATH, draw_hands=True, draw_objects=True)
    write_annotations(frames, ANNOTATIONS_PATH)


if __name__ == "__main__":
    main()
