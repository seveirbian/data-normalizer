"""把 Frames 里除视频帧之外的内容(元数据+检测结果)保存成 JSON。"""

import dataclasses
import json
import logging
import math
import os

import numpy as np

from tools.ego_pipe.frames import Frames

logger = logging.getLogger(__name__)


def write_annotations(frames: Frames, path: str) -> None:
    logger.info("开始写入 annotations -> %s", path)

    data = dataclasses.asdict(frames)
    del data["frames"]
    data = _to_jsonable(data)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

    logger.info("写入 annotations -> %s", path)


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
