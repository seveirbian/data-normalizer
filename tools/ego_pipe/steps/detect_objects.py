"""用 Grounding DINO(开集/文本提示物体检测)检测每帧里的物体,就地写回 Frames。

Grounding DINO 的能力与限制:
- 纯逐帧检测,不带任何时序/跟踪能力:每一帧的检测完全独立计算,同一个物理物体在
  相邻帧之间**没有**跨帧一致的 ID(不支持 ReID/多目标跟踪)。frame i 和 frame i+1
  里同一个物体各自有自己的 box/score,谁跟谁是"同一个东西"需要额外的跟踪算法
  (比如 ByteTrack/SORT,基于 box IoU + 运动预测做帧间关联)才能得到,这份实现不做。
- 只输出 bounding box,不做实例分割(没有 mask)。
- 开集检测依赖文本提示(labels),同一个物体用不同措辞提示,召回/准确率可能不同;
  `box_threshold`/`text_threshold` 需要按实际场景调。

曾经尝试过批处理(多帧一起过模型)加速,后来回退了,原因是实测收益不划算:
- 1920x1080 图片、32 帧实测,batch_size 从 1 提到 8,每帧总耗时只从 277ms 降到
  232ms(~16%),而且 batch=4 之后基本不再下降。
- 瓶颈其实是图片预处理(resize+转 tensor,CPU 操作),固定在 ~90ms/帧,不受批处理
  影响;批处理只加速了模型前向那部分,批量越大预处理占比反而越高,收益迅速见顶。
- 而且实际运行时 GPU 经常被其他进程占用大半显存,可用 batch_size 天花板比理论值
  低很多(实测 batch_size=16 直接 CUDA OOM)。
真要大幅提速,得从预处理入手(比如降低输入分辨率),而不是批处理;先维持逐帧处理,
换取代码简单。
"""

import logging

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.utils import logging as hf_logging

from tools.ego_pipe.frames import Frames

logger = logging.getLogger(__name__)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_LABELS = ["hand", "tool", "part", "box", "wire", "bottle", "cup"]


def detect_objects(
    frames: Frames,
    labels: list[str] = DEFAULT_LABELS,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
    show_progress: bool = True,
) -> None:
    text = ". ".join(labels) + "."
    if not show_progress:
        hf_logging.disable_progress_bar()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    finally:
        if not show_progress:
            hf_logging.enable_progress_bar()

    detected_objects: list[list[dict]] = []
    for i in tqdm(
        range(frames.n_frames), desc="detect_objects", unit="frame", disable=not show_progress
    ):
        image = Image.fromarray(frames.frames[i])
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(frames.height, frames.width)],
        )[0]

        frame_objects = [
            {"box": tuple(box.tolist()), "label": label, "score": float(score)}
            for box, label, score in zip(
                result["boxes"], result["text_labels"], result["scores"]
            )
        ]
        detected_objects.append(frame_objects)

    frames.detected_objects = detected_objects
    n_detections = sum(len(objs) for objs in detected_objects)
    logger.info("%d 帧中共检测到 %d 个物体", frames.n_frames, n_detections)
