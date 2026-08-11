"""用 Grounding DINO(开集/文本提示物体检测)检测每帧里的物体,就地写回 Frames。"""

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
