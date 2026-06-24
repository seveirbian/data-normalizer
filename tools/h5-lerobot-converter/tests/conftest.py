import io
import json
import numpy as np
import pytest
import h5py
import cv2


def make_jpeg_bytes(h: int = 32, w: int = 48) -> bytes:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


@pytest.fixture
def mini_h5(tmp_path):
    """10帧的最小合法 A2D HDF5 文件。"""
    n = 10
    path = tmp_path / "test_ep.h5"

    with h5py.File(path, "w") as f:
        vlen = h5py.vlen_dtype(np.uint8)
        for cam_path in [
            "cameras/hand_left/color/data",
            "cameras/hand_right/color/data",
            "cameras/head/color/data",
            "cameras/head/depth/data",
        ]:
            ds = f.create_dataset(cam_path, (n,), dtype=vlen)
            for i in range(n):
                raw = make_jpeg_bytes()
                ds[i] = np.frombuffer(raw, dtype=np.uint8)

        f.create_dataset("joints/state/arm/position",      data=np.random.rand(n, 14))
        f.create_dataset("joints/state/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("joints/state/head/position",     data=np.random.rand(n, 2))
        f.create_dataset("joints/state/waist/position",    data=np.random.rand(n, 2))
        f.create_dataset("joints/state/robot/position",    data=np.random.rand(n, 3))
        f.create_dataset("joints/state/robot/orientation", data=np.random.rand(n, 4))

        f.create_dataset("joints/action/arm/position",      data=np.random.rand(n, 14))
        f.create_dataset("joints/action/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("joints/action/head/position",     data=np.random.rand(n, 2))
        f.create_dataset("joints/action/waist/position",    data=np.random.rand(n, 2))
        f.create_dataset("joints/action/robot/position",    data=np.random.rand(n, 3))
        f.create_dataset("joints/action/robot/velocity",    data=np.random.rand(n, 2))

        f.create_dataset("timestamp", data=np.arange(n, dtype=np.int64) * 33_333_333)

        meta = {"ver": "2.1.0", "time_align_info": {"fps": 30.0, "frame_count": n}}
        f.create_dataset("metadata.json", data=json.dumps(meta))

    return path


@pytest.fixture
def r1_mini_h5(tmp_path):
    """10帧最小合法 R1 HDF5 文件。"""
    n = 10
    path = tmp_path / "test_r1_ep.h5"

    with h5py.File(path, "w") as f:
        vlen = h5py.vlen_dtype(np.uint8)
        for cam_path in [
            "cameras/hand_left/color/data",
            "cameras/hand_left/depth/data",
            "cameras/hand_right/color/data",
            "cameras/hand_right/depth/data",
            "cameras/head/color/data",
        ]:
            ds = f.create_dataset(cam_path, (n,), dtype=vlen)
            for i in range(n):
                raw = make_jpeg_bytes()
                ds[i] = np.frombuffer(raw, dtype=np.uint8)

        f.create_dataset("joints/state/arm/position",       data=np.random.rand(n, 12))
        f.create_dataset("joints/state/arm/velocity",       data=np.random.rand(n, 12))
        f.create_dataset("joints/state/effector/position",  data=np.random.rand(n, 2))
        f.create_dataset("joints/action/arm/position",      data=np.random.rand(n, 12))
        f.create_dataset("joints/action/arm/velocity",      data=np.random.rand(n, 12))
        f.create_dataset("joints/action/effector/position", data=np.random.rand(n, 2))
        f.create_dataset("timestamp", data=np.arange(n, dtype=np.int64) * 33_333_333)

        meta = {"ver": "2.1.0", "equipment_info": {"manufacturer": "星海图", "model": "R1"},
                "time_align_info": {"fps": 30.0, "frame_count": n}}
        f.create_dataset("metadata.json", data=json.dumps(meta))

    return path
