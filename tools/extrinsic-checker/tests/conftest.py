import json

import cv2
import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_a2d_h5(tmp_path):
    """Minimal a2d-like h5: head color+depth, head joint group, camera params."""
    path = tmp_path / "a2d_tiny.h5"
    vlen = h5py.vlen_dtype(np.uint8)

    def enc(img, ext):
        ok, buf = cv2.imencode(ext, img)
        assert ok
        return np.frombuffer(buf.tobytes(), np.uint8)

    params = {
        "intrinsic": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0},
        "extrinsic": {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      "translation_vector": [0, 0, 0]},
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("parameters/camera/head.json", data=json.dumps(params))
        cds = f.create_dataset("cameras/head/color/data", (2,), dtype=vlen)
        dds = f.create_dataset("cameras/head/depth/data", (2,), dtype=vlen)
        for i in range(2):
            cds[i] = enc((np.random.rand(48, 64, 3) * 255).astype(np.uint8), ".jpg")
            dds[i] = enc(np.full((48, 64), 700, np.uint16), ".png")
        hp = f.create_dataset("joints/state/head/position", (2, 2), dtype="float32")
        hp[:] = [[0.1, 0.2], [0.3, 0.4]]
    return str(path)
