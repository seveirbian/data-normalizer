import pytest

from h5vid_exporter.reader import H5Reader


def test_available_camera_topics(tiny_h5):
    with H5Reader(tiny_h5) as r:
        assert r.available_camera_topics() == [
            "cameras/head/color",
            "cameras/head/depth",
        ]


def test_validate_topic_ok(tiny_h5):
    with H5Reader(tiny_h5) as r:
        r.validate_topic("cameras/head/color")  # no raise


def test_validate_topic_missing_lists_available(tiny_h5):
    with H5Reader(tiny_h5) as r:
        with pytest.raises(ValueError) as exc:
            r.validate_topic("cameras/nope/color")
    msg = str(exc.value)
    assert "cameras/nope/color" in msg
    assert "cameras/head/color" in msg


def test_validate_topic_group_without_data(tiny_h5):
    with H5Reader(tiny_h5) as r:
        with pytest.raises(ValueError):
            r.validate_topic("cameras/head")  # group, but no /data child


def test_frame_count(tiny_h5):
    with H5Reader(tiny_h5) as r:
        assert r.frame_count("cameras/head/color") == 3


def test_frame_bytes_decodable(tiny_h5):
    import numpy as np
    import cv2
    with H5Reader(tiny_h5) as r:
        b = r.frame_bytes("cameras/head/color", 0)
    assert isinstance(b, bytes) and len(b) > 0
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape == (48, 64, 3)
