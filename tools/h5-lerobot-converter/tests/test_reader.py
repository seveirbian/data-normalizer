import numpy as np
from h5_lerobot_converter.reader import HDF5Reader


def test_read_state_shape(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.state.shape == (10, 27), f"got {data.state.shape}"
    assert data.state.dtype == np.float32


def test_read_action_shape(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.action.shape == (10, 25), f"got {data.action.shape}"
    assert data.action.dtype == np.float32


def test_read_timestamps_seconds(mini_h5):
    data = HDF5Reader().read(mini_h5)
    assert data.timestamps.shape == (10,)
    assert abs(data.timestamps[0]) < 1e-6
    assert abs(data.timestamps[1] - 1 / 30) < 1e-4


def test_read_images_four_cameras(mini_h5):
    data = HDF5Reader().read(mini_h5)
    expected_keys = {
        "observation.images.hand_left",
        "observation.images.hand_right",
        "observation.images.head",
        "observation.images.head_depth",
    }
    assert set(data.images.keys()) == expected_keys
    for key, frames in data.images.items():
        assert len(frames) == 10, f"{key}: expected 10 frames"
        assert isinstance(frames[0], bytes), f"{key}: frame should be bytes"
        assert len(frames[0]) > 100, f"{key}: frame too small"
