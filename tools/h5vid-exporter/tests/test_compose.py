import numpy as np
import pytest

from h5vid_exporter.compose import compose_row


def test_compose_row_hstacks_same_height():
    a = np.zeros((120, 80, 3), np.uint8)
    b = np.zeros((120, 100, 3), np.uint8)
    out = compose_row([a, b])
    assert out.shape == (120, 180, 3)


def test_compose_row_single_tile():
    a = np.zeros((120, 80, 3), np.uint8)
    out = compose_row([a])
    assert out.shape == (120, 80, 3)


def test_compose_row_differing_heights_raises():
    a = np.zeros((120, 80, 3), np.uint8)
    b = np.zeros((100, 80, 3), np.uint8)
    with pytest.raises(ValueError):
        compose_row([a, b])


def test_compose_row_empty_raises():
    with pytest.raises(ValueError):
        compose_row([])
