"""Read camera image-stream topics from a guodi-style h5 file.

A "topic" is a group path (e.g. ``cameras/head/color``) that directly contains
a ``data`` dataset of shape ``(num_frames,)`` holding encoded jpg/png bytes.
"""

import h5py
import numpy as np


class H5Reader:
    def __init__(self, path):
        self._f = h5py.File(path, "r")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def available_camera_topics(self):
        """Sorted list of topic paths: groups that directly contain a 'data' dataset."""
        topics = []

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "data" in obj:
                child = obj["data"]
                if isinstance(child, h5py.Dataset):
                    topics.append(name)

        self._f.visititems(visit)
        return sorted(topics)

    def _is_topic(self, topic):
        grp = self._f.get(topic)
        return (
            isinstance(grp, h5py.Group)
            and "data" in grp
            and isinstance(grp["data"], h5py.Dataset)
        )

    def validate_topic(self, topic):
        if not self._is_topic(topic):
            raise ValueError(
                f"Topic {topic!r} is not a valid image-stream topic. "
                f"Available topics: {self.available_camera_topics()}"
            )

    def frame_count(self, topic):
        return int(self._f[topic]["data"].shape[0])

    def frame_bytes(self, topic, index):
        elem = self._f[topic]["data"][index]
        if isinstance(elem, bytes):
            return elem
        if isinstance(elem, np.ndarray):
            return elem.tobytes()
        return bytes(elem)
