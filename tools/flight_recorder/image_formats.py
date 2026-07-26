"""Raw camera-buffer contracts shared by the flight recorder and its tests."""

from __future__ import annotations

import numpy as np


YUYV_ENCODING = "yuyv"
YUYV_BYTES_PER_PIXEL = 2


def pack_yuyv_frame(frame: np.ndarray, width: int, height: int) -> bytes:
    """Return a tightly packed YUYV 4:2:2 frame without libcamera row padding.

    Picamera2 represents packed YUYV as ``(height, stride // 2, 2)``. The
    configured stride can exceed ``width * 2``. ROS ``sensor_msgs/Image`` uses
    a tight ``step = width * 2`` here, so remove only the padded columns.
    """
    if frame.dtype != np.uint8:
        raise ValueError(f"YUYV frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[0] != height or frame.shape[2] != 2:
        raise ValueError(
            "YUYV frame shape must be "
            f"({height}, padded_width>=width, 2), got {frame.shape}"
        )
    if frame.shape[1] < width:
        raise ValueError(
            f"YUYV frame width {frame.shape[1]} is smaller than requested {width}"
        )
    return np.ascontiguousarray(frame[:, :width, :]).tobytes()
