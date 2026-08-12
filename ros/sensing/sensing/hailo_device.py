"""One process-wide Hailo VDevice shared by every inference backend.

A VDevice takes the physical Hailo-8 exclusively per process, so the mine
detector and the depth monitor must configure their HEFs on the same handle.
The HailoRT scheduler then timeshares the device between the resident models.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_vdevice = None


def shared_vdevice():
    global _vdevice
    with _lock:
        if _vdevice is None:
            from hailo_platform import HailoSchedulingAlgorithm, VDevice

            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            _vdevice = VDevice(params)
        return _vdevice
