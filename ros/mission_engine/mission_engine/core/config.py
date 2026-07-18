"""Camera model — the single source of the downward camera's geometry.

The YOLO datagen imports this same model to stamp training labels, so a
change here (tilt, FOV, resolution) is a dataset regen and the physical
mount must match. Defaults are the Camera Module 2 in the 2x2-binned
survey mode (sensing RFD, camera capture mode section).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    width_px: int = 1640
    height_px: int = 1232
    hfov_deg: float = 62.2  # CM2 datasheet; square pixels give the 48.8 vfov
    # Optical-axis pitch from nadir toward body-forward. The physical CM2
    # mount is nadir (decision closed 2026-07-17); datagen randomizes tilt
    # per scene on top of this for viewpoint diversity.
    tilt_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError(f"bad resolution {self.width_px}x{self.height_px}")
        if not (0.0 < self.hfov_deg < 180.0):
            raise ValueError(f"bad hfov_deg {self.hfov_deg}")

    @property
    def focal_px(self) -> float:
        """Pinhole focal length in pixels (square pixels assumed)."""
        return (self.width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def cx(self) -> float:
        return self.width_px / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0
