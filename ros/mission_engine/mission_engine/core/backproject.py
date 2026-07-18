"""World <-> image projection for the downward camera.

Optical frame: +x right (u), +y down (v), +z forward along the optical
axis. `project_raw` is the distortion-free pinhole projection ("raw"
pixel coordinates); `ground_point` is its inverse for detection ingest
(pixel ray intersected with the flat-ground plane). For a level nadir
camera `ground_point` reduces to the legacy IARC_mission_10 formula
offset = (norm - 0.5) * 2 * alt * tan(half_fov) per axis
(drones/scout_noah1_quinn_brady.py), which is the exact pinhole inverse
in that special case; the generalization adds attitude, camera tilt and
yaw, which that formula assumed away.
"""

from __future__ import annotations

import math
from typing import Tuple

from .config import CameraModel
from .geometry import Quat, Vec3, quat_conj, quat_rotate


class BehindCamera(Exception):
    """The point does not project: it lies on or behind the image plane."""


class AboveHorizon(Exception):
    """The pixel ray does not descend: it never meets the ground plane."""


def _cam_to_body(tilt_deg: float) -> Tuple[Vec3, Vec3, Vec3]:
    """Optical axes expressed in the FRD body frame (the columns of the
    camera->body rotation). tilt_deg = 0 looks straight down with image-up
    toward body-forward; positive tilt pitches the view forward."""
    c = math.cos(math.radians(tilt_deg))
    s = math.sin(math.radians(tilt_deg))
    x_body = (0.0, 1.0, 0.0)
    y_body = (-c, 0.0, s)
    z_body = (s, 0.0, c)
    return (x_body, y_body, z_body)


def project_raw(cam: CameraModel, pos: Vec3, q: Quat, point: Vec3) -> Tuple[float, float]:
    """Project a world (NED) point into raw pixel coordinates for a camera
    at `pos` on a body with attitude `q` (body->NED). Raises BehindCamera
    when the point is not in front of the camera; callers clip to the
    frame themselves (points outside the frame still project)."""
    d_ned = (point[0] - pos[0], point[1] - pos[1], point[2] - pos[2])
    d_body = quat_rotate(quat_conj(q), d_ned)
    x_ax, y_ax, z_ax = _cam_to_body(cam.tilt_deg)
    x_cam = x_ax[0] * d_body[0] + x_ax[1] * d_body[1] + x_ax[2] * d_body[2]
    y_cam = y_ax[0] * d_body[0] + y_ax[1] * d_body[1] + y_ax[2] * d_body[2]
    z_cam = z_ax[0] * d_body[0] + z_ax[1] * d_body[1] + z_ax[2] * d_body[2]
    if z_cam <= 0.0:
        raise BehindCamera(f"z_cam = {z_cam:.6f}")
    return (
        cam.cx + cam.focal_px * x_cam / z_cam,
        cam.cy + cam.focal_px * y_cam / z_cam,
    )


def ground_point(
    cam: CameraModel,
    pos: Vec3,
    q: Quat,
    pixel: Tuple[float, float],
    ground_z: float = 0.0,
) -> Tuple[float, float]:
    """Intersect a pixel's ray with the flat-ground plane z = ground_z;
    returns (north, east). Raises AboveHorizon when the ray does not
    descend toward the plane (grazing rays still intersect, arbitrarily
    far out — the depression-angle floor is the caller's policy)."""
    d_cam = (
        (pixel[0] - cam.cx) / cam.focal_px,
        (pixel[1] - cam.cy) / cam.focal_px,
        1.0,
    )
    x_ax, y_ax, z_ax = _cam_to_body(cam.tilt_deg)
    d_body = tuple(
        x_ax[i] * d_cam[0] + y_ax[i] * d_cam[1] + z_ax[i] * d_cam[2] for i in range(3)
    )
    d_ned = quat_rotate(q, d_body)
    if d_ned[2] <= 0.0:
        raise AboveHorizon(f"ray d_down = {d_ned[2]:.6f}")
    t = (ground_z - pos[2]) / d_ned[2]
    if t <= 0.0:
        raise AboveHorizon(f"camera below ground plane (t = {t:.6f})")
    return (pos[0] + t * d_ned[0], pos[1] + t * d_ned[1])
