"""Shared marker builders for the gz overlay scripts (via ros_gz_marker_bridge).

Line markers (LINE_STRIP/LINE_LIST) must NOT set scale: a zero scale axis
collapses the marker in gz, and gz ignores line width anyway (its default
scale 1,1,1 renders the line fine). Solids (CYLINDER/SPHERE) set all three.
Marker id 0 is special in gz (auto-assigned a new id each message ->
accumulation), so all ids must be >= 1.
"""
import math

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

# per-drone RGB: red, green, blue, amber
COLORS = [(0.90, 0.15, 0.15), (0.15, 0.85, 0.25), (0.20, 0.45, 0.95), (0.95, 0.75, 0.10)]


def col(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def quat_z_to(v):
    """Quaternion (x,y,z,w) rotating local +Z onto direction v."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    d = np.asarray(v, float) / n
    c = float(d[2])
    if c > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if c < -1.0 + 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross([0.0, 0.0, 1.0], d)
    axis /= np.linalg.norm(axis)
    a = math.acos(c)
    s = math.sin(a / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(a / 2.0))


def line(ns, mid, pts, rgb, kind=Marker.LINE_STRIP, life=None):
    """Polyline (no scale -> gz default; line width is not controllable)."""
    m = Marker()
    m.ns = ns; m.id = int(mid); m.action = Marker.ADD; m.type = kind
    m.pose.orientation.w = 1.0; m.color = col(rgb)
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
    if life is not None:
        m.lifetime = life
    return m


def rod(ns, mid, a, b, rgb, dia=0.03, life=None):
    """Thin cylinder from a to b (a directional, visible segment)."""
    a, b = np.array(a, float), np.array(b, float)
    v = b - a
    m = Marker()
    m.ns = ns; m.id = int(mid); m.action = Marker.ADD; m.type = Marker.CYLINDER
    mid_pt = (a + b) / 2.0
    qx, qy, qz, qw = quat_z_to(v)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, mid_pt)
    m.pose.orientation.x, m.pose.orientation.y = qx, qy
    m.pose.orientation.z, m.pose.orientation.w = qz, qw
    m.scale.x = m.scale.y = dia
    m.scale.z = max(float(np.linalg.norm(v)), 1e-3)
    m.color = col(rgb)
    if life is not None:
        m.lifetime = life
    return m


def text(ns, mid, p, rgb, txt, h=0.6, life=None):
    m = Marker()
    m.ns = ns; m.id = int(mid); m.action = Marker.ADD; m.type = Marker.TEXT_VIEW_FACING
    m.text = txt; m.color = col(rgb); m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = h  # all 3 set so the bridge forwards it
    m.pose.position.x, m.pose.position.y = float(p[0]), float(p[1])
    m.pose.position.z = float(p[2]) + 0.8
    if life is not None:
        m.lifetime = life
    return m
