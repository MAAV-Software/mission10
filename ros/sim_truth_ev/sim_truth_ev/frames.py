from __future__ import annotations

import math


Q_ENU_TO_NED = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
Q_FLU_TO_FRD = (0.0, 1.0, 0.0, 0.0)


def enu_vector_to_ned(v):
    return (float(v[1]), float(v[0]), -float(v[2]))


def flu_vector_to_frd(v):
    return (float(v[0]), -float(v[1]), -float(v[2]))


def q_normalize(q):
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(x) / n for x in q)


def q_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def ros_enu_flu_to_px4_ned_frd(q_xyzw):
    q_enu_flu = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
    return q_normalize(q_multiply(q_multiply(Q_ENU_TO_NED, q_enu_flu), Q_FLU_TO_FRD))
