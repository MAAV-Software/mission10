"""Pure core: deterministic, stdlib-only, unit-tested, replayable.

Conventions used throughout:
- world frame is local NED (north, east, down), metres; ground at z = 0
- yaw in radians, 0 = north, positive toward east
- quaternions are (w, x, y, z), unit, rotating FRD body vectors into NED
"""
