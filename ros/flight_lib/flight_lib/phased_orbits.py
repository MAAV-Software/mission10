"""Phased-orbits geometry — per-drone overlapping orbit circles.

The qualifier pattern (RFD `proof-of-intelligent-flight` §2-3): `n` drones, each
on its *own* circle, centers strung along a line and offset downrange, all of
radius `R`, all turning at the same rate `omega`, locked a constant `phase_step`
apart. Unlike :mod:`flight_lib.orbit` (one shared ring), the circles overlap so
flight paths intersect; collision is a timing problem the phase offset solves.

Not a single rigid hub: the centers are spread along a line (`spacing` apart), so
this is overlapping offset circles, not one wheel. The launch-relative frame
forbids a shared hub (it would need all drones to spawn at one point), so the
hub is sheared out along the spawn line.

Frame convention
----------------
Right-handed z-up world frame (x east, y north, z up; yaw CCW about +z from +x,
wrapped to (-pi, pi]). The consuming offboard node converts to PX4 NED.

Layout
------
Drone `i` orbits center::

    C_i = (base_x + spacing * i,  base_y + downrange)

so the centers are `spacing` apart along x (east), all pushed `downrange` along
+y (north). The near (line-side, -y) edge of each circle is its hover spot
`H_i = (base_x + spacing*i, base_y)`; motion bulges away from the operator.

Phase
-----
Drone `i` rides phase `phase0 + phase_step * i`. The default `phase0 = -pi/2`
puts drone 0 at the bottom (line-side "S") edge of its circle at `t = 0`.

Insertion
---------
:func:`phased_orbit_insertion` is the safe in-plane transition from a drone
hovering at its circle *center* onto its circle at phase `phi_i`: radius grows
`0 -> R` while the angle spins by `spin` (default +235 deg, validated to keep the
4-drone min separation ~2.09 m in-plane, vs collision for a pure-radial push).

Separation (closed form)
------------------------
For two equal-R, co-omega circles whose centers differ by `a = spacing*|i-j|`
along the line and whose phases differ by `dphi`, the separation vector is that
fixed offset plus a rotating vector of constant magnitude `2R*|sin(dphi/2)|`. So
the per-revolution distance sweeps `[|a - m|, a + m]` and::

    d_min(i, j) = | spacing*|i-j|  -  2*R*|sin(dphi / 2)| |

This is exact (no per-revolution numeric search). Two consequences the tests pin
as proof-of-intelligence properties, not just safety:

- `d_max - d_min = 2*min(a, m)` is the *swing* — a pair whose phases differ
  (`m > 0`) and whose circles overlap visibly opens and closes over each
  revolution. Zero swing = a frozen pair = choreography a judge can dismiss.
- overlapping circles (`spacing < 2R`) mean the paths genuinely cross, so the
  avoidance is real rather than trivial spacing.

:func:`min_pairwise_separation` is the brute-force cross-check the tests pin the
closed form against.
"""

from __future__ import annotations

import numpy as np

from flight_lib.orbit import _yaw_for

__all__ = [
    "phased_orbit_centers",
    "phased_orbit_phases",
    "phased_orbit_setpoint",
    "phased_orbit_velocity",
    "phased_orbit_positions",
    "phased_orbit_insertion",
    "phased_orbit_insertion_velocity",
    "phased_orbit_peeloff",
    "peeloff_duration",
    "separation_bounds",
    "pair_separation_bounds",
    "pair_min_separation",
    "pair_separation_swing",
    "schedule_min_separation",
    "min_pairwise_separation",
]

DEFAULT_PHASE_STEP = np.pi / 2.0
DEFAULT_PHASE0 = -np.pi / 2.0
DEFAULT_INSERTION_SPIN = np.radians(235.0)


def phased_orbit_phases(n, *, phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """Per-drone phase offsets as a length-`n` array.

    Uniform `phase0 + phase_step*i` unless an explicit `phases` sequence is given
    (used for non-uniform schedules like the conservative `[0, pi, pi, 0]`).
    """
    if phases is not None:
        out = np.asarray(phases, dtype=float)
        if out.shape != (n,):
            raise ValueError(f"phases must have length {n}, got {out.shape}")
        return out
    return phase0 + phase_step * np.arange(n)


def phased_orbit_centers(n, *, spacing=3.0, downrange=4.6, base=(0.0, 0.0)):
    """Orbit centers as an (n, 2) array of (x, y) in the z-up world frame."""
    if n < 1:
        raise ValueError("need at least 1 drone")
    bx, by = base
    i = np.arange(n)
    return np.column_stack([bx + spacing * i, np.full(n, by + downrange)])


def phased_orbit_setpoint(t, index, n, radius, omega, *,
                          spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                          phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None,
                          mod_amp=0.0, mod_phase=0.0,
                          yaw_mode="inward", fixed_yaw=0.0):
    """Position and yaw for drone `index` of `n` at time `t`.

    With `mod_amp` != 0 the angle carries a non-steady (time-varying speed)
    modulation ``mod_amp * (sin(ω t + mod_phase) − sin(mod_phase))``: the drone
    speeds up and slows along its circle instead of turning at constant rate.
    Per-drone (`mod_amp`, `mod_phase`) are chosen so the four phase off each
    other's path-crossings, lifting the orbit min separation above the
    constant-speed floor on the same circles and revolution count (asserted in
    test_modulation_lifts_min_separation). The ``− sin(mod_phase)`` term zeroes
    the modulation at ``t = 0``, so the orbit joins the insertion (which ends at
    the steady phase) without a jump. ``|mod_amp| < 1`` keeps the angular rate
    positive.

    Returns ``(position, yaw)`` where ``position`` is an (x, y, z) numpy array in
    the z-up world frame and ``yaw`` is a scalar in (-pi, pi].
    """
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    bx, by = base
    cx = bx + spacing * index
    cy = by + downrange
    phase = phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)[index]
    theta = (omega * t + phase
             + mod_amp * (np.sin(omega * t + mod_phase) - np.sin(mod_phase)))
    pos = np.array([
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        altitude,
    ])
    return pos, float(_yaw_for(theta, yaw_mode, fixed_yaw))


def phased_orbit_velocity(t, index, n, radius, omega, *,
                          spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                          phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None,
                          yaw_mode="inward", fixed_yaw=0.0):
    """Velocity of :func:`phased_orbit_setpoint` in the z-up world frame."""
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    phase = phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)[index]
    theta = omega * t + phase
    return np.array([-radius * omega * np.sin(theta), radius * omega * np.cos(theta), 0.0])


def phased_orbit_insertion(s, index, n, radius, *,
                           spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                           spin=DEFAULT_INSERTION_SPIN,
                           phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None,
                           yaw_mode="inward", fixed_yaw=0.0):
    """Spiral insertion for drone `index` at progress `s` in [0, 1].

    `s=0` is the circle center (radius 0); `s=1` is the orbit phase `phi_i` at
    radius `R`. Radius grows linearly while the angle spins from `phi_i - spin`
    to `phi_i`, so the path curls the same rotational way as the orbit. Returns
    ``(position, yaw)`` in the z-up world frame; call with `s` running 1 -> 0 for
    the reverse (orbit -> center) transition.
    """
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    u = float(np.clip(s, 0.0, 1.0))
    bx, by = base
    cx = bx + spacing * index
    cy = by + downrange
    phi = phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)[index]
    theta = phi - spin * (1.0 - u)
    r = radius * u
    pos = np.array([
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
        altitude,
    ])
    return pos, float(_yaw_for(theta, yaw_mode, fixed_yaw))


def phased_orbit_insertion_velocity(s, index, n, radius, *, s_dot,
                                    spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                                    spin=DEFAULT_INSERTION_SPIN,
                                    phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0,
                                    phases=None, yaw_mode="inward", fixed_yaw=0.0):
    """Velocity of :func:`phased_orbit_insertion` for progress rate ``s_dot``."""
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    u = float(np.clip(s, 0.0, 1.0))
    phi = phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)[index]
    theta = phi - spin * (1.0 - u)
    r = radius * u
    theta_dot = spin * float(s_dot)
    return np.array([
        radius * float(s_dot) * np.cos(theta) - r * np.sin(theta) * theta_dot,
        radius * float(s_dot) * np.sin(theta) + r * np.cos(theta) * theta_dot,
        0.0,
    ])


DEFAULT_PEEL_SPIN = np.pi / 2.0


def peeloff_duration(n, *, lead_in=0.5, stagger=3.0, peel_duration=4.0):
    """Total return time for the staggered peel-off of `n` drones."""
    return float(lead_in + (n - 1) * stagger + peel_duration)


def phased_orbit_peeloff(t, index, n, radius, omega, *,
                         peel_order=None, lead_in=0.5, stagger=3.0, peel_duration=4.0,
                         spin=DEFAULT_PEEL_SPIN,
                         spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                         phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None,
                         yaw_mode="inward", fixed_yaw=0.0):
    """Staggered peel-off return for drone `index` at return-relative time `t>=0`.

    Each drone keeps orbiting (CCW at the same `omega` as the orbit) until its
    peel time ``t_peel = lead_in + rank*stagger``, where `rank` is its position
    in `peel_order` (default ``n-1, ..., 0`` — highest index peels first). It
    then curls smoothly into its own circle center over `peel_duration`: radius
    `R -> 0` on a smoothstep while the angle advances a further `spin`.

    Peeling one drone at a time keeps the overlapping circles from colliding
    mid-collapse: only one drone crosses the centers-aligned zone at any instant,
    so the return clears the reverse-retrace floor (see
    test_peeloff_separation_beats_retrace). The motion never flips rotational
    sense.

    At `t=0` every drone is still on its orbit, so this joins
    :func:`phased_orbit_setpoint` when the orbit ends on an integer revolution.
    Returns ``(position, yaw)`` in the z-up world frame.
    """
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    order = list(range(n - 1, -1, -1)) if peel_order is None else list(peel_order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"peel_order must be a permutation of 0..{n - 1}, got {order}")
    rank = order.index(index)
    t_peel = lead_in + rank * stagger
    bx, by = base
    cx = bx + spacing * index
    cy = by + downrange
    phi = phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)[index]
    if t < t_peel:
        theta = phi + omega * t
        r = radius
    else:
        u = float(np.clip((t - t_peel) / max(1e-6, peel_duration), 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)  # smoothstep: zero rate at both ends
        theta = phi + omega * t_peel + spin * s
        r = radius * (1.0 - s)
    pos = np.array([cx + r * np.cos(theta), cy + r * np.sin(theta), altitude])
    return pos, float(_yaw_for(theta, yaw_mode, fixed_yaw))


def phased_orbit_positions(t, n, radius, omega, *,
                           spacing=3.0, downrange=4.6, base=(0.0, 0.0), altitude=0.0,
                           phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """All `n` drone positions at time `t` as an (n, 3) array (viz / checks)."""
    if n < 1:
        raise ValueError("need at least 1 drone")
    bx, by = base
    i = np.arange(n)
    theta = omega * t + phased_orbit_phases(n, phase_step=phase_step, phase0=phase0, phases=phases)
    return np.column_stack([
        bx + spacing * i + radius * np.cos(theta),
        by + downrange + radius * np.sin(theta),
        np.full(n, float(altitude)),
    ])


def separation_bounds(a, radius, dphi):
    """``(d_min, d_max)`` for two equal-R co-omega circles, centers `a` apart,
    phases `dphi` apart. The analytic core: a fixed offset `a` plus a rotating
    vector of magnitude `m = 2R|sin(dphi/2)|`, so distance sweeps
    ``[|a - m|, a + m]``."""
    m = 2.0 * radius * abs(np.sin(dphi / 2.0))
    return abs(a - m), a + m


def _pair_dphi(i, j, phase_step, phases):
    if phases is not None:
        ph = np.asarray(phases, dtype=float)
        return float(ph[i] - ph[j])
    return float(phase_step * (i - j))


def pair_separation_bounds(i, j, radius, *, spacing=3.0,
                           phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """``(d_min, d_max)`` over a revolution for the (i, j) pair (closed form).

    `phase0` is accepted for call-site symmetry; separation depends only on the
    phase *difference*, so it has no effect."""
    if i == j:
        raise ValueError("need two distinct drones for a separation")
    a = spacing * abs(i - j)
    dphi = _pair_dphi(i, j, phase_step, phases)
    return separation_bounds(a, radius, dphi)


def pair_min_separation(i, j, radius, *, spacing=3.0,
                        phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """Minimum separation over a revolution for the (i, j) pair (closed form)."""
    return pair_separation_bounds(
        i, j, radius, spacing=spacing, phase_step=phase_step, phase0=phase0, phases=phases)[0]


def pair_separation_swing(i, j, radius, *, spacing=3.0,
                          phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """``d_max - d_min`` for the (i, j) pair — how much the gap visibly opens and
    closes per revolution. Zero swing = a frozen pair (choreography); a positive
    swing is the temporal deconfliction a judge must be able to see."""
    lo, hi = pair_separation_bounds(
        i, j, radius, spacing=spacing, phase_step=phase_step, phase0=phase0, phases=phases)
    return hi - lo


def schedule_min_separation(n, radius, *, spacing=3.0,
                            phase_step=DEFAULT_PHASE_STEP, phase0=DEFAULT_PHASE0, phases=None):
    """Tightest `d_min` over all pairs — the schedule's separation budget."""
    if n < 2:
        raise ValueError("need at least 2 drones for a separation")
    return min(
        pair_min_separation(i, j, radius, spacing=spacing,
                            phase_step=phase_step, phase0=phase0, phases=phases)
        for i in range(n)
        for j in range(i + 1, n)
    )


def min_pairwise_separation(positions):
    """Smallest horizontal (xy) distance among rows of an (n, >=2) array.

    Brute force over all pairs at one instant — the numeric cross-check for the
    closed-form separations and the tool for evaluating arbitrary (e.g. spread
    transient) configurations the closed form does not cover.
    """
    p = np.asarray(positions, dtype=float)
    n = len(p)
    if n < 2:
        raise ValueError("need at least 2 positions for a separation")
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, float(np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])))
    return best
