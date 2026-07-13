"""flight_lib — pure-math flight geometry (orbit + hover/land). No ROS, no PX4."""

from flight_lib.orbit import (
    min_separation,
    orbit_setpoint,
    radius_for_separation,
    ring_positions,
    wrap_angle,
)
from flight_lib.phased_orbits import (
    min_pairwise_separation,
    pair_min_separation,
    pair_separation_bounds,
    pair_separation_swing,
    phased_orbit_centers,
    phased_orbit_insertion,
    phased_orbit_insertion_velocity,
    phased_orbit_peeloff,
    phased_orbit_phases,
    peeloff_duration,
    phased_orbit_positions,
    phased_orbit_setpoint,
    phased_orbit_velocity,
    separation_bounds,
    schedule_min_separation,
)
from flight_lib.profiles import hover_setpoint, land_duration, land_setpoint
from flight_lib.deconflict import (
    closest_point_of_approach,
    follower_phase_rate,
    phase_deconflict_rate,
    reflex_velocity,
)
from flight_lib.bvc import (
    buffered_uncertainty_halfspaces,
    buffered_uncertainty_voronoi_clip,
    buffered_voronoi_clip,
    buffered_voronoi_halfspaces,
    normal_uncertainty_buffer,
)
from flight_lib.rel_localization import RelativePositionEKF, gnss_common_mode

__all__ = [
    "min_separation",
    "radius_for_separation",
    "orbit_setpoint",
    "ring_positions",
    "wrap_angle",
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
    "hover_setpoint",
    "land_setpoint",
    "land_duration",
    "closest_point_of_approach",
    "follower_phase_rate",
    "phase_deconflict_rate",
    "reflex_velocity",
    "buffered_voronoi_clip",
    "buffered_voronoi_halfspaces",
    "buffered_uncertainty_voronoi_clip",
    "buffered_uncertainty_halfspaces",
    "normal_uncertainty_buffer",
    "RelativePositionEKF",
    "gnss_common_mode",
]
