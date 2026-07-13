import math

import numpy as np
import pytest

from flight_lib.deconflict import (
    closest_point_of_approach,
    follower_phase_rate,
    phase_deconflict_rate,
    reflex_velocity,
)


def test_cpa_head_on():
    t, d = closest_point_of_approach([4.0, 0.0, 0.0], [-2.0, 0.0, 0.0])
    assert t == pytest.approx(2.0)
    assert d == pytest.approx(0.0)


def test_follower_phase_rate_corrects_ahead_error_and_clamps():
    rate, error = follower_phase_rate(0.5, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert error == pytest.approx(0.5)
    assert 0.7 <= rate < 1.0
    rate, _ = follower_phase_rate(-2.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert rate == pytest.approx(1.3)


def test_reflex_slows_veers_right_and_repels():
    output = reflex_velocity([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0)
    assert output[0] < 1.2
    assert output[1] < 0.0
    assert output[2] == pytest.approx(0.0)


def test_phase_deconflict_no_conflict_passthrough():
    rate, sev = phase_deconflict_rate(0.5, 0.0, 0.435, t_cpa=10.0, d_cpa=0.5,
                                      vehicle_id=0, peer_id=1)
    assert rate == pytest.approx(0.435)
    assert sev == 0.0
    rate, sev = phase_deconflict_rate(0.5, 0.0, 0.435, t_cpa=1.0, d_cpa=5.0,
                                      vehicle_id=0, peer_id=1)
    assert rate == pytest.approx(0.435)
    assert sev == 0.0


def test_phase_deconflict_ahead_speeds_up():
    rate, sev = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=0.0, d_cpa=0.0,
                                      vehicle_id=0, peer_id=1)
    assert rate > 1.0
    assert sev == pytest.approx(1.0)


def test_phase_deconflict_behind_slows_down():
    rate, sev = phase_deconflict_rate(-0.5, 0.0, 1.0, t_cpa=0.0, d_cpa=0.0,
                                      vehicle_id=1, peer_id=0)
    assert rate < 1.0
    assert sev == pytest.approx(1.0)


def test_phase_deconflict_reciprocal_opposite_bias():
    rate_a, _ = phase_deconflict_rate(0.5, -0.5, 1.0, t_cpa=1.0, d_cpa=1.0,
                                      vehicle_id=0, peer_id=1)
    rate_b, _ = phase_deconflict_rate(-0.5, 0.5, 1.0, t_cpa=1.0, d_cpa=1.0,
                                      vehicle_id=1, peer_id=0)
    assert rate_a > 1.0 > rate_b


def test_phase_deconflict_tiebreak_on_id():
    rate_lo, _ = phase_deconflict_rate(0.0, 0.0, 1.0, t_cpa=0.0, d_cpa=0.0,
                                       vehicle_id=0, peer_id=1)
    rate_hi, _ = phase_deconflict_rate(0.0, 0.0, 1.0, t_cpa=0.0, d_cpa=0.0,
                                       vehicle_id=1, peer_id=0)
    assert rate_lo > 1.0 > rate_hi


def test_phase_deconflict_clamps_to_max_scale():
    rate, _ = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=0.0, d_cpa=0.0,
                                    vehicle_id=0, peer_id=1, max_scale=1.5)
    assert rate == pytest.approx(1.5)


def test_phase_deconflict_severity_scales_with_depth_and_time():
    _, deep = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=1.0, d_cpa=0.5,
                                    vehicle_id=0, peer_id=1)
    _, shallow = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=1.0, d_cpa=2.0,
                                       vehicle_id=0, peer_id=1)
    assert deep > shallow
    _, soon = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=0.5, d_cpa=1.0,
                                    vehicle_id=0, peer_id=1)
    _, late = phase_deconflict_rate(0.5, 0.0, 1.0, t_cpa=3.0, d_cpa=1.0,
                                    vehicle_id=0, peer_id=1)
    assert soon > late
