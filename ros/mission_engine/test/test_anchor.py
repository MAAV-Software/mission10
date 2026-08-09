import math

import pytest

from mission_engine.core.anchor import (
    AnchorConfig,
    AnchorTransform,
    CorrectionConfig,
    SetpointCorrection,
    TagAnchorMap,
)


def transform(dn, de, dyaw=0.0, points=((0.0, 0.0),)):
    return AnchorTransform(dn=dn, de=de, dyaw=dyaw, n_tags=len(points), points=points)


def settle(anchor, tag="tag36h11:6", fix=(10.0, 5.0), t0=0.0, n=None):
    """Feed a tight burst until the tag's datum is accepted."""
    n = n if n is not None else anchor.cfg.settle_obs
    for i in range(n):
        anchor.observe(t0 + 0.1 * i, tag, fix, fix, 4.0)
    return anchor.refs[tag]


def test_reference_settles_and_residual_is_zero_at_the_datum():
    a = TagAnchorMap()
    ref = settle(a)
    assert ref.settled
    assert ref.origin == pytest.approx((10.0, 5.0))
    fx = a.observe(1.0, "tag36h11:6", (10.0, 5.0), (10.0, 5.0), 4.0)
    assert fx.settled
    assert fx.residual_m == pytest.approx(0.0)


def test_residual_reports_the_drift_of_the_frame():
    a = TagAnchorMap()
    settle(a)
    fx = a.observe(1.0, "tag36h11:6", (12.0, 5.0), (12.0, 5.0), 4.0)
    assert fx.residual == pytest.approx((2.0, 0.0))
    d = a.drift(1.0)
    assert (d.dn, d.de) == pytest.approx((2.0, 0.0))
    assert d.dyaw == 0.0 and d.n_tags == 1


def test_a_smeared_settling_window_is_not_frozen_into_a_datum():
    """Drift during settling must not silently become the datum."""
    a = TagAnchorMap(AnchorConfig(settle_obs=4, settle_spread_m=0.2))
    for i in range(4):
        a.observe(0.1 * i, "t", (10.0 + i, 5.0), (10.0 + i, 5.0), 4.0)
    assert not a.refs["t"].settled
    assert a.drift(0.4) is None


def test_gates_reject_grazing_and_out_of_band_sightings():
    a = TagAnchorMap(AnchorConfig(max_radial_m=2.0, min_agl_m=0.5, max_agl_m=6.0))
    assert a.observe(0.0, "t", (14.0, 5.0), (10.0, 5.0), 4.0) is None  # 4 m radial
    assert a.observe(0.0, "t", (10.0, 5.0), (10.0, 5.0), 0.2) is None  # too low
    assert a.observe(0.0, "t", (10.0, 5.0), (10.0, 5.0), 9.0) is None  # too high
    assert a.observe(0.0, "t", (10.0, 5.0), (10.0, 5.0), 4.0) is not None
    assert a.n_rejected == 3


def test_disagreement_needs_the_gate_held_for_the_full_persist_window():
    a = TagAnchorMap(AnchorConfig(gate_m=1.0, gate_persist_s=1.0))
    settle(a)
    assert a.disagreement(1.0) is None  # at the datum
    a.observe(2.0, "tag36h11:6", (13.0, 5.0), (13.0, 5.0), 4.0)
    assert a.disagreement(2.0) is None  # breach starts, not yet held
    assert a.disagreement(2.5) is None
    reason = a.disagreement(3.1)
    assert reason is not None and "3.00 m" in reason


def test_a_breach_that_clears_resets_the_persist_timer():
    a = TagAnchorMap(AnchorConfig(gate_m=1.0, gate_persist_s=1.0))
    settle(a)
    a.observe(2.0, "tag36h11:6", (13.0, 5.0), (13.0, 5.0), 4.0)
    assert a.disagreement(2.0) is None
    a.observe(2.5, "tag36h11:6", (10.0, 5.0), (10.0, 5.0), 4.0)
    assert a.disagreement(2.5) is None  # median of the two residuals is 1.5...
    a.observe(2.6, "tag36h11:6", (10.0, 5.0), (10.0, 5.0), 4.0)
    assert a.disagreement(2.6) is None  # ...and now back at the datum
    assert a._breach_t0 is None


# The LZ pair: two tags roughly at opposing corners, about 1.54 m apart,
# with the pad between them. Nothing about the layout is surveyed.
PAIR = {"tag36h11:6": (0.0, -0.77), "tag36h11:7": (0.0, 0.77)}


def settle_pair(a, t0=0.0):
    for tag, fix in PAIR.items():
        settle(a, tag=tag, fix=fix, t0=t0)


def rotate(point, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (c * point[0] - s * point[1], s * point[0] + c * point[1])


def test_a_bad_frame_cannot_carry_the_drift_estimate():
    """Per-tag medians absorb a single bad frame."""
    a = TagAnchorMap()
    settle_pair(a)
    for tag, fix in PAIR.items():
        a.observe(2.0, tag, (fix[0] + 1.0, fix[1]), (fix[0] + 1.0, fix[1]), 4.0)
        a.observe(2.1, tag, (fix[0] + 9.0, fix[1]), (fix[0] + 9.0, fix[1]), 4.0)
        a.observe(2.2, tag, (fix[0] + 1.0, fix[1]), (fix[0] + 1.0, fix[1]), 4.0)
    assert a.drift(2.2).max_displacement == pytest.approx(1.0)


def test_two_tags_separate_yaw_drift_from_translation():
    """A yaw drift moves the two fixes in opposite directions. Averaging the
    residuals would report no drift at all; the rigid fit reports the yaw."""
    a = TagAnchorMap()
    settle_pair(a)
    for tag, datum in PAIR.items():
        fix = rotate(datum, 10.0)
        a.observe(2.0, tag, fix, fix, 4.0)
    d = a.drift(2.0)
    assert d.n_tags == 2
    assert math.degrees(d.dyaw) == pytest.approx(10.0)
    assert math.hypot(d.dn, d.de) == pytest.approx(0.0, abs=1e-9)
    # 2 * (baseline/2) * sin(dyaw/2), the arc each tag swept
    assert d.max_displacement == pytest.approx(1.54 * math.sin(math.radians(5.0)))


def test_a_single_tag_reads_a_yaw_drift_as_translation():
    """Rotation is unobservable from one point. The guard still fires, but the
    decomposition is wrong, which is why the correction wants the pair."""
    a = TagAnchorMap()
    settle(a, tag="tag36h11:6", fix=PAIR["tag36h11:6"])
    fix = rotate(PAIR["tag36h11:6"], 10.0)
    a.observe(2.0, "tag36h11:6", fix, fix, 4.0)
    d = a.drift(2.0)
    assert d.n_tags == 1 and d.dyaw == 0.0
    assert math.hypot(d.dn, d.de) == pytest.approx(d.max_displacement)


def test_a_pair_has_no_redundancy_to_reject_an_outlier_with():
    """Two correspondences determine the transform exactly, so a persistently
    bad tag is fitted, not rejected. Outlier rejection needs a third tag."""
    a = TagAnchorMap()
    settle_pair(a)
    a.observe(2.0, "tag36h11:6", (0.0, -0.77), (0.0, -0.77), 4.0)
    a.observe(2.0, "tag36h11:7", (9.0, 0.77), (9.0, 0.77), 4.0)
    assert a.drift(2.0).max_displacement > 4.0


def test_stale_fixes_stop_counting():
    a = TagAnchorMap(AnchorConfig(stale_s=5.0))
    settle(a)
    a.observe(1.0, "tag36h11:6", (13.0, 5.0), (13.0, 5.0), 4.0)
    assert a.drift(2.0) is not None
    assert a.drift(20.0) is None


def test_the_wall_encounter_reproduces_as_a_disagreement():
    """Flight 1 epoch 9: the camera held the drone 1.60 m from the pad while
    EKF2 placed it 6.99 m from the same anchor (wall-impact REPORT.md). The
    fix moves with the frame, so the residual carries the 5.40 m split."""
    a = TagAnchorMap(AnchorConfig(gate_m=1.5, gate_persist_s=1.0))
    settle(a, fix=(0.0, 0.0), t0=0.0)
    reason = None
    for i in range(20):  # 2 s of 10 Hz CM2 frames over the pad
        t = 180.0 + 0.1 * i
        a.observe(t, "tag36h11:6", (5.40, 0.0), (7.0, 0.0), 4.0)
        reason = a.disagreement(t)
        if reason is not None:
            break
    assert reason is not None and "5.40 m" in reason
    assert t - 180.0 == pytest.approx(1.0)  # one persist window, no longer


def test_report_carries_what_the_debrief_needs():
    a = TagAnchorMap()
    settle(a)
    a.observe(2.0, "tag36h11:6", (11.5, 5.0), (11.5, 5.0), 4.0)
    r = a.report()
    assert r["tags"]["tag36h11:6"]["settled"] is True
    assert r["max_residual_m"] == pytest.approx(1.5)
    assert r["n_fixes"] == a.cfg.settle_obs + 1


def test_config_rejects_a_gate_inside_the_settling_noise():
    with pytest.raises(ValueError):
        AnchorConfig(gate_m=0.3, settle_spread_m=0.6)


# --------------------------------------------------------------- correction


def settle_correction(c, dn, de, t0=0.0, dt=0.1, n=400):
    """Run the slew to completion and report when it arrived."""
    for i in range(n):
        t = t0 + dt * (i + 1)
        c.update(t, transform(dn, de))
        if c.pending_m < 1e-9:
            return t - t0
    raise AssertionError("correction never converged")


def test_the_offset_converts_both_ways_and_cancels():
    """The engine plans where the airframe truly is, and PX4 is handed a
    setpoint that puts it there. The pair of conversions is the whole idea."""
    c = SetpointCorrection()
    settle_correction(c, 2.0, -1.0)
    # The flight layer thinks it is 2 m north of truth, so the engine is told
    # the truth, and a plan for the truth is commanded 2 m north.
    assert c.to_plan((10.0, 5.0)) == pytest.approx((8.0, 6.0))
    assert c.to_flight(c.to_plan((10.0, 5.0))) == pytest.approx((10.0, 5.0))


def test_a_reacquisition_after_a_blind_stretch_does_not_command_a_step():
    """Flight 1 went 47 s without a sighting. A fix arriving after that gap is
    a step, and the airframe must walk to it, not lunge."""
    c = SetpointCorrection(CorrectionConfig(max_rate_mps=0.30))
    c.update(0.0, None)
    c.update(0.1, transform(5.0, 0.0))
    assert c.applied[0] == pytest.approx(0.03)  # one tick of slew, not 5 m
    assert c.pending_m == pytest.approx(4.97)
    elapsed = settle_correction(c, 5.0, 0.0, t0=0.1)
    assert elapsed == pytest.approx(5.0 / 0.30, rel=0.02)


def test_the_offset_holds_through_a_blind_stretch():
    """The error accumulated up to the last sighting does not go away because
    the tags left the frame. Decaying to zero would hand it back."""
    c = SetpointCorrection()
    settle_correction(c, 1.0, 0.0)
    for i in range(600):  # 60 s with no fix
        c.update(100.0 + 0.1 * i, None)
    assert c.applied == pytest.approx((1.0, 0.0))


def test_a_drift_past_the_limit_is_clamped_and_flagged():
    """Beyond the limit the rim compensates what it will and says so, rather
    than quietly applying an arbitrarily large offset."""
    c = SetpointCorrection(CorrectionConfig(max_correction_m=5.0))
    c.update(0.0, transform(6.62, 0.0))  # the flight-1 wall encounter
    assert c.saturated
    settle_correction(c, 6.62, 0.0)
    assert math.hypot(*c.applied) == pytest.approx(5.0)


def test_the_limit_clamps_direction_not_axes():
    c = SetpointCorrection(CorrectionConfig(max_correction_m=5.0))
    c.update(0.0, transform(8.0, 6.0))  # 10 m at a bearing
    assert c.target == pytest.approx((4.0, 3.0))


def test_rotation_is_resolved_at_the_tags_and_extrapolated_nowhere():
    """A yaw drift moves the tags, and that motion is corrected. The rotation
    itself is not applied to the plan: the pair resolves yaw far worse than
    position, so carrying it out to a distant waypoint overstates it."""
    c = SetpointCorrection()
    # The LZ pair, sitting 20 m from the frame origin, rotated 10 degrees.
    pair = ((-20.0, -0.77), (-20.0, 0.77))
    t = AnchorTransform(dn=0.0, de=0.0, dyaw=math.radians(10.0), n_tags=2, points=pair)
    c.update(0.0, t)
    # The tags moved about 20 * sin(10 deg); the offset is that, not zero.
    assert c.pending_m == pytest.approx(t.displacement_at(t.centroid))
    assert c.pending_m == pytest.approx(3.47, abs=0.02)


def test_the_offset_ignores_the_fits_translation_term():
    """The rigid fit's translation is about the coordinate origin. Anchoring on
    it would inject metres of correction that nothing measured.

    The case is a pure rotation about the tags: the pad turns under the camera
    and goes nowhere. Flight 1 anchored 20 m from its frame origin, which is
    where the two quantities part company."""
    pair = ((-20.0, -0.77), (-20.0, 0.77))
    mid = (-20.0, 0.0)
    # A rotation about `mid` is p -> R(p - mid) + mid, whose translation term
    # is mid - R*mid. Nothing at the tags moves.
    c10, s10 = math.cos(math.radians(10.0)), math.sin(math.radians(10.0))
    t = AnchorTransform(
        dn=mid[0] - (c10 * mid[0] - s10 * mid[1]),
        de=mid[1] - (s10 * mid[0] + c10 * mid[1]),
        dyaw=math.radians(10.0),
        n_tags=2,
        points=pair,
    )
    assert math.hypot(t.dn, t.de) > 3.0  # what the translation term claims
    assert t.displacement_at(t.centroid) == pytest.approx(0.0, abs=1e-9)
    corr = SetpointCorrection()
    corr.update(0.0, t)
    assert corr.pending_m == pytest.approx(0.0, abs=1e-9)


def test_correction_config_rejects_nonsense():
    with pytest.raises(ValueError):
        CorrectionConfig(max_rate_mps=0.0)
    with pytest.raises(ValueError):
        CorrectionConfig(max_correction_m=-1.0)
