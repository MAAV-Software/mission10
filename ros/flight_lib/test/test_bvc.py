import numpy as np

from flight_lib import (
    buffered_uncertainty_voronoi_clip,
    buffered_voronoi_clip,
    buffered_voronoi_halfspaces,
    normal_uncertainty_buffer,
)


def test_transparent_when_no_peers():
    out = buffered_voronoi_clip([0.0, 0.0], [5.0, 3.0], [], 0.5)
    assert np.allclose(out, [5.0, 3.0])


def test_transparent_when_goal_already_safe():
    # peer far away -> goal is inside the BVC, returned unchanged
    out = buffered_voronoi_clip([0.0, 0.0], [1.0, 0.0], [[20.0, 0.0]], 0.5)
    assert np.allclose(out, [1.0, 0.0])


def test_clip_stays_on_own_side_buffered():
    # goal pulls own across the bisector toward the peer; clip must hold the buffer
    own, peer, r = [0.0, 0.0], [2.0, 0.0], 0.5
    out = buffered_voronoi_clip(own, [10.0, 0.0], [peer], r)
    a_mat, b_vec = buffered_voronoi_halfspaces(own, [peer], r)
    assert np.all(a_mat @ out <= b_vec + 1e-6)         # feasible
    assert out[0] <= 1.0 - r + 1e-6                     # bisector at x=1, buffered inward


def test_reciprocal_swap_keeps_two_radii():
    # both drones want to swap straight through each other; the provable BVC guarantee is that
    # independently clipping keeps them at least 2*safety_radius apart.
    r = 0.5
    a, b = [0.0, 0.0], [2.0, 0.0]
    a_clip = buffered_voronoi_clip(a, [10.0, 0.0], [b], r)
    b_clip = buffered_voronoi_clip(b, [-8.0, 0.0], [a], r)
    assert np.linalg.norm(np.asarray(a_clip) - np.asarray(b_clip)) >= 2.0 * r - 1e-6


def test_clip_is_nearest_feasible_point():
    # single active constraint: the clip is the orthogonal projection onto the buffered bisector
    own, peer, r = [0.0, 0.0], [2.0, 0.0], 0.5
    goal = [10.0, 4.0]
    out = buffered_voronoi_clip(own, goal, [peer], r)
    assert np.allclose(out, [1.0 - r, 4.0], atol=1e-3)  # x clipped, y untouched


def test_three_peers_feasible():
    own, r = [0.0, 0.0], 0.4
    peers = [[1.5, 0.0], [-1.0, 1.0], [0.0, -1.2]]
    out = buffered_voronoi_clip(own, [5.0, 5.0], peers, r)
    a_mat, b_vec = buffered_voronoi_halfspaces(own, peers, r)
    assert np.all(a_mat @ out <= b_vec + 1e-6)


def test_uncertainty_buffer_matches_gaussian_quantile():
    # variance 0.25 (std 0.5) along the normal, delta=0.05 -> 0.5 * Phi^-1(0.95) = 0.5*1.6449
    buf = normal_uncertainty_buffer([1.0, 0.0], [[0.25, 0.0], [0.0, 9.0]], 0.05)
    assert buf == np.float64(0.5 * 1.6448536269514722).item() or abs(buf - 0.822) < 1e-3


def test_buac_zero_cov_reduces_to_bvc():
    own, peer, r = [0.0, 0.0], [2.0, 0.0], 0.5
    zero = [[0.0, 0.0], [0.0, 0.0]]
    a = buffered_voronoi_clip(own, [10.0, 0.0], [peer], r)
    b = buffered_uncertainty_voronoi_clip(own, [10.0, 0.0], [(peer, zero)], r, 0.05)
    assert np.allclose(a, b)


def test_buac_uncertainty_along_normal_inflates_buffer():
    own, peer, r = [0.0, 0.0], [2.0, 0.0], 0.5  # normal is +x
    plain = buffered_voronoi_clip(own, [10.0, 0.0], [peer], r)
    along = buffered_uncertainty_voronoi_clip(
        own, [10.0, 0.0], [(peer, [[0.4, 0.0], [0.0, 0.0]])], r, 0.05)
    assert along[0] < plain[0] - 1e-3   # more conservative -> clipped further from the peer


def test_buac_perpendicular_uncertainty_is_ignored():
    own, peer, r = [0.0, 0.0], [2.0, 0.0], 0.5  # normal is +x; cov only in y
    plain = buffered_voronoi_clip(own, [10.0, 0.0], [peer], r)
    perp = buffered_uncertainty_voronoi_clip(
        own, [10.0, 0.0], [(peer, [[0.0, 0.0], [0.0, 0.9]])], r, 0.05)
    assert np.allclose(plain, perp)


def test_buac_lower_delta_more_conservative():
    own, peer, r, cov = [0.0, 0.0], [2.0, 0.0], 0.5, [[0.4, 0.0], [0.0, 0.0]]
    loose = buffered_uncertainty_voronoi_clip(own, [10.0, 0.0], [(peer, cov)], r, 0.20)
    tight = buffered_uncertainty_voronoi_clip(own, [10.0, 0.0], [(peer, cov)], r, 0.01)
    assert tight[0] < loose[0] - 1e-3   # smaller collision prob -> bigger buffer
