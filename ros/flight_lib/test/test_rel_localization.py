import numpy as np

from flight_lib import RelativePositionEKF, gnss_common_mode


def test_common_mode_cancels_in_difference():
    # two receivers at the same instant share the identical common-mode bias
    b = gnss_common_mode(3.7)
    own_true = np.array([5.0, 0.0])
    peer_true = np.array([8.0, 1.0])
    own_gnss = own_true + b
    peer_gnss = peer_true + b
    diff = peer_gnss - own_gnss
    assert np.allclose(diff, peer_true - own_true)  # bias gone, true relative position left


def test_gnss_only_converges_to_relative_position():
    rng = np.random.default_rng(0)
    true_rel = np.array([3.0, 1.0])
    ekf = RelativePositionEKF()
    R = (0.4 ** 2) * np.eye(2)
    for _ in range(200):
        z = true_rel + rng.normal(0, 0.4, 2)
        ekf.update_gnss(z, R)
    assert np.linalg.norm(ekf.mean - true_rel) < 0.15


def test_range_update_makes_covariance_anisotropic():
    # relative position along +x; range tightens the x (radial) axis, leaves y (tangential) loose
    ekf = RelativePositionEKF()
    ekf.init_from_gnss([4.0, 0.0], (0.5 ** 2) * np.eye(2))
    for _ in range(50):
        ekf.update_range(4.0, 0.05 ** 2)
    cov = ekf.cov
    assert cov[0, 0] < cov[1, 1]              # radial variance < tangential
    assert cov[0, 0] < 0.05 ** 2 + 1e-6       # radial driven down near the range noise floor


def test_range_does_not_tighten_tangential():
    ekf = RelativePositionEKF()
    ekf.init_from_gnss([4.0, 0.0], (0.5 ** 2) * np.eye(2))
    before = ekf.cov[1, 1]
    for _ in range(50):
        ekf.update_range(4.0, 0.05 ** 2)
    assert abs(ekf.cov[1, 1] - before) < 1e-9  # perpendicular axis untouched by range


def test_predict_grows_covariance():
    ekf = RelativePositionEKF(vel_noise_std=0.3)
    ekf.init_from_gnss([2.0, 0.0], (0.1 ** 2) * np.eye(2))
    trace0 = np.trace(ekf.cov)
    ekf.predict([0.0, 0.0], 0.5)
    assert np.trace(ekf.cov) > trace0  # uncertainty grows while only dead-reckoning


def test_predict_dead_reckons_position():
    ekf = RelativePositionEKF()
    ekf.init_from_gnss([0.0, 0.0], 0.1 * np.eye(2))
    ekf.predict([1.0, -2.0], 0.5)  # rel velocity * dt
    assert np.allclose(ekf.mean, [0.5, -1.0])


def test_fused_estimate_beats_gnss_alone_radially():
    # range + diff-GNSS fusion converges tight radially even with loose GNSS
    rng = np.random.default_rng(1)
    true_rel = np.array([5.0, 0.0])
    ekf = RelativePositionEKF()
    R_g = (0.6 ** 2) * np.eye(2)
    for _ in range(150):
        ekf.update_gnss(true_rel + rng.normal(0, 0.6, 2), R_g)
        ekf.update_range(5.0 + rng.normal(0, 0.08), 0.08 ** 2)
    assert ekf.cov[0, 0] < ekf.cov[1, 1]
    assert np.linalg.norm(ekf.mean - true_rel) < 0.2
