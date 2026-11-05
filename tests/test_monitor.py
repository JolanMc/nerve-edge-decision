import numpy as np
import pytest

from nerve_edge.integrity import bits_equal
from nerve_edge.monitor import (
    ActivationBounds,
    Calibrated,
    Mahalanobis,
    Radiometry,
    chip_statistics,
    combine,
    energy,
    max_logit,
    msp,
    window_means,
    window_threshold,
)


def test_output_scores_rank_confident_inputs_lower():
    confident = np.array([[8.0, -2.0]])
    unsure = np.array([[0.1, 0.0]])
    for f in (msp, max_logit, energy):
        assert f(confident)[0] < f(unsure)[0]


def test_mahalanobis_matches_brute_force():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(400, 5))
    y = (rng.random(400) > 0.5).astype(int)
    f[y == 1] += 2.0
    m = Mahalanobis(ridge=1e-3).fit(f, y)
    q = rng.normal(size=(7, 5))
    brute = np.array([min((v - mu) @ m.precision @ (v - mu) for mu in m.means) for v in q])
    np.testing.assert_allclose(m.score(q), brute, rtol=1e-10)
    far = rng.normal(size=(50, 5)) + 8.0
    assert m.score(far).min() > np.quantile(m.score(f), 0.99)


def test_chip_statistics_layout():
    rng = np.random.default_rng(0)
    x = rng.random((4, 20, 20, 3))
    s = chip_statistics(x)
    assert s.shape == (4, 12)
    np.testing.assert_allclose(s[:, 0:3], x.mean(axis=(1, 2)))
    np.testing.assert_allclose(s[:, 3:6], x.std(axis=(1, 2)))
    np.testing.assert_allclose(s[:, 6:9], x.min(axis=(1, 2)))
    np.testing.assert_allclose(s[:, 9:12], x.max(axis=(1, 2)))


def test_radiometry_flags_exposure_changes():
    rng = np.random.default_rng(0)
    train = np.clip(rng.normal(0.4, 0.1, (500, 20, 20, 3)), 0, 1)
    test = np.clip(rng.normal(0.4, 0.1, (200, 20, 20, 3)), 0, 1)
    r = Radiometry().fit(train)
    clean = r.score(test)
    dark = r.score(test * 0.2)
    bright = r.score(np.clip(test * 3.5, 0, 1))
    assert dark.min() > clean.max()
    assert bright.min() > clean.max()
    assert r.state_bytes() == 2 * 12 * 4


def test_activation_bounds_zero_inside_positive_outside():
    rng = np.random.default_rng(0)
    taps = [rng.integers(10, 200, size=(500, 4)).astype(np.uint8), rng.integers(0, 50, size=(500, 3)).astype(np.uint8)]
    b = ActivationBounds().fit(taps)
    assert b.exceedance(taps).max() == 0.0
    assert b.violations(taps).max() == 0
    probe = [t[:2].copy() for t in taps]
    probe[0][0, 1] = 255
    probe[1][1, 2] = 255
    assert b.violations(probe).tolist() == [1, 1]
    assert (b.exceedance(probe) > 0).all()
    assert b.n_channels == 7
    assert b.state_bytes() == 14


def test_activation_bounds_below_lower_bound():
    f = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = ActivationBounds().fit([f])
    # Channel 0 bounds [1, 3] (width 2): a value of 0 is half a width below.
    assert b.exceedance([np.array([[0.0, 3.0]], dtype=np.float32)])[0] == pytest.approx(0.5)


def test_calibrated_tail_is_a_conformal_p_value():
    clean = np.arange(1.0, 100.0)  # 99 calibration scores 1..99
    cal = Calibrated(clean)
    # P(V >= s) with the +1 correction: (#{V >= s} + 1) / (n + 1)
    assert cal.tail(np.array([0.0]))[0] == pytest.approx(100 / 100)
    assert cal.tail(np.array([50.0]))[0] == pytest.approx(51 / 100)
    assert cal.tail(np.array([99.0]))[0] == pytest.approx(2 / 100)
    assert cal.tail(np.array([1e9]))[0] == pytest.approx(1 / 100)
    # Exchangeable fresh scores: P(tail <= alpha) <= alpha.
    rng = np.random.default_rng(0)
    cal = Calibrated(rng.normal(size=20000))
    p = cal.tail(rng.normal(size=50000))
    for alpha in (0.01, 0.05, 0.2):
        assert np.mean(p <= alpha) <= alpha + 0.005


def test_combine_is_monotone_and_takes_the_strongest_evidence():
    a = np.array([0.5, 0.01, 0.5])
    b = np.array([0.5, 0.5, 0.001])
    s = combine([a, b])
    np.testing.assert_allclose(s, -np.log([0.5, 0.01, 0.001]))
    assert combine([np.array([0.2])])[0] < combine([np.array([0.1])])[0]


def test_window_threshold_controls_false_alarms():
    rng = np.random.default_rng(1)
    calib = rng.exponential(size=5000)
    fresh = rng.exponential(size=20000)
    for w in (1, 8, 32):
        thr = window_threshold(calib, w, far=0.01)
        far = np.mean(window_means(fresh, w) > thr)
        assert far < 0.03


def test_window_means_drop_the_remainder():
    s = np.arange(10, dtype=float)
    np.testing.assert_allclose(window_means(s, 4), [1.5, 5.5])


def test_bits_equal_is_bit_exact():
    a = np.array([0.0, 1.0], dtype=np.float32)
    b = np.array([-0.0, 1.0], dtype=np.float32)
    assert not bits_equal(a, b)
    n = np.array([np.nan], dtype=np.float32)
    assert bits_equal(n, n.copy())
