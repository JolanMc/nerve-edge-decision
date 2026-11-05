"""Monitor scores. Convention everywhere: a higher score means more anomalous.

Output-based baselines:
  maximum softmax probability, MSP (Hendrycks & Gimpel, 2017), max logit
  (Hendrycks et al., 2022), energy (Liu et al., 2020), each negated so that
  higher means more anomalous.
Feature-based:
  Mahalanobis with class means and a shared covariance (Lee et al., 2018,
  without input pre-processing); deep k-NN on normalised features (Sun et
  al., 2022), kept as a reference only because it stores training features.
Nerve Edge Decision components:
  Radiometry: two-sided z-scores of per-band statistics of the raw chip
    (mean, standard deviation, minimum, maximum) against the training chips:
    a sensor-health gate for exposure, saturation and dead pixels.
  ActivationBounds: per-channel bounds learnt on clean data, applied to the
    pooled features or to the per-channel maxima after each convolution. The
    bounds are those of range restriction (Chen et al., 2021, "Ranger"), used
    here to detect rather than to clip. They need no reference copy of the
    weights.
  combine: calibrated fusion. Each score becomes a conformal p-value against
    clean validation scores (Bates et al., 2023); the fused score is
    -log(min p), so an alarm at level a on the fused score is an alarm at
    level a on at least one component.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax


# --------------------------------------------------------------------------- #
# Output-based scores
# --------------------------------------------------------------------------- #
def msp(logits: np.ndarray) -> np.ndarray:
    return -softmax(logits.astype(np.float64), axis=1).max(axis=1)


def max_logit(logits: np.ndarray) -> np.ndarray:
    return -logits.astype(np.float64).max(axis=1)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    return -temperature * logsumexp(z, axis=1)


# --------------------------------------------------------------------------- #
# Feature-based scores
# --------------------------------------------------------------------------- #
class Mahalanobis:
    """min_k (f - mu_k)^T S^-1 (f - mu_k), shared covariance with relative ridge."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> Mahalanobis:
        f = feats.astype(np.float64)
        self.classes = np.unique(labels)
        self.means = np.stack([f[labels == c].mean(axis=0) for c in self.classes])
        centred = f - self.means[np.searchsorted(self.classes, labels)]
        cov = centred.T @ centred / len(f)
        d = cov.shape[0]
        cov += self.ridge * (np.trace(cov) / d) * np.eye(d)
        self.precision = np.linalg.inv(cov)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        diff = feats.astype(np.float64)[:, None, :] - self.means[None, :, :]  # [N, K, D]
        proj = (diff.reshape(-1, diff.shape[-1]) @ self.precision).reshape(diff.shape)
        return (proj * diff).sum(axis=-1).min(axis=1)

    def state_bytes(self, dtype=np.float32) -> int:
        return int((self.means.size + self.precision.size) * np.dtype(dtype).itemsize)


class DeepKNN:
    """Distance to the k-th nearest normalised training feature (not embeddable)."""

    def __init__(self, k: int = 50, bank_size: int = 10000, seed: int = 0):
        self.k, self.bank_size, self.seed = k, bank_size, seed

    @staticmethod
    def _norm(f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)

    def fit(self, feats: np.ndarray) -> DeepKNN:
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(feats), size=min(self.bank_size, len(feats)), replace=False)
        self.bank = self._norm(feats[idx])
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="brute").fit(self.bank)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(self._norm(feats))
        return d[:, -1].astype(np.float64)


# --------------------------------------------------------------------------- #
# Input radiometry
# --------------------------------------------------------------------------- #
def chip_statistics(x01: np.ndarray) -> np.ndarray:
    """Per-band mean, standard deviation, minimum and maximum of NHWC chips in [0, 1].

    Returns [N, 4 * bands]: the means of all bands, then the standard
    deviations, the minima and the maxima.
    """
    n, h, w, c = x01.shape
    x = np.ascontiguousarray(x01.reshape(n, h * w, c).transpose(0, 2, 1))  # [N, bands, pixels]
    stats = [x.mean(-1, dtype=np.float64), x.std(-1, dtype=np.float64), x.min(-1), x.max(-1)]
    return np.concatenate(stats, axis=1).astype(np.float64)


class Radiometry:
    """max_j |z_j| over the 12 chip statistics, z-scored against the training chips."""

    def fit(self, x01: np.ndarray) -> Radiometry:
        s = chip_statistics(x01)
        self.mu = s.mean(axis=0)
        self.sigma = np.maximum(s.std(axis=0), 1e-6)
        return self

    def score(self, x01: np.ndarray) -> np.ndarray:
        return np.abs((chip_statistics(x01) - self.mu) / self.sigma).max(axis=1)

    def state_bytes(self, dtype=np.float32) -> int:
        return int(2 * len(self.mu) * np.dtype(dtype).itemsize)


# --------------------------------------------------------------------------- #
# Activation bounds
# --------------------------------------------------------------------------- #
class ActivationBounds:
    """Bounds [lo_c, hi_c] per channel, over a list of [N, C] arrays (one per layer)."""

    def fit(self, taps: list[np.ndarray]) -> ActivationBounds:
        self.lo = [t.astype(np.float64).min(axis=0) for t in taps]
        self.hi = [t.astype(np.float64).max(axis=0) for t in taps]
        self.width = [
            np.maximum(hi - lo, 1.0 if t.dtype == np.uint8 else 1e-6)
            for hi, lo, t in zip(self.hi, self.lo, taps, strict=True)
        ]
        return self

    def exceedance(self, taps: list[np.ndarray]) -> np.ndarray:
        """Sum over channels of the distance outside the bounds, in units of the bound width."""
        s = np.zeros(len(taps[0]), dtype=np.float64)
        for t, lo, hi, w in zip(taps, self.lo, self.hi, self.width, strict=True):
            t = t.astype(np.float64)
            s += ((np.maximum(t - hi, 0.0) + np.maximum(lo - t, 0.0)) / w).sum(axis=1)
        return s

    def violations(self, taps: list[np.ndarray]) -> np.ndarray:
        """Number of channels outside their bounds."""
        n = np.zeros(len(taps[0]), dtype=np.int64)
        for t, lo, hi in zip(taps, self.lo, self.hi, strict=True):
            t = t.astype(np.float64)
            n += ((t > hi) | (t < lo)).sum(axis=1)
        return n

    @property
    def n_channels(self) -> int:
        return int(sum(len(h) for h in self.hi))

    def state_bytes(self, dtype=np.uint8) -> int:
        return int(2 * self.n_channels * np.dtype(dtype).itemsize)


# --------------------------------------------------------------------------- #
# Calibrated fusion
# --------------------------------------------------------------------------- #
class Calibrated:
    """Conformal tail probability P(V >= s) of a score against clean validation scores V."""

    def __init__(self, clean_scores: np.ndarray):
        self.v = np.sort(np.asarray(clean_scores, dtype=np.float64))

    def tail(self, s: np.ndarray) -> np.ndarray:
        n = len(self.v)
        at_least = n - np.searchsorted(self.v, np.asarray(s, dtype=np.float64), side="left")
        return (at_least + 1.0) / (n + 1.0)


def combine(tails: list[np.ndarray]) -> np.ndarray:
    """Fused score -log(min p) over calibrated tail probabilities."""
    return -np.log(np.minimum.reduce([np.asarray(t, dtype=np.float64) for t in tails]))


# --------------------------------------------------------------------------- #
# Windows (for persistent faults)
# --------------------------------------------------------------------------- #
def window_means(scores: np.ndarray, window: int) -> np.ndarray:
    """Means over consecutive, non-overlapping windows (the remainder is dropped)."""
    n = (len(scores) // window) * window
    return scores[:n].reshape(-1, window).mean(axis=1)


def window_threshold(clean_scores: np.ndarray, window: int, far: float, n_draws: int = 20000, seed: int = 0) -> float:
    """Threshold on the window mean giving a false-alarm rate `far` on clean data.

    Windows are drawn at random (with replacement) from the clean scores; an
    alarm is raised when the window mean is strictly above the threshold.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clean_scores), size=(n_draws, window))
    means = clean_scores[idx].mean(axis=1)
    return float(np.quantile(means, 1.0 - far, method="higher"))
