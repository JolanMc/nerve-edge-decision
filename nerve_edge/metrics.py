"""Detection metrics. Out-of-distribution (OOD) samples are the positive class.

AUROC: probability that a random OOD sample scores higher than a random
in-distribution (ID) sample, ties counting one half (Mann-Whitney U).

FPR95: share of OOD samples still accepted when the threshold accepts 95 % of
ID samples (an input is accepted when its score is at or below the
threshold). Lower is better. This is the usual "FPR at 95 % TPR", with ID as
the class the threshold must keep.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def auroc(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    s = np.concatenate([score_id, score_ood]).astype(np.float64)
    r = rankdata(s, method="average")
    n_id, n_ood = len(score_id), len(score_ood)
    u = r[n_id:].sum() - n_ood * (n_ood + 1) / 2.0
    return float(u / (n_id * n_ood))


def fpr_at_tpr(score_id: np.ndarray, score_ood: np.ndarray, tpr: float = 0.95) -> float:
    thr = np.quantile(score_id.astype(np.float64), tpr, method="higher")
    return float(np.mean(score_ood <= thr))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


def balanced_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))
