import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nerve_edge.metrics import auroc, balanced_accuracy, fpr_at_tpr, wilson


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    s_id = np.round(rng.normal(0, 1, 500), 1)
    s_ood = np.round(rng.normal(0.7, 1, 300), 1)
    y = np.r_[np.zeros(500), np.ones(300)]
    assert auroc(s_id, s_ood) == pytest.approx(roc_auc_score(y, np.r_[s_id, s_ood]), abs=1e-12)


def test_auroc_extremes():
    assert auroc(np.zeros(10), np.ones(5)) == 1.0
    assert auroc(np.ones(10), np.zeros(5)) == 0.0
    assert auroc(np.zeros(10), np.zeros(5)) == 0.5


def test_fpr95_definition():
    s_id = np.arange(100, dtype=float)  # 95 % of ID at or below 95
    s_ood = np.array([10.0, 94.0, 95.0, 96.0, 200.0])
    # threshold = 95 -> OOD accepted when score <= 95: 10, 94, 95
    assert fpr_at_tpr(s_id, s_ood) == pytest.approx(3 / 5)


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_balanced_accuracy():
    logits = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 0, 1, 1])
    assert balanced_accuracy(logits, y) == pytest.approx((1.0 + 0.5) / 2)
