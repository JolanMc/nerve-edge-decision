import numpy as np
import pytest

from nerve_edge.data import CORRUPTIONS, Normaliser, _safe_members, corrupt, scene_split, to01
from nerve_edge.pipeline import tap_list


def test_corruptions_are_deterministic_bounded_and_effective():
    x = np.random.default_rng(0).random((16, 20, 20, 3)).astype(np.float32)
    for name in CORRUPTIONS:
        for sev in ("mild", "strong"):
            a = corrupt(x, name, sev)
            b = corrupt(x, name, sev)
            assert a.dtype == np.float32 and a.shape == x.shape
            assert np.array_equal(a, b), (name, sev)
            assert a.min() >= 0.0 and a.max() <= 1.0, (name, sev)
            assert not np.array_equal(a, x), (name, sev)
    assert not np.array_equal(corrupt(x, "noise", "mild"), corrupt(x, "noise", "strong"))


def test_scene_split_is_disjoint_and_complete():
    rng = np.random.default_rng(0)
    scene = np.array([f"s{i}" for i in rng.integers(0, 300, size=5000)])
    split = scene_split(scene, seed=0)
    idx = np.concatenate(list(split.values()))
    assert np.array_equal(np.sort(idx), np.arange(len(scene)))
    sets = {k: set(scene[v]) for k, v in split.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert abs(len(split["test"]) / len(scene) - 0.2) < 0.02
    assert abs(len(split["val"]) / len(scene) - 0.1) < 0.02


def test_archive_members_are_validated():
    ok = ["planesnet/", "planesnet/1__20160714_165520_0c59__-118.4316_33.937.png"]
    assert _safe_members(ok) == ok[1:]
    for bad in (
        ["planesnet/../evil.png"],
        ["/planesnet/1__a__1_2.png"],
        ["other/1__a__1_2.png"],
        ["planesnet/readme.txt"],
        ["planesnet/sub/1__a__1_2.png"],
    ):
        with pytest.raises(RuntimeError):
            _safe_members(bad)


def test_normaliser_standardises_and_transposes():
    x = np.random.default_rng(0).integers(0, 256, size=(64, 20, 20, 3)).astype(np.uint8)
    n = Normaliser.fit(x)
    z = n(to01(x))
    assert z.shape == (64, 3, 20, 20) and z.dtype == np.float32 and z.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(z.mean(axis=(0, 2, 3)), 0.0, atol=1e-4)
    np.testing.assert_allclose(z.std(axis=(0, 2, 3)), 1.0, atol=1e-3)


def test_tap_list_orders_layers_numerically():
    out = {f"tap{i}_max": np.full((1, 1), i) for i in (10, 2, 0, 1)}
    out["features"] = np.zeros((1, 4))
    assert [int(t[0, 0]) for t in tap_list(out)] == [0, 1, 2, 10]
