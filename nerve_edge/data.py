"""Data: PlanesNet (in-distribution), CIFAR-10 test (far OOD) and sensor faults.

PlanesNet (R. Hammell, CC BY-SA 4.0) holds 32,000 RGB chips of 20 x 20 pixels
cut from PlanetScope imagery (3 m ground sampling distance) over California:
8,000 "plane" and 24,000 "no-plane" chips. It is downloaded from the author's
repository at a pinned commit and checked against a SHA-256 digest.

CIFAR-10 test images (natural photographs) are used only as far
out-of-distribution inputs. They are fetched from a public mirror at a pinned
commit with a sparse git checkout.

Nerve Edge Decision never redistributes either dataset.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLANESNET_COMMIT = "0da37d8e4167b9e0d37f8bf2366fe3cce2e2ea07"
PLANESNET_URL = f"https://raw.githubusercontent.com/rhammell/planesnet/{PLANESNET_COMMIT}/planesnet.7z"
PLANESNET_SHA256 = "fca293df333a1fe17f5fe81057bc35a44efe239751f401595365fb9b1b35b7ef"
PLANESNET_N = 32000
PLANESNET_NAME = re.compile(r"^([01])__(.+?)__(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.png$")
CLASS_NAMES = ("no-plane", "plane")

CIFAR_REPO = "https://github.com/YoongiKim/CIFAR-10-images"
CIFAR_COMMIT = "d7d05618061fe37d830fb0419063d113c575ccbc"
CIFAR_N_TEST = 10000

CHIP = 20  # PlanesNet chips are 20 x 20 pixels


# --------------------------------------------------------------------------- #
# Download and integrity
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_planesnet(raw_dir: Path) -> Path:
    """Download planesnet.7z at the pinned commit and verify its SHA-256."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "planesnet.7z"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(PLANESNET_URL, tmp)
        tmp.rename(archive)
    digest = sha256_file(archive)
    if digest != PLANESNET_SHA256:
        raise RuntimeError(f"planesnet.7z SHA-256 mismatch: got {digest}, expected {PLANESNET_SHA256}")
    return archive


def _safe_members(names: list[str]) -> list[str]:
    """Accept only 'planesnet/' and 'planesnet/<valid name>.png' entries."""
    keep = []
    for n in names:
        if n.rstrip("/") == "planesnet":
            continue
        parts = n.split("/")
        if (
            len(parts) != 2
            or parts[0] != "planesnet"
            or n.startswith("/")
            or ".." in parts
            or not PLANESNET_NAME.match(parts[1])
        ):
            raise RuntimeError(f"unexpected archive member: {n!r}")
        keep.append(n)
    return keep


@dataclass
class PlanesNet:
    x: np.ndarray  # uint8 [N, 20, 20, 3]
    y: np.ndarray  # int64 [N], 1 = plane
    scene: np.ndarray  # str [N], PlanetScope scene id
    lon: np.ndarray  # float64 [N]
    lat: np.ndarray  # float64 [N]
    name: np.ndarray  # str [N], original file name

    def __len__(self) -> int:
        return len(self.y)


def load_planesnet(raw_dir: Path) -> PlanesNet:
    """Download (if needed), extract, parse and cache PlanesNet as a .npz file."""
    raw_dir = Path(raw_dir)
    cache = raw_dir / "planesnet.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return PlanesNet(d["x"], d["y"], d["scene"], d["lon"], d["lat"], d["name"])

    import py7zr  # imported lazily: only needed once
    from PIL import Image

    archive = download_planesnet(raw_dir)
    out = raw_dir / "extracted"
    with py7zr.SevenZipFile(archive, "r") as z:
        members = _safe_members(z.getnames())
        if not (out / "planesnet").exists():
            z.extractall(path=out)
    members.sort()
    if len(members) != PLANESNET_N:
        raise RuntimeError(f"expected {PLANESNET_N} chips, found {len(members)}")

    x = np.empty((len(members), CHIP, CHIP, 3), dtype=np.uint8)
    y = np.empty(len(members), dtype=np.int64)
    scene, lon, lat, name = [], [], [], []
    for i, m in enumerate(members):
        base = m.split("/")[1]
        g = PLANESNET_NAME.match(base)
        with Image.open(out / m) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.shape != (CHIP, CHIP, 3):
            raise RuntimeError(f"{base}: unexpected shape {arr.shape}")
        x[i] = arr
        y[i] = int(g.group(1))
        scene.append(g.group(2))
        lon.append(float(g.group(3)))
        lat.append(float(g.group(4)))
        name.append(base)
    ds = PlanesNet(x, y, np.array(scene), np.array(lon), np.array(lat), np.array(name))
    if int(ds.y.sum()) != 8000:
        raise RuntimeError(f"expected 8000 plane chips, found {int(ds.y.sum())}")
    np.savez_compressed(cache, x=ds.x, y=ds.y, scene=ds.scene, lon=ds.lon, lat=ds.lat, name=ds.name)
    return ds


def scene_split(
    scene: np.ndarray, seed: int = 0, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> dict[str, np.ndarray]:
    """Split by PlanetScope scene so that no acquisition appears in two splits.

    Scenes are shuffled with a fixed seed and assigned to test, then validation,
    until each reaches its share of chips; the rest is training. The same
    airports still appear in several acquisitions (see the README limitations).
    """
    scenes, counts = np.unique(scene, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(scenes))
    n = len(scene)
    target = {"test": fractions[2] * n, "val": fractions[1] * n}
    filled = {"test": 0, "val": 0}
    where = {}
    for i in order:
        s = scenes[i]
        if filled["test"] < target["test"]:
            where[s] = "test"
            filled["test"] += counts[i]
        elif filled["val"] < target["val"]:
            where[s] = "val"
            filled["val"] += counts[i]
        else:
            where[s] = "train"
    labels = np.array([where[s] for s in scene])
    return {k: np.flatnonzero(labels == k) for k in ("train", "val", "test")}


# --------------------------------------------------------------------------- #
# CIFAR-10 test images (far out-of-distribution)
# --------------------------------------------------------------------------- #
def fetch_cifar10_test(raw_dir: Path) -> Path:
    """Sparse, partial clone of the CIFAR-10 test images at a pinned commit."""
    raw_dir = Path(raw_dir)
    repo = raw_dir / "cifar10"
    if not (repo / "test").exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", CIFAR_REPO, str(repo)],
            check=True,
            env=env,
        )
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "test"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "checkout", CIFAR_COMMIT], check=True, env=env)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != CIFAR_COMMIT:
        raise RuntimeError(f"CIFAR-10 mirror at {head}, expected {CIFAR_COMMIT}")
    n = sum(1 for p in (repo / "test").rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    if n != CIFAR_N_TEST:
        raise RuntimeError(f"expected {CIFAR_N_TEST} CIFAR-10 test images, found {n}")
    return repo / "test"


def load_cifar10_far_ood(raw_dir: Path, n_per_class: int = 200, seed: int = 0) -> np.ndarray:
    """A fixed, class-balanced subset of CIFAR-10 test images, resized to 20 x 20."""
    from PIL import Image

    test_dir = fetch_cifar10_test(raw_dir)
    rng = np.random.default_rng(seed)
    out = []
    for cls in sorted(p.name for p in test_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in (test_dir / cls).iterdir() if p.suffix.lower() in {".jpg", ".png"})
        for f in rng.choice(files, size=n_per_class, replace=False):
            with Image.open(f) as im:
                im = im.convert("RGB").resize((CHIP, CHIP), Image.BILINEAR)
                out.append(np.asarray(im, dtype=np.uint8))
    return np.stack(out)


# --------------------------------------------------------------------------- #
# Sensor faults and acquisition shifts (inputs in [0, 1], layout NHWC)
# --------------------------------------------------------------------------- #
def gaussian_noise(x, rng, sigma):
    return np.clip(x + rng.normal(0.0, sigma, x.shape), 0.0, 1.0)


def dead_pixels(x, rng, frac):
    """Pixels stuck at 0 or 1 on all bands (hot/dead detector elements)."""
    out = x.copy()
    mask = rng.random(x.shape[:3]) < frac
    out[mask] = rng.integers(0, 2, size=(int(mask.sum()), 1)).astype(x.dtype)
    return out


def stripes(x, rng, sigma):
    """Column-wise offsets per band, as left by an uncalibrated push-broom detector."""
    n, _, w, c = x.shape
    return np.clip(x + rng.normal(0.0, sigma, (n, 1, w, c)), 0.0, 1.0)


def saturation(x, rng, gain):
    return np.clip(x * gain, 0.0, 1.0)


def low_light(x, rng, factor, sigma=0.02):
    return np.clip(x * factor + rng.normal(0.0, sigma, x.shape), 0.0, 1.0)


def blur(x, rng, sigma):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma=(0, sigma, sigma, 0), mode="nearest")


def band_shift(x, rng, px):
    """Red band misregistered by `px` pixels along columns (edge replicated)."""
    out = x.copy()
    out[:, :, px:, 0] = x[:, :, :-px, 0]
    out[:, :, :px, 0] = x[:, :, :1, 0]
    return out


def cloud(x, rng, opacity):
    """A bright, soft blob of random centre and size over each chip."""
    n, h, w, _ = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = rng.uniform(0, h, (n, 1, 1))
    cx = rng.uniform(0, w, (n, 1, 1))
    r = rng.uniform(4.0, 9.0, (n, 1, 1))
    alpha = opacity * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
    return x * (1 - alpha[..., None]) + 0.9 * alpha[..., None]


#: name -> (function, {"mild": parameter, "strong": parameter})
CORRUPTIONS = {
    "noise": (gaussian_noise, {"mild": 0.05, "strong": 0.15}),
    "dead_pixels": (dead_pixels, {"mild": 0.02, "strong": 0.10}),
    "stripes": (stripes, {"mild": 0.03, "strong": 0.10}),
    "saturation": (saturation, {"mild": 1.8, "strong": 3.5}),
    "low_light": (low_light, {"mild": 0.5, "strong": 0.2}),
    "blur": (blur, {"mild": 0.7, "strong": 1.5}),
    "band_shift": (band_shift, {"mild": 1, "strong": 3}),
    "cloud": (cloud, {"mild": 0.5, "strong": 0.9}),
}


def corrupt(x01: np.ndarray, name: str, severity: str, seed: int = 0) -> np.ndarray:
    """Apply a named corruption at 'mild' or 'strong' severity, deterministically."""
    fn, params = CORRUPTIONS[name]
    rng = np.random.default_rng([seed, sorted(CORRUPTIONS).index(name), ("mild", "strong").index(severity)])
    return fn(x01.astype(np.float32), rng, params[severity]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@dataclass
class Normaliser:
    mean: np.ndarray  # [3]
    std: np.ndarray  # [3]

    @classmethod
    def fit(cls, x_uint8: np.ndarray) -> Normaliser:
        x = x_uint8.astype(np.float64) / 255.0
        return cls(x.mean(axis=(0, 1, 2)).astype(np.float32), x.std(axis=(0, 1, 2)).astype(np.float32))

    def __call__(self, x01: np.ndarray) -> np.ndarray:
        """NHWC in [0, 1] -> NCHW standardised float32."""
        z = (x01.astype(np.float32) - self.mean) / self.std
        return np.ascontiguousarray(z.transpose(0, 3, 1, 2))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def to01(x_uint8: np.ndarray) -> np.ndarray:
    return x_uint8.astype(np.float32) / 255.0
