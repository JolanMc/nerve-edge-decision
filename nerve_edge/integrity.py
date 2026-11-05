"""Reference-based integrity checks: CRC-32 of the parameters and golden inputs.

* CRC-32 per parameter tensor, compared with the values recorded at
  validation time. CRC-32 detects every single-bit error by construction;
  for random multi-bit errors the chance of a miss is about 2^-32.
* Golden ("canary") inputs: K stored inputs whose INT8 outputs were recorded
  bit for bit. Integer inference is deterministic, so any difference on a
  canary means the computation changed.
"""

from __future__ import annotations

import time
import zlib

import numpy as np


def crc_table(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {k: zlib.crc32(np.ascontiguousarray(v).tobytes()) for k, v in arrays.items()}


def crc_mismatches(arrays: dict[str, np.ndarray], reference: dict[str, int]) -> list[str]:
    return [k for k, v in crc_table(arrays).items() if v != reference[k]]


def crc_scan_seconds(arrays: dict[str, np.ndarray], repeats: int = 500) -> float:
    """Median time to recompute every CRC (one full scrub of the parameters)."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        crc_table(arrays)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-exact comparison of two float32 arrays (NaN and -0.0 included)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))
