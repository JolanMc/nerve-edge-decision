"""Latency measurements on the host CPU (ONNX Runtime, one thread).

Variants are measured in interleaved rounds to limit the effect of CPU
frequency drift. Absolute numbers depend on the host; at batch 1 a network
this small is dominated by per-operator overhead, so the operation counts and
the batched per-chip figures are what transfers to an embedded target.
"""

from __future__ import annotations

import platform
import time

import numpy as np

from .export import INPUT


def _stats(v: list[int], per: int = 1) -> dict:
    a = np.asarray(v, dtype=np.float64) / 1e3 / per
    return {"median_us": float(np.median(a)), "p95_us": float(np.percentile(a, 95)), "runs": len(v)}


def compare_calls(calls: dict, rounds: int = 10, runs: int = 200, warmup: int = 300, per: int = 1) -> dict:
    """Median and 95th percentile time (microseconds, divided by `per`) of each zero-argument callable."""
    for f in calls.values():
        for _ in range(warmup):
            f()
    samples = {k: [] for k in calls}
    for _ in range(rounds):
        for k, f in calls.items():
            s = samples[k]
            for _ in range(runs):
                t0 = time.perf_counter_ns()
                f()
                s.append(time.perf_counter_ns() - t0)
    return {k: _stats(v, per) for k, v in samples.items()}


def compare_latency(sessions: dict, x: np.ndarray, **kw) -> dict:
    """Time one `run` of every output of each session on the batch `x` (NCHW float32)."""
    feed = {INPUT: np.ascontiguousarray(x, dtype=np.float32)}
    calls = {}
    for k, s in sessions.items():
        names = [o.name for o in s.get_outputs()]
        calls[k] = lambda s=s, names=names: s.run(names, feed)
    return compare_calls(calls, **kw)


def time_call(fn, runs: int = 2000, warmup: int = 200) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    return _stats(ts)


def host() -> dict:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    import onnxruntime

    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "threads": 1,
    }
