"""ONNX Runtime helpers: deterministic single-thread sessions and batched runs."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

from .export import INPUT


def session(model, threads: int = 1) -> ort.InferenceSession:
    """CPU session; `model` is a path, serialized bytes or an onnx.ModelProto."""
    if isinstance(model, onnx.ModelProto):
        model = model.SerializeToString()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(
        model if isinstance(model, (bytes, str)) else str(model), so, providers=["CPUExecutionProvider"]
    )


def run(sess: ort.InferenceSession, x: np.ndarray, batch: int = 512) -> dict[str, np.ndarray]:
    """Run all outputs over `x` (NCHW float32) in batches; concatenate per output."""
    names = [o.name for o in sess.get_outputs()]
    parts = {k: [] for k in names}
    for i in range(0, len(x), batch):
        outs = sess.run(names, {INPUT: np.ascontiguousarray(x[i : i + batch], dtype=np.float32)})
        for k, v in zip(names, outs, strict=True):
            parts[k].append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}
