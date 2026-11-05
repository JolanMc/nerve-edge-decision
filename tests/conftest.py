"""Shared fixtures: a tiny CNN built directly in ONNX (no PyTorch needed)."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def tiny_cnn(seed: int = 0) -> onnx.ModelProto:
    """Conv-Relu-Conv-Relu-MaxPool-Conv-Relu-GAP-Flatten-Gemm on 3x20x20 inputs."""
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.3):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    inits = [
        numpy_helper.from_array(w(8, 3, 3, 3), "w1"),
        numpy_helper.from_array(w(8, scale=0.1), "b1"),
        numpy_helper.from_array(w(8, 8, 3, 3, scale=0.15), "w2"),
        numpy_helper.from_array(w(8, scale=0.1), "b2"),
        numpy_helper.from_array(w(16, 8, 3, 3, scale=0.15), "w3"),
        numpy_helper.from_array(w(16, scale=0.1), "b3"),
        numpy_helper.from_array(w(2, 16, scale=0.5), "wf"),
        numpy_helper.from_array(w(2, scale=0.1), "bf"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], name="conv2", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("MaxPool", ["r2"], ["p2"], name="pool2", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p2", "w3", "b3"], ["c3"], name="conv3", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c3"], ["r3"], name="relu3"),
        helper.make_node("GlobalAveragePool", ["r3"], ["g"], name="gap"),
        helper.make_node("Flatten", ["g"], ["f"], name="flatten", axis=1),
        helper.make_node("Gemm", ["f", "wf", "bf"], ["logits"], name="fc", transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 20, 20])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """(fp32 model, int8 model, calibration inputs)."""
    from nerve_edge.export import quantize_int8

    d = tmp_path_factory.mktemp("tiny")
    fp32 = tiny_cnn()
    p32 = d / "tiny_fp32.onnx"
    p8 = d / "tiny_int8.onnx"
    onnx.save(fp32, str(p32))
    x = np.random.default_rng(1).standard_normal((256, 3, 20, 20)).astype(np.float32)
    quantize_int8(p32, p8, x[:128])
    return fp32, onnx.load(str(p8)), x
