"""ONNX export, INT8 static quantisation, and monitor taps added by graph surgery.

The deployed ("plain") graph outputs logits only. The monitored graph is the
same graph plus read-only taps:

* tap<i>_max: per-channel maximum of the activation after each convolution
  (ReduceMax over height and width), read from the INT8 tensor itself;
* features: the 128-d pooled feature vector, read through an extra
  DequantizeLinear node.

Taps only add consumers to existing tensors, so the quantised core is
unchanged. The pipeline checks that plain and monitored graphs return
bit-identical logits.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT = "input"
LOGITS = "logits"
FEATURES = "features"


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_fp32(net, path: Path, opset: int = 17) -> None:
    """Export EdgeNet (batch-norm folded) to ONNX with a dynamic batch axis."""
    import torch

    from .model import ExportNet

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[0]

    core = LogitsOnly(ExportNet(net).eval()).eval()
    x = torch.randn(8, 3, 20, 20, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = net.eval()(x)
        got = core(x)
    if not torch.allclose(ref, got, atol=1e-4, rtol=1e-4):
        raise RuntimeError("batch-norm folding changed the network output")
    torch.onnx.export(
        core,
        x[:1],
        str(path),
        input_names=[INPUT],
        output_names=[LOGITS],
        dynamic_axes={INPUT: {0: "batch"}, LOGITS: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


# --------------------------------------------------------------------------- #
# Static INT8 quantisation (ONNX Runtime, QDQ format)
# --------------------------------------------------------------------------- #
def quantize_int8(fp32_path: Path, int8_path: Path, calib_x: np.ndarray, batch: int = 64) -> dict:
    """Per-channel symmetric INT8 weights, asymmetric UINT8 activations, MinMax calibration."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, x):
            self.batches = [x[i : i + batch] for i in range(0, len(x), batch)]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            self.i += 1
            return {INPUT: self.batches[self.i - 1]}

    prep = Path(fp32_path).with_name(Path(fp32_path).stem + ".prep.onnx")
    quant_pre_process(str(fp32_path), str(prep))
    config = {
        "quant_format": "QDQ",
        "per_channel": True,
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "calibrate_method": "MinMax",
        "calibration_samples": int(len(calib_x)),
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    quantize_static(
        str(prep),
        str(int8_path),
        Reader(calib_x),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prep.unlink(missing_ok=True)
    onnx.checker.check_model(onnx.load(str(int8_path)))
    return config


# --------------------------------------------------------------------------- #
# Graph surgery: monitor taps
# --------------------------------------------------------------------------- #
def _consumers(graph) -> dict[str, list]:
    c = defaultdict(list)
    for n in graph.node:
        for i in n.input:
            c[i].append(n)
    return c


def _producer(graph) -> dict[str, object]:
    return {o: n for n in graph.node for o in n.output}


def _opset(model) -> int:
    return max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))


def conv_activation_tensors(model) -> list[str]:
    """Tensors holding each convolution's (ReLU-ed) activation, in graph order.

    FP32 graph: the Relu outputs. INT8 QDQ graph: the output of the
    QuantizeLinear that follows each Conv (the quantiser may have absorbed the
    ReLU into the quantisation range).
    """
    g = model.graph
    cons = _consumers(g)
    int8 = any(n.op_type == "QuantizeLinear" for n in g.node)
    taps = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        t = n.output[0]
        nxt = cons[t]
        if len(nxt) == 1 and nxt[0].op_type == "Relu":
            t = nxt[0].output[0]
            nxt = cons[t]
        if int8:
            qs = [c for c in nxt if c.op_type == "QuantizeLinear"]
            if len(qs) != 1:
                raise RuntimeError(f"no unique QuantizeLinear after Conv {n.name}")
            taps.append(qs[0].output[0])
        else:
            taps.append(t)
    return taps


def _pooled_tensor(model) -> tuple[str, object | None]:
    """The GlobalAveragePool output; for INT8 also the QuantizeLinear reading it."""
    g = model.graph
    cons = _consumers(g)
    gap = [n for n in g.node if n.op_type == "GlobalAveragePool"]
    if len(gap) != 1:
        raise RuntimeError("expected exactly one GlobalAveragePool")
    t = gap[0].output[0]
    qs = [c for c in cons[t] if c.op_type == "QuantizeLinear"]
    return t, (qs[0] if qs else None)


def add_monitor_taps(model: onnx.ModelProto, layer_taps: bool = True) -> tuple[onnx.ModelProto, list[str]]:
    """Return a copy of `model` with a features output and, if `layer_taps`, per-channel max taps.

    Without layer taps ("lite" monitor) the only addition is the pooled feature
    vector, which the network computes anyway.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    opset = _opset(m)
    int8 = any(n.op_type == "QuantizeLinear" for n in g.node)
    tap_names = []
    for j, t in enumerate(conv_activation_tensors(m) if layer_taps else []):
        out = f"tap{j}_max"
        if opset >= 18:
            axes = f"jorak_axes_{j}"
            g.initializer.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), axes))
            node = helper.make_node("ReduceMax", [t, axes], [out], name=f"jorak_tap{j}", keepdims=0)
        else:
            node = helper.make_node("ReduceMax", [t], [out], name=f"jorak_tap{j}", axes=[2, 3], keepdims=0)
        g.node.append(node)
        g.output.append(
            helper.make_tensor_value_info(out, TensorProto.UINT8 if int8 else TensorProto.FLOAT, ["batch", None])
        )
        tap_names.append(out)

    pooled, q = _pooled_tensor(m)
    if int8:
        if q is None:
            raise RuntimeError("pooled features are not quantised")
        deq = "jorak_features_dq"
        g.node.append(
            helper.make_node("DequantizeLinear", [q.output[0], q.input[1], q.input[2]], [deq], name="jorak_features_dq")
        )
        pooled = deq
    g.node.append(helper.make_node("Flatten", [pooled], [FEATURES], name="jorak_features", axis=1))
    g.output.append(helper.make_tensor_value_info(FEATURES, TensorProto.FLOAT, ["batch", None]))
    onnx.checker.check_model(m)
    return m, tap_names


# --------------------------------------------------------------------------- #
# Parameter memory of an INT8 QDQ graph
# --------------------------------------------------------------------------- #
def quantized_parameters(model: onnx.ModelProto) -> list[dict]:
    """INT8 weights and INT32 biases that feed DequantizeLinear nodes of Conv/Gemm.

    Returns dicts {name, kind ('weight'|'bias'), dtype, shape, layer}, in the
    order of their DequantizeLinear nodes (not necessarily the order of the
    layers); `layer` is the name of the consuming Conv/Gemm node.
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    cons = _consumers(g)
    out = []
    seen = set()
    for n in g.node:
        if n.op_type != "DequantizeLinear" or n.input[0] not in inits or n.input[0] in seen:
            continue
        arr = numpy_helper.to_array(inits[n.input[0]])
        users = cons[n.output[0]]
        target = [u for u in users if u.op_type in ("Conv", "Gemm", "MatMul")]
        if not target:
            continue
        if arr.dtype == np.int8:
            kind = "weight"
        elif arr.dtype == np.int32:
            kind = "bias"
        else:
            continue
        seen.add(n.input[0])
        out.append(
            {
                "name": n.input[0],
                "kind": kind,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "layer": target[0].name,
            }
        )
    return out


def macs(model: onnx.ModelProto, input_hw=(20, 20)) -> dict:
    """Multiply-accumulate count per Conv/Gemm for one input (FP32 graph, 'same' padding,
    2x2 pooling), and the number of values each convolution writes."""
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    h, w = input_hw
    per, outputs = [], []
    for n in g.node:
        if n.op_type == "Conv":
            wt = inits[n.input[1]]
            cout, cin_g, kh, kw = wt.shape
            per.append((n.name, int(cout * cin_g * kh * kw * h * w)))
            outputs.append((n.name, int(cout * h * w)))
        elif n.op_type == "MaxPool":
            h, w = h // 2, w // 2
        elif n.op_type == "Gemm":
            wt = inits[n.input[1]]
            per.append((n.name, int(wt.shape[0] * wt.shape[1])))
    return {
        "per_layer": per,
        "total": int(sum(v for _, v in per)),
        "conv_outputs": outputs,
        "conv_output_values": int(sum(v for _, v in outputs)),
    }
