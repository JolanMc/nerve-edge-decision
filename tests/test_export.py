import numpy as np
import onnx
from onnx import TensorProto, helper

from nerve_edge.export import add_monitor_taps, conv_activation_tensors, macs, quantized_parameters
from nerve_edge.integrity import bits_equal
from nerve_edge.runtime import run, session


def test_taps_found_in_both_graphs(tiny_models):
    fp32, int8, _ = tiny_models
    assert len(conv_activation_tensors(fp32)) == 3
    assert len(conv_activation_tensors(int8)) == 3


def test_monitored_graph_keeps_logits_bit_identical(tiny_models):
    fp32, int8, x = tiny_models
    for m in (fp32, int8):
        mon, names = add_monitor_taps(m)
        a = run(session(m), x)["logits"]
        out = run(session(mon), x)
        assert bits_equal(a, out["logits"])
        assert [out[n].shape[1] for n in names] == [8, 8, 16]
        assert out["features"].shape == (len(x), 16)


def test_lite_monitor_adds_features_only(tiny_models):
    fp32, int8, x = tiny_models
    for m in (fp32, int8):
        lite, names = add_monitor_taps(m, layer_taps=False)
        full, _ = add_monitor_taps(m, layer_taps=True)
        assert names == []
        assert [o.name for o in lite.graph.output] == ["logits", "features"]
        a = run(session(lite), x)
        b = run(session(full), x)
        assert bits_equal(a["logits"], run(session(m), x)["logits"])
        assert bits_equal(a["features"], b["features"])


def test_int8_taps_are_uint8(tiny_models):
    _, int8, x = tiny_models
    mon, names = add_monitor_taps(int8)
    out = run(session(mon), x[:4])
    assert all(out[n].dtype == np.uint8 for n in names)


def test_fp32_features_and_taps_are_correct(tiny_models):
    fp32, _, x = tiny_models
    mon, names = add_monitor_taps(fp32)
    # Expose the last ReLU map to recompute the features and the last tap by hand.
    probe = onnx.ModelProto()
    probe.CopyFrom(mon)
    probe.graph.output.append(helper.make_tensor_value_info("r3", TensorProto.FLOAT, ["batch", 16, 10, 10]))
    out = run(session(probe), x[:16])
    np.testing.assert_allclose(out["features"], out["r3"].mean(axis=(2, 3)), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out[names[2]], out["r3"].max(axis=(2, 3)))


def test_quantized_parameters(tiny_models):
    _, int8, _ = tiny_models
    p = quantized_parameters(int8)
    kinds = [q["kind"] for q in p]
    assert kinds.count("weight") == 4 and kinds.count("bias") == 4
    assert {q["dtype"] for q in p if q["kind"] == "weight"} == {"int8"}
    assert {q["dtype"] for q in p if q["kind"] == "bias"} == {"int32"}


def test_macs(tiny_models):
    fp32, _, _ = tiny_models
    # conv1 8*3*9*400 + conv2 8*8*9*400 + conv3 16*8*9*100 + fc 2*16
    assert macs(fp32)["total"] == 86400 + 230400 + 115200 + 32


def test_live_layer_names_follow_graph_order(tiny_models):
    from nerve_edge.live import layer_names

    _, int8, _ = tiny_models
    names = layer_names(int8)
    assert list(names.values()) == ["conv1", "conv2", "conv3", "fc"]
