import numpy as np
import onnx
from onnx import numpy_helper

from nerve_edge.export import add_monitor_taps
from nerve_edge.faults import ParameterMemory, value_change
from nerve_edge.integrity import bits_equal, crc_mismatches, crc_table
from nerve_edge.runtime import run, session


def test_value_change_twos_complement():
    a = np.array([3, -1, 127, -128], dtype=np.int8)
    assert value_change(a, 0, 7) == -128  # 3 -> -125
    assert value_change(a, 1, 0) == -1  # -1 -> -2
    assert value_change(a, 1, 7) == 128  # -1 -> 127
    assert value_change(a, 2, 6) == -64  # 127 -> 63
    assert value_change(a, 3, 7) == 128  # -128 -> 0
    b = np.array([5], dtype=np.int32)
    assert value_change(b, 0, 31) == -(2**31)


def test_flip_is_an_involution_and_local(tiny_models):
    _, int8, _ = tiny_models
    mem = ParameterMemory(int8)
    name = mem.names[0]
    once = mem.flipped([(name, 0, 7)])[name]
    assert int(once.reshape(-1)[0]) - int(mem.arrays[name].reshape(-1)[0]) == value_change(mem.arrays[name], 0, 7)
    assert np.array_equal(once.reshape(-1)[1:], mem.arrays[name].reshape(-1)[1:])
    twice = mem.flipped([(name, 0, 7), (name, 0, 6)])[name]
    assert int(twice.reshape(-1)[0]) != int(once.reshape(-1)[0])


def test_locate_covers_boundaries(tiny_models):
    _, int8, _ = tiny_models
    mem = ParameterMemory(int8)
    assert mem.locate(0) == (mem.names[0], 0, 0)
    last = mem.locate(mem.total_bits - 1)
    assert last[0] == mem.names[-1]
    assert last[1] == mem.arrays[mem.names[-1]].size - 1
    assert last[2] == mem.arrays[mem.names[-1]].dtype.itemsize * 8 - 1
    assert mem.total_bytes * 8 == mem.total_bits


def test_sample_draws_distinct_bits(tiny_models):
    _, int8, _ = tiny_models
    mem = ParameterMemory(int8)
    flips = mem.sample(np.random.default_rng(0), 200)
    assert len(set(flips)) == 200


def test_sample_targeted_respects_kind_and_bits(tiny_models):
    _, int8, _ = tiny_models
    mem = ParameterMemory(int8)
    rng = np.random.default_rng(0)
    for _ in range(100):
        ((name, elem, bit),) = mem.sample_targeted(rng, "bias", [24, 31])
        assert mem.kind_of(name) == "bias" and bit in (24, 31)
        assert 0 <= elem < mem.arrays[name].size
        ((name, elem, bit),) = mem.sample_targeted(rng, "weight", [7])
        assert mem.kind_of(name) == "weight" and bit == 7
        assert abs(value_change(mem.arrays[name], elem, bit)) == 128


def test_build_changes_only_the_flipped_tensor(tiny_models):
    _, int8, x = tiny_models
    mem = ParameterMemory(int8)
    name = next(n for n in mem.names if mem.kind_of(n) == "weight")
    m = onnx.load_from_string(mem.build([(name, 0, 7)]))
    new = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    old = {i.name: numpy_helper.to_array(i) for i in int8.graph.initializer}
    changed = [k for k in old if not np.array_equal(old[k], new[k])]
    assert changed == [name]


def test_no_flip_reproduces_outputs_bit_for_bit(tiny_models):
    _, int8, x = tiny_models
    mon, _ = add_monitor_taps(int8)
    mem = ParameterMemory(mon)
    a = run(session(mon), x)["logits"]
    b = run(session(mem.build([])), x)["logits"]
    assert bits_equal(a, b)


def test_crc_detects_any_single_flip(tiny_models):
    _, int8, _ = tiny_models
    mem = ParameterMemory(int8)
    ref = crc_table(mem.arrays)
    rng = np.random.default_rng(3)
    for _ in range(200):
        flips = mem.sample(rng, 1)
        assert crc_mismatches({**mem.arrays, **mem.flipped(flips)}, ref) == [flips[0][0]]
