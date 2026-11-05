"""Bit-flip injection in the parameter memory of an INT8 QDQ graph.

Fault model (v0.1): single-event upsets in the memory that holds the quantised
parameters, i.e. the INT8 weights and the INT32 biases of every convolution
and of the classifier. Every bit of that memory is equally likely to flip.
Requantisation constants (scales, zero points) are out of scope for now.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from .export import quantized_parameters

_UNSIGNED = {np.dtype(np.int8): np.uint8, np.dtype(np.int32): np.uint32}


class ParameterMemory:
    """Flat, bit-addressable view of the quantised parameters of a model."""

    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.specs = quantized_parameters(model)
        if not self.specs:
            raise ValueError("no quantised parameters found")
        inits = {i.name: i for i in model.graph.initializer}
        self.arrays = {s["name"]: numpy_helper.to_array(inits[s["name"]]).copy() for s in self.specs}
        self.names = [s["name"] for s in self.specs]
        self.bits_per_elem = np.array([self.arrays[n].dtype.itemsize * 8 for n in self.names], dtype=np.int64)
        sizes = np.array([self.arrays[n].size for n in self.names], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes * self.bits_per_elem)])
        self.total_bits = int(self.offsets[-1])
        self.total_bytes = int(sum(a.nbytes for a in self.arrays.values()))

    def locate(self, global_bit: int) -> tuple[str, int, int]:
        t = int(np.searchsorted(self.offsets, global_bit, side="right") - 1)
        local = int(global_bit - self.offsets[t])
        bits = int(self.bits_per_elem[t])
        return self.names[t], local // bits, local % bits

    def sample(self, rng: np.random.Generator, k: int) -> list[tuple[str, int, int]]:
        """k distinct bits drawn uniformly over the whole parameter memory."""
        return [self.locate(int(b)) for b in rng.choice(self.total_bits, size=k, replace=False)]

    def sample_targeted(self, rng: np.random.Generator, kind: str, bits: list[int]) -> list[tuple[str, int, int]]:
        """One flip: a uniformly drawn element of the given kind, one of the given bit positions."""
        names = [n for n in self.names if self.kind_of(n) == kind]
        sizes = np.array([self.arrays[n].size for n in names], dtype=np.int64)
        t = int(rng.choice(len(names), p=sizes / sizes.sum()))
        return [(names[t], int(rng.integers(sizes[t])), int(rng.choice(bits)))]

    def flipped(self, flips) -> dict[str, np.ndarray]:
        """Copies of the tensors touched by `flips`, with the bits inverted."""
        out: dict[str, np.ndarray] = {}
        for name, elem, bit in flips:
            if name not in out:
                out[name] = self.arrays[name].copy()
            a = out[name]
            flat = a.reshape(-1).view(_UNSIGNED[a.dtype])
            flat[elem] ^= flat.dtype.type(1 << bit)
        return out

    def build(self, flips) -> bytes:
        """Serialized model with the flipped parameters."""
        m = onnx.ModelProto()
        m.CopyFrom(self.model)
        mod = self.flipped(flips)
        for init in m.graph.initializer:
            if init.name in mod:
                init.CopyFrom(numpy_helper.from_array(mod[init.name], init.name))
        return m.SerializeToString()

    def layer_of(self, name: str) -> str:
        return next(s["layer"] for s in self.specs if s["name"] == name)

    def kind_of(self, name: str) -> str:
        return next(s["kind"] for s in self.specs if s["name"] == name)


def value_change(array: np.ndarray, elem: int, bit: int) -> int:
    """Signed change of the stored integer when `bit` of element `elem` flips."""
    v = int(array.reshape(-1)[elem])
    width = array.dtype.itemsize * 8
    u = v & ((1 << width) - 1)
    u ^= 1 << bit
    new = u - (1 << width) if u >= 1 << (width - 1) else u
    return new - v
