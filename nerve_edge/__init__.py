"""Nerve Edge Decision: a runtime monitor for INT8 neural networks on the edge.

It watches two failure modes of a deployed, quantised classifier:

* inputs that leave the operating domain (out-of-distribution inputs,
  sensor faults), and
* parameter memory that no longer matches the validated model
  (bit flips in INT8 weights and INT32 biases, as caused by single-event
  upsets or tampering),

and it reports what each check costs in latency and memory.
"""

__version__ = "0.1.0"
