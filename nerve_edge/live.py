"""Live monitoring of one INT8 model: the engine behind the interactive demo.

`LiveMonitor` loads the monitored INT8 graph, fits the lite and full monitors
exactly as the benchmark does (training split for the statistics, validation
split for the conformal calibration) and answers two questions:

* `check_inputs`: is this chip, or this batch of chips, still in the
  validated domain?
* `inject`: what happens when bits flip in the INT8 parameter memory, and
  which checks notice?

Alarms follow the on-board rule of the design note: a monitor raises an alarm
when one of its components exceeds its own threshold. The thresholds are set on
clean test chips that the demo does not show (a random half of the test split),
at 1 % false alarms for the monitor as a whole (Bonferroni: FAR / number of
components each).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from . import data as D
from . import faults as F
from . import integrity as I
from . import monitor as M
from . import runtime as R
from .pipeline import FULL_EXTRA, LITE_COMPONENTS, tap_list

COMPONENTS = LITE_COMPONENTS + FULL_EXTRA
CLASS_NAMES = ("no plane", "plane")
FAR = 0.01
WINDOW = 32


@dataclass
class Verdict:
    alarm_lite: np.ndarray  # bool [N]
    alarm_full: np.ndarray  # bool [N]
    alarm_msp: np.ndarray  # bool [N], softmax baseline at the same false-alarm rate
    ratio: dict[str, np.ndarray]  # component -> score / its threshold (> 1 = that component fires)
    pred: np.ndarray  # int [N]
    confidence: np.ndarray  # float [N], max softmax probability


def layer_names(model: onnx.ModelProto) -> dict[str, str]:
    """Compute-node name -> conv1..convN / fc, in graph (topological) order."""
    names, conv = {}, 0
    for n in model.graph.node:
        if n.op_type == "Conv":
            conv += 1
            names[n.name] = f"conv{conv}"
        elif n.op_type in ("Gemm", "MatMul"):
            names[n.name] = "fc"
    return names


def _quantile(scores: np.ndarray, far: float) -> float:
    """Smallest threshold with at most `far` of the clean scores strictly above it."""
    return float(np.quantile(scores, 1.0 - far, method="higher"))


class LiveMonitor:
    def __init__(self, model_path: Path, raw_dir: Path, seed: int = 0):
        self.model = onnx.load(str(model_path))
        self.sess = R.session(self.model, threads=2)
        ds = D.load_planesnet(raw_dir)
        split = D.scene_split(ds.scene, seed=seed)
        self.norm = D.Normaliser.fit(ds.x[split["train"]])
        x01 = {k: D.to01(ds.x[v]) for k, v in split.items()}
        ys = {k: ds.y[v] for k, v in split.items()}
        outs = {k: R.run(self.sess, self.norm(v)) for k, v in x01.items()}

        self.radiometry = M.Radiometry().fit(x01["train"])
        self.maha = M.Mahalanobis().fit(outs["train"]["features"], ys["train"])
        self.feature_bounds = M.ActivationBounds().fit([outs["train"]["features"]])
        self.layer_bounds = M.ActivationBounds().fit(tap_list(outs["train"]))

        # Test split: a random half sets the thresholds, the other half is what the demo shows.
        n = len(ys["test"])
        perm = np.random.default_rng(seed).permutation(n)
        held, shown = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
        t_out = {k: v[held] for k, v in outs["test"].items()}
        comp = self._components(t_out, x01["test"][held])
        self.thr_lite = {k: _quantile(comp[k], FAR / len(LITE_COMPONENTS)) for k in LITE_COMPONENTS}
        self.thr_full = {k: _quantile(comp[k], FAR / len(COMPONENTS)) for k in COMPONENTS}
        self.thr_msp = _quantile(M.msp(t_out["logits"]), FAR)
        self.thr_layer = _quantile(comp["LayerBounds"], FAR)
        self.thr_feat = _quantile(comp["FeatureBounds"], FAR)
        # Stream mode: a degraded sensor degrades every chip that follows, so the
        # mean of each score over the last WINDOW chips is compared with a window threshold.
        self.win_lite = {k: M.window_threshold(comp[k], WINDOW, FAR / len(LITE_COMPONENTS)) for k in LITE_COMPONENTS}
        self.win_full = {k: M.window_threshold(comp[k], WINDOW, FAR / len(COMPONENTS)) for k in COMPONENTS}
        self.win_msp = M.window_threshold(M.msp(t_out["logits"]), WINDOW, FAR)

        self.x01 = x01["test"][shown]
        self.y = ys["test"][shown]
        self.clean_out = {k: v[shown] for k, v in outs["test"].items()}

        # Bit flips: parameter memory, CRC reference, golden inputs from the validation split.
        self.mem = F.ParameterMemory(self.model)
        self.crc_ref = I.crc_table(self.mem.arrays)
        self.layers = layer_names(self.model)
        self.golden = self.norm(x01["val"][:8])
        self.golden_ref = R.run(self.sess, self.golden)["logits"]

    # ------------------------------------------------------------------ inputs
    def _components(self, out: dict, x01: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "Radiometry": self.radiometry.score(x01),
            "Mahalanobis": self.maha.score(out["features"]),
            "FeatureBounds": self.feature_bounds.exceedance([out["features"]]),
            "LayerBounds": self.layer_bounds.exceedance(tap_list(out)),
        }

    def check_inputs(self, x01: np.ndarray) -> Verdict:
        out = R.run(self.sess, self.norm(x01))
        s = self._components(out, x01)
        lg = out["logits"].astype(np.float64)
        prob = np.exp(lg - lg.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        return Verdict(
            alarm_lite=np.any([s[k] > self.thr_lite[k] for k in LITE_COMPONENTS], axis=0),
            alarm_full=np.any([s[k] > self.thr_full[k] for k in COMPONENTS], axis=0),
            alarm_msp=M.msp(lg) > self.thr_msp,
            ratio={k: s[k] / max(self.thr_full[k], 1e-12) for k in COMPONENTS},
            pred=lg.argmax(1),
            confidence=prob.max(1),
        )

    def check_stream(self, x01: np.ndarray) -> dict:
        """One window of consecutive chips from the same sensor: window means against window thresholds."""
        out = R.run(self.sess, self.norm(x01))
        s = self._components(out, x01)
        mean = {k: float(np.mean(v)) for k, v in s.items()}
        msp_mean = float(np.mean(M.msp(out["logits"])))
        fired_lite = [k for k in LITE_COMPONENTS if mean[k] > self.win_lite[k]]
        fired_full = [k for k in COMPONENTS if mean[k] > self.win_full[k]]
        return {
            "pred": out["logits"].argmax(1),
            "alarm_lite": bool(fired_lite),
            "alarm_full": bool(fired_full),
            "fired_lite": fired_lite,
            "fired_full": fired_full,
            "alarm_msp": msp_mean > self.win_msp,
            "ratio": {k: mean[k] / max(self.win_full[k], 1e-12) for k in COMPONENTS},
        }

    def corrupt(self, x01: np.ndarray, fault: str, severity: str, seed: int = 0) -> np.ndarray:
        if fault == "none":
            return x01.astype(np.float32)
        return D.corrupt(x01, fault, severity, seed=seed)

    # ------------------------------------------------------------------ bit flips
    def sample_flips(self, rng: np.random.Generator, mode: str, layer: str, k: int) -> list[tuple[str, int, int]]:
        """mode: 'uniform' (k bits anywhere), 'weight_sign' or 'bias_high' (one bit, optionally in `layer`)."""
        if mode == "uniform":
            return self.mem.sample(rng, k)
        kind, bits = ("weight", [7]) if mode == "weight_sign" else ("bias", list(range(24, 32)))
        names = [
            n
            for n in self.mem.names
            if self.mem.kind_of(n) == kind and (layer == "any" or self.layers.get(self.mem.layer_of(n)) == layer)
        ]
        if not names:
            raise ValueError(f"no {kind} tensor in layer {layer}")
        sizes = np.array([self.mem.arrays[n].size for n in names], dtype=np.int64)
        t = int(rng.choice(len(names), p=sizes / sizes.sum()))
        return [(names[t], int(rng.integers(sizes[t])), int(rng.choice(bits)))]

    def inject(self, flips: list[tuple[str, int, int]], n_chips: int = 512) -> dict:
        sess = R.session(self.mem.build(flips), threads=2)
        x = self.norm(self.x01[:n_chips])
        out = R.run(sess, x)
        clean = {k: v[:n_chips] for k, v in self.clean_out.items()}
        y = self.y[:n_chips]
        pred, ref = out["logits"].argmax(1), clean["logits"].argmax(1)
        layer_alarm = self.layer_bounds.exceedance(tap_list(out)) > self.thr_layer
        feat_alarm = self.feature_bounds.exceedance([out["features"]]) > self.thr_feat
        golden = R.run(sess, self.golden)["logits"]
        rows = []
        for n, e, b in flips:
            old = int(self.mem.arrays[n].reshape(-1)[e])
            rows.append(
                {
                    "layer": self.layers.get(self.mem.layer_of(n), self.mem.layer_of(n)),
                    "kind": self.mem.kind_of(n),
                    "element": int(e),
                    "bit": int(b),
                    "old": old,
                    "new": old + F.value_change(self.mem.arrays[n], e, b),
                }
            )
        return {
            "flips": rows,
            "n": int(len(y)),
            "accuracy_clean": float((ref == y).mean()),
            "accuracy_faulty": float((pred == y).mean()),
            "changed": float((pred != ref).mean()),
            "crc_tensors": I.crc_mismatches({**self.mem.arrays, **self.mem.flipped(flips)}, self.crc_ref),
            "golden_changed": not I.bits_equal(golden, self.golden_ref),
            "layer_alarm_rate": float(layer_alarm.mean()),
            "feature_alarm_rate": float(feat_alarm.mean()),
        }
