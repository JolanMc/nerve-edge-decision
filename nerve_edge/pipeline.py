"""End-to-end pipeline: data, training, INT8 export, monitor fitting,
out-of-distribution benchmark, bit-flip campaigns, latency and memory.

    python -m nerve_edge.pipeline --work runs/v0.1 --raw data

Every stage is deterministic for a given seed (timings excepted). Results land
in <work>/results.json; `python -m nerve_edge.report` turns them into tables
and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from . import bench as B
from . import data as D
from . import export as E
from . import faults as F
from . import integrity as I
from . import metrics as Q
from . import monitor as M
from . import runtime as R

CFG = {
    "seed": 0,
    "epochs": 30,
    "batch": 128,
    "lr": 3e-3,
    "weight_decay": 5e-4,
    "calibration_samples": 2000,
    "far_ood_per_class": 200,
    "fault_eval_samples": 1000,
    "fault_threads": 2,
    "canaries": 32,
    "canary_k": [1, 8, 32],
    "windows": [1, 8, 32],
    "window_far": 0.01,
    "uniform_k": [1, 4, 16, 64, 256],
    "uniform_trials": {"1": 1500, "default": 300},
    "targeted": {
        "weight_sign": {"kind": "weight", "bits": [7], "trials": 500},
        "bias_high": {"kind": "bias", "bits": [24, 25, 26, 27, 28, 29, 30, 31], "trials": 200},
    },
    "major_change": 0.01,
    "harmful_drop_points": 3.0,
}

#: Scores computed for every input; higher means more anomalous.
DETECTORS = [
    "NerveLite",
    "NerveFull",
    "Radiometry",
    "Mahalanobis",
    "FeatureBounds",
    "LayerBounds",
    "Energy",
    "MaxLogit",
    "MSP",
    "DeepKNN",
]
#: Components fused into the Nerve Edge Decision score (lite), and the extra one of the full variant.
LITE_COMPONENTS = ["Radiometry", "Mahalanobis", "FeatureBounds"]
FULL_EXTRA = ["LayerBounds"]
#: Online detectors evaluated on windows of consecutive inferences in the bit-flip campaigns.
WINDOW_DETECTORS = ["layer_bounds", "feature_bounds", "mahalanobis"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tap_list(out: dict) -> list[np.ndarray]:
    """Layer taps of a monitored graph's outputs, in layer order (tap0, tap1, ..., tap10, ...)."""
    keys = sorted((k for k in out if k.startswith("tap")), key=lambda s: int(s[3:].split("_")[0]))
    return [out[k] for k in keys]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- #
# Data, training, export
# --------------------------------------------------------------------------- #
def stage_data(raw: Path, cfg: dict):
    ds = D.load_planesnet(raw)
    split = D.scene_split(ds.scene, seed=cfg["seed"])
    norm = D.Normaliser.fit(ds.x[split["train"]])
    cifar = D.load_cifar10_far_ood(raw, n_per_class=cfg["far_ood_per_class"], seed=cfg["seed"])
    info = {
        "planesnet_sha256": D.PLANESNET_SHA256,
        "planesnet_commit": D.PLANESNET_COMMIT,
        "cifar10_commit": D.CIFAR_COMMIT,
        "n_total": int(len(ds)),
        "splits": {
            k: {
                "chips": int(len(v)),
                "planes": int(ds.y[v].sum()),
                "plane_share": float(ds.y[v].mean()),
                "scenes": int(len(np.unique(ds.scene[v]))),
            }
            for k, v in split.items()
        },
        "scene_overlap": {
            "train_test": int(len(set(ds.scene[split["train"]]) & set(ds.scene[split["test"]]))),
            "train_val": int(len(set(ds.scene[split["train"]]) & set(ds.scene[split["val"]]))),
            "val_test": int(len(set(ds.scene[split["val"]]) & set(ds.scene[split["test"]]))),
        },
        "far_ood_cifar10": int(len(cifar)),
        "normaliser": norm.to_dict(),
    }
    return ds, split, norm, cifar, info


def stage_train(work: Path, xs: dict, ys: dict, cfg: dict):
    import torch

    from .model import EdgeNet, count_params
    from .train import train

    ckpt = work / "edgenet.pt"
    hist_path = work / "train_history.json"
    if ckpt.exists() and hist_path.exists():
        net = EdgeNet()
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        net.eval()
        hist = json.loads(hist_path.read_text())
        log(f"reusing checkpoint {ckpt}")
    else:
        net, hist = train(
            xs["train"],
            ys["train"],
            xs["val"],
            ys["val"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            seed=cfg["seed"],
            log=log,
        )
        torch.save(net.state_dict(), ckpt)
        hist_path.write_text(json.dumps(hist, indent=1))
    hist["params"] = count_params(net)
    return net, hist


def stage_export(work: Path, net, xs: dict, cfg: dict):
    """FP32 and INT8 graphs, each in three variants: plain, lite monitor, full monitor."""
    paths = {"fp32": work / "edgenet_fp32.onnx", "int8": work / "edgenet_int8.onnx"}
    E.export_fp32(net, paths["fp32"])
    rng = np.random.default_rng(cfg["seed"])
    calib_idx = rng.choice(len(xs["train"]), size=cfg["calibration_samples"], replace=False)
    qcfg = E.quantize_int8(paths["fp32"], paths["int8"], xs["train"][calib_idx])
    plain = {k: onnx.load(str(p)) for k, p in paths.items()}
    lite, full, taps = {}, {}, {}
    files = list(paths.values())
    for k, m in plain.items():
        lite[k], _ = E.add_monitor_taps(m, layer_taps=False)
        full[k], taps[k] = E.add_monitor_taps(m, layer_taps=True)
        for variant, model in (("lite", lite[k]), ("full", full[k])):
            p = work / f"edgenet_{k}_monitor_{variant}.onnx"
            onnx.save(model, str(p))
            files.append(p)
    info = {
        "quantization": qcfg,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_bytes(p.read_bytes())} for p in files},
        "taps": taps,
        "macs": E.macs(plain["fp32"]),
    }
    return plain, lite, full, info


def check_equivalence(plain: dict, lite: dict, full: dict, net, xs: dict) -> dict:
    """Monitor taps must not change the computation; ONNX FP32 must match PyTorch."""
    import torch

    x = xs["test"][:1024]
    res = {}
    for k in ("fp32", "int8"):
        a = R.run(R.session(plain[k]), x)
        b = R.run(R.session(lite[k]), x)
        c = R.run(R.session(full[k]), x)
        res[f"{k}_logits_bit_identical"] = bool(
            I.bits_equal(a["logits"], b["logits"]) and I.bits_equal(a["logits"], c["logits"])
        )
        res[f"{k}_features_lite_vs_full_bit_identical"] = bool(I.bits_equal(b["features"], c["features"]))
    with torch.no_grad():
        t = net.eval()(torch.from_numpy(x)).numpy()
    o = R.run(R.session(plain["fp32"]), x)["logits"]
    res["onnx_fp32_vs_torch_max_abs_diff"] = float(np.abs(o - t).max())
    if not all(v for key, v in res.items() if key.endswith("identical")):
        raise RuntimeError(f"monitor taps changed the computation: {res}")
    if res["onnx_fp32_vs_torch_max_abs_diff"] > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch: {res}")
    return res


def classification(outs: dict, ys: dict) -> dict:
    r = {}
    y = ys["test"]
    for k in ("fp32", "int8"):
        lg = outs[k]["test"]["logits"]
        pred = lg.argmax(1)
        r[k] = {
            "accuracy": Q.accuracy(lg, y),
            "balanced_accuracy": Q.balanced_accuracy(lg, y),
            "plane_recall": float((pred[y == 1] == 1).mean()),
            "false_alarm_rate": float((pred[y == 0] == 1).mean()),
            "n": int(len(y)),
        }
    a, b = outs["fp32"]["test"]["logits"].argmax(1), outs["int8"]["test"]["logits"].argmax(1)
    r["fp32_int8_agreement"] = float((a == b).mean())
    r["int8_minus_fp32_accuracy_points"] = 100 * (r["int8"]["accuracy"] - r["fp32"]["accuracy"])
    return r


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #
def component_scores(out: dict, x01: np.ndarray, mon: dict) -> dict[str, np.ndarray]:
    """Every individual score for a batch: `out` from the full monitored graph, `x01` the raw chips."""
    lg = out["logits"]
    return {
        "Radiometry": mon["radiometry"].score(x01),
        "Mahalanobis": mon["mahalanobis"].score(out["features"]),
        "FeatureBounds": mon["feature_bounds"].exceedance([out["features"]]),
        "LayerBounds": mon["layer_bounds"].exceedance(tap_list(out)),
        "Energy": M.energy(lg),
        "MaxLogit": M.max_logit(lg),
        "MSP": M.msp(lg),
        "DeepKNN": mon["knn"].score(out["features"]),
    }


def all_scores(out: dict, x01: np.ndarray, mon: dict) -> dict[str, np.ndarray]:
    """Individual scores plus the two calibrated fusions (lite and full)."""
    s = component_scores(out, x01, mon)
    cal = mon["calibrators"]
    lite = [cal[n].tail(s[n]) for n in LITE_COMPONENTS]
    s["NerveLite"] = M.combine(lite)
    s["NerveFull"] = M.combine(lite + [cal[n].tail(s[n]) for n in FULL_EXTRA])
    return {d: s[d] for d in DETECTORS}


def fit_monitors(outs: dict, x01: dict, ys: dict, seed: int) -> dict:
    """Fit every monitor on the training split; calibrate the fusion on the validation split."""
    radiometry = M.Radiometry().fit(x01["train"])
    mons = {}
    for k in ("fp32", "int8"):
        tr = outs[k]["train"]
        m = {
            "radiometry": radiometry,
            "mahalanobis": M.Mahalanobis().fit(tr["features"], ys["train"]),
            "feature_bounds": M.ActivationBounds().fit([tr["features"]]),
            "layer_bounds": M.ActivationBounds().fit(tap_list(tr)),
            "knn": M.DeepKNN(seed=seed).fit(tr["features"]),
        }
        va = component_scores(outs[k]["val"], x01["val"], m)
        m["calibrators"] = {n: M.Calibrated(va[n]) for n in LITE_COMPONENTS + FULL_EXTRA}
        mons[k] = m
    return mons


def ood_conditions(x01_test: np.ndarray, cifar: np.ndarray) -> dict[str, np.ndarray]:
    cond = {"far_ood/cifar10": D.to01(cifar)}
    for name in D.CORRUPTIONS:
        for sev in ("mild", "strong"):
            cond[f"{name}/{sev}"] = D.corrupt(x01_test, name, sev)
    return cond


def stage_ood(sessions: dict, mons: dict, norm, cifar, outs: dict, x01: dict, ys: dict) -> dict:
    conditions = ood_conditions(x01["test"], cifar)
    res = {}
    for k in ("fp32", "int8"):
        id_scores = all_scores(outs[k]["test"], x01["test"], mons[k])
        clean_pred = outs[k]["test"]["logits"].argmax(1)
        res[k] = {}
        for cname, xc01 in conditions.items():
            out = R.run(sessions[k], norm(xc01))
            sc = all_scores(out, xc01, mons[k])
            entry = {
                "detectors": {
                    d: {"auroc": Q.auroc(id_scores[d], sc[d]), "fpr95": Q.fpr_at_tpr(id_scores[d], sc[d])}
                    for d in DETECTORS
                },
                "n_ood": int(len(xc01)),
                "n_id": int(len(id_scores[DETECTORS[0]])),
            }
            if not cname.startswith("far_ood"):
                entry["accuracy_under_shift"] = Q.accuracy(out["logits"], ys["test"])
                entry["prediction_change_vs_clean"] = float((out["logits"].argmax(1) != clean_pred).mean())
            res[k][cname] = entry
        log(f"OOD benchmark done for {k}")
    return res


# --------------------------------------------------------------------------- #
# Bit flips
# --------------------------------------------------------------------------- #
def stage_faults(full_int8: onnx.ModelProto, mon: dict, xs: dict, ys: dict, outs_int8: dict, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"] + 1)
    threads = cfg["fault_threads"]
    mem = F.ParameterMemory(full_int8)
    crc_ref = I.crc_table(mem.arrays)
    tensor_index = {n: i for i, n in enumerate(mem.names)}

    idx_eval = rng.choice(len(xs["test"]), size=cfg["fault_eval_samples"], replace=False)
    idx_can = rng.choice(len(xs["val"]), size=cfg["canaries"], replace=False)
    x_eval, y_eval = xs["test"][idx_eval], ys["test"][idx_eval]
    x_all = np.concatenate([xs["val"][idx_can], x_eval])
    n_can = len(idx_can)

    clean = R.run(R.session(full_int8, threads=threads), x_all)
    rebuilt = R.run(R.session(mem.build([]), threads=threads), x_all)
    if not I.bits_equal(clean["logits"], rebuilt["logits"]):
        raise RuntimeError("rebuilding the model without flips changed its outputs")
    ref_can = clean["logits"][:n_can]
    ref_pred = clean["logits"][n_can:].argmax(1)

    def window_scores(out: dict) -> dict[str, np.ndarray]:
        return {
            "layer_bounds": mon["layer_bounds"].exceedance(tap_list(out)),
            "feature_bounds": mon["feature_bounds"].exceedance([out["features"]]),
            "mahalanobis": mon["mahalanobis"].score(out["features"]),
        }

    # Window thresholds for the campaign: `window_far` false alarms per window for
    # the clean model, calibrated on the test chips the campaign does not use
    # (same scenes, disjoint chips, as a commissioning phase would provide) and
    # checked on the campaign chips. For comparison, thresholds calibrated on the
    # validation scenes are applied to the whole clean test split (in random order).
    far, windows = cfg["window_far"], cfg["windows"]
    test_sc = window_scores(outs_int8["test"])
    val_sc = window_scores(outs_int8["val"])
    held_out = np.ones(len(xs["test"]), dtype=bool)
    held_out[idx_eval] = False
    thr = {d: {w: M.window_threshold(test_sc[d][held_out], w, far) for w in windows} for d in WINDOW_DETECTORS}
    clean_eval_sc = window_scores({key: v[n_can:] for key, v in clean.items()})
    clean_far = {
        d: {str(w): float(np.mean(M.window_means(clean_eval_sc[d], w) > thr[d][w])) for w in windows}
        for d in WINDOW_DETECTORS
    }
    perm = np.random.default_rng(cfg["seed"] + 2).permutation(len(xs["test"]))
    thr_val = {d: {w: M.window_threshold(val_sc[d], w, far) for w in windows} for d in WINDOW_DETECTORS}
    far_val_to_test = {
        d: {str(w): float(np.mean(M.window_means(test_sc[d][perm], w) > thr_val[d][w])) for w in windows}
        for d in WINDOW_DETECTORS
    }

    def trial(flips, campaign: str, k: int, t: int) -> dict:
        out = R.run(R.session(mem.build(flips), threads=threads), x_all)
        lg = out["logits"]
        pred = lg[n_can:].argmax(1)
        sc = window_scores({key: v[n_can:] for key, v in out.items()})
        return {
            "campaign": campaign,
            "k": k,
            "trial": t,
            # [tensor index, element, bit, change of the stored integer]
            "flips": [[tensor_index[n], int(e), int(b), int(F.value_change(mem.arrays[n], e, b))] for n, e, b in flips],
            "change": float((pred != ref_pred).mean()),
            "accuracy": float((pred == y_eval).mean()),
            "nonfinite": bool(not np.isfinite(lg).all()),
            "crc": bool(I.crc_mismatches({**mem.arrays, **mem.flipped(flips)}, crc_ref)),
            "canary": {str(K): bool(not I.bits_equal(lg[:K], ref_can[:K])) for K in cfg["canary_k"]},
            **{
                d: {str(w): float(np.mean(M.window_means(sc[d], w) > thr[d][w])) for w in cfg["windows"]}
                for d in WINDOW_DETECTORS
            },
        }

    trials = []
    t0 = time.perf_counter()
    for k in cfg["uniform_k"]:
        n = cfg["uniform_trials"].get(str(k), cfg["uniform_trials"]["default"])
        for t in range(n):
            trials.append(trial(mem.sample(rng, k), "uniform", k, t))
        log(f"faults: uniform, {k} bit(s), {n} trials ({time.perf_counter() - t0:.0f} s elapsed)")
    for name, spec in cfg["targeted"].items():
        for t in range(spec["trials"]):
            trials.append(trial(mem.sample_targeted(rng, spec["kind"], spec["bits"]), name, 1, t))
        log(f"faults: {name}, {spec['trials']} trials ({time.perf_counter() - t0:.0f} s elapsed)")

    weights = [n for n in mem.names if mem.kind_of(n) == "weight"]
    biases = [n for n in mem.names if mem.kind_of(n) == "bias"]
    return {
        "parameter_memory": {
            "tensors": len(mem.names),
            "bits": mem.total_bits,
            "bytes": mem.total_bytes,
            "weights_bytes": int(sum(mem.arrays[n].nbytes for n in weights)),
            "biases_bytes": int(sum(mem.arrays[n].nbytes for n in biases)),
        },
        "tensors": [{**s, "bytes": int(mem.arrays[s["name"]].nbytes)} for s in mem.specs],
        "eval_chips": int(len(idx_eval)),
        "canary_chips": int(n_can),
        "clean_accuracy_eval": float((ref_pred == y_eval).mean()),
        "window_calibration_chips": int(held_out.sum()),
        "thresholds": {d: {str(w): v for w, v in t.items()} for d, t in thr.items()},
        "clean_false_alarm_rate": clean_far,
        "false_alarm_rate_validation_thresholds_on_test": far_val_to_test,
        "campaign_seconds": time.perf_counter() - t0,
        "trials": trials,
    }


# --------------------------------------------------------------------------- #
# Latency and memory
# --------------------------------------------------------------------------- #
def stage_bench(
    plain: dict, lite: dict, full: dict, mons: dict, xs: dict, x01: dict, faults: dict, cfg: dict, work: Path
) -> dict:
    sessions = {}
    for k in ("fp32", "int8"):
        sessions[k] = R.session(plain[k])
        sessions[f"{k}_lite"] = R.session(lite[k])
        sessions[f"{k}_full"] = R.session(full[k])
    lat1 = B.compare_latency(sessions, xs["test"][:1])
    nb = 512
    lat_b = B.compare_latency(sessions, xs["test"][:nb], rounds=8, runs=8, warmup=4, per=nb)

    # Decision step on the INT8 model, written as it would be ported (batch 1, float32),
    # and the vectorised reference implementation amortised over a batch.
    mon = mons["int8"]
    rad, maha, fb, lb = mon["radiometry"], mon["mahalanobis"], mon["feature_bounds"], mon["layer_bounds"]
    cal = mon["calibrators"]
    thr = {n: np.float32(np.quantile(cal[n].v, 0.99)) for n in LITE_COMPONENTS + FULL_EXTRA}
    mu, inv_sg = rad.mu.astype(np.float32), (1.0 / rad.sigma).astype(np.float32)
    means, prec = maha.means.astype(np.float32), maha.precision.astype(np.float32)
    flo, fhi = fb.lo[0].astype(np.float32), fb.hi[0].astype(np.float32)
    finv = (1.0 / fb.width[0]).astype(np.float32)
    llo, lhi = np.concatenate(lb.lo).astype(np.float32), np.concatenate(lb.hi).astype(np.float32)
    linv = np.concatenate([1.0 / w for w in lb.width]).astype(np.float32)

    npix = D.CHIP * D.CHIP

    def post_lite(feat: np.ndarray, chip: np.ndarray) -> bool:
        c = chip.reshape(-1, 3).T  # [bands, pixels]: one pass, four running statistics per band
        s1, s2 = c.sum(1), np.einsum("ij,ij->i", c, c)
        m1 = s1 / npix
        st = np.concatenate((m1, np.sqrt(np.maximum(s2 / npix - m1 * m1, 0)), c.min(1), c.max(1)))
        r = np.abs((st - mu) * inv_sg).max()
        d = feat - means
        m = np.einsum("ij,ij->i", d @ prec, d).min()
        e = (np.maximum(feat - fhi, 0) + np.maximum(flo - feat, 0)) @ finv
        return bool(r > thr["Radiometry"]) | bool(m > thr["Mahalanobis"]) | bool(e > thr["FeatureBounds"])

    def post_full(feat: np.ndarray, taps: list[np.ndarray], chip: np.ndarray) -> bool:
        alarm = post_lite(feat, chip)
        t = np.concatenate(taps).astype(np.float32)
        lb_score = (np.maximum(t - lhi, 0) + np.maximum(llo - t, 0)) @ linv
        return alarm | bool(lb_score > thr["LayerBounds"])

    out1 = R.run(sessions["int8_full"], xs["test"][:1])
    chip1 = x01["test"][0]
    f1 = out1["features"][0].astype(np.float32)
    taps1 = [t[0] for t in tap_list(out1)]
    post1 = B.compare_calls(
        {"lite": lambda: post_lite(f1, chip1), "full": lambda: post_full(f1, taps1, chip1)},
        rounds=5,
        runs=1000,
        warmup=500,
    )
    outb = R.run(sessions["int8_full"], xs["test"][:nb])
    chipb, featb, tapsb = x01["test"][:nb], outb["features"], tap_list(outb)
    post_b = B.compare_calls(
        {
            "lite": lambda: (rad.score(chipb), maha.score(featb), fb.exceedance([featb])),
            "full": lambda: (rad.score(chipb), maha.score(featb), fb.exceedance([featb]), lb.exceedance(tapsb)),
        },
        rounds=3,
        runs=20,
        warmup=3,
        per=nb,
    )

    mem = F.ParameterMemory(full["int8"])
    crc_t = I.crc_scan_seconds(mem.arrays)
    can = xs["val"][: max(cfg["canary_k"])]
    sess_plain = sessions["int8"]
    can_t = {
        str(K): B.time_call(lambda K=K: sess_plain.run([E.LOGITS], {E.INPUT: can[:K]}), runs=500, warmup=50)
        for K in cfg["canary_k"]
    }

    def init_bytes(m: onnx.ModelProto) -> int:
        return int(sum(numpy_helper.to_array(i).nbytes for i in m.graph.initializer))

    n_cls, d_feat = (int(v) for v in maha.means.shape)
    mc = E.macs(plain["fp32"])
    total_macs = int(mc["total"])
    maha_macs = n_cls * (d_feat * d_feat + d_feat)
    return {
        "host": B.host(),
        "latency_batch1": lat1,
        f"latency_per_chip_batch{nb}": lat_b,
        "decision_step_batch1": post1,
        f"decision_step_per_chip_batch{nb}": post_b,
        "crc_full_scan_us": crc_t * 1e6,
        "canary_check_us": can_t,
        "memory_bytes": {
            "fp32_initializers": init_bytes(plain["fp32"]),
            "int8_initializers": init_bytes(plain["int8"]),
            "int8_parameter_memory": faults["parameter_memory"]["bytes"],
            "fp32_onnx_file": (work / "edgenet_fp32.onnx").stat().st_size,
            "int8_onnx_file": (work / "edgenet_int8.onnx").stat().st_size,
            "radiometry_state_fp32": rad.state_bytes(np.float32),
            "mahalanobis_state_fp32": maha.state_bytes(np.float32),
            "mahalanobis_state_fp16": maha.state_bytes(np.float16),
            "feature_bounds_state_fp32": fb.state_bytes(np.float32),
            "feature_bounds_state_uint8": fb.state_bytes(np.uint8),
            "layer_bounds_state_uint8": lb.state_bytes(np.uint8),
            "fusion_thresholds_fp32": 4 * (len(LITE_COMPONENTS) + len(FULL_EXTRA)),
            "canaries_k8_inputs_uint8": 8 * D.CHIP * D.CHIP * 3,
            "canaries_k8_reference_logits": 8 * n_cls * 4,
            "crc_table": 4 * len(mem.names),
        },
        "operation_counts": {
            "network_macs": total_macs,
            "mahalanobis_macs": maha_macs,
            "mahalanobis_share_of_network": maha_macs / total_macs,
            "radiometry_input_values": D.CHIP * D.CHIP * 3,
            "feature_bounds_comparisons": 2 * fb.n_channels,
            "layer_bounds_comparisons": 2 * lb.n_channels,
            "layer_bounds_channels": lb.n_channels,
            # the per-channel maxima read every value the convolutions write
            "layer_tap_input_values": mc["conv_output_values"],
            "layer_tap_share_of_network": mc["conv_output_values"] / total_macs,
        },
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path("runs/v0.1"))
    ap.add_argument("--raw", type=Path, default=Path("data"))
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests (numbers are meaningless)")
    a = ap.parse_args(argv)

    cfg = json.loads(json.dumps(CFG))
    cfg["epochs"] = a.epochs
    if a.quick:
        cfg.update(
            epochs=1,
            calibration_samples=256,
            fault_eval_samples=256,
            uniform_k=[1, 16],
            uniform_trials={"1": 20, "default": 5},
        )
        for spec in cfg["targeted"].values():
            spec["trials"] = 5
    cfg["quick"] = bool(a.quick)
    work = a.work
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    stage_s = {}

    def mark(name: str, t0: float) -> float:
        stage_s[name] = time.perf_counter() - t0
        return time.perf_counter()

    t = time.perf_counter()
    log("data")
    ds, split, norm, cifar, data_info = stage_data(a.raw, cfg)
    x01 = {k: D.to01(ds.x[v]) for k, v in split.items()}
    xs = {k: norm(v) for k, v in x01.items()}
    ys = {k: ds.y[v] for k, v in split.items()}
    t = mark("data", t)

    log("train")
    net, train_info = stage_train(work, xs, ys, cfg)
    t = mark("train", t)

    log("export + quantise")
    plain, lite, full, export_info = stage_export(work, net, xs, cfg)
    equiv = check_equivalence(plain, lite, full, net, xs)
    log(f"equivalence: {equiv}")
    t = mark("export", t)

    sessions = {k: R.session(full[k]) for k in full}
    outs = {k: {s: R.run(sessions[k], xs[s]) for s in ("train", "val", "test")} for k in full}
    cls = classification(outs, ys)
    log(f"classification: {json.dumps(cls)}")

    log("fit monitors")
    mons = fit_monitors(outs, x01, ys, cfg["seed"])
    t = mark("monitors", t)

    log("OOD benchmark")
    ood = stage_ood(sessions, mons, norm, cifar, outs, x01, ys)
    t = mark("ood", t)

    log("bit-flip campaigns")
    faults = stage_faults(full["int8"], mons["int8"], xs, ys, outs["int8"], cfg)
    t = mark("faults", t)

    log("latency and memory")
    bench = stage_bench(plain, lite, full, mons, xs, x01, faults, cfg, work)
    t = mark("bench", t)

    import onnxruntime
    import scipy
    import sklearn
    import torch

    results = {
        "version": "0.1.0",
        "config": cfg,
        "detectors": DETECTORS,
        "fusion": {"lite": LITE_COMPONENTS, "full": LITE_COMPONENTS + FULL_EXTRA},
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "data": data_info,
        "training": train_info,
        "export": export_info,
        "equivalence": equiv,
        "classification": cls,
        "ood": ood,
        "faults": faults,
        "bench": bench,
        "stage_seconds": stage_s,
        "runtime_seconds": time.perf_counter() - t_start,
    }
    (work / "results.json").write_text(json.dumps(results, separators=(",", ":"), default=_json_default))
    log(f"done in {results['runtime_seconds'] / 60:.1f} min -> {work / 'results.json'}")
    return results


if __name__ == "__main__":
    main()
