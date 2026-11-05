"""Turn results.json into a Markdown summary, a metrics file and PNG figures.

python -m nerve_edge.report --work runs/v0.1 --out results
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np

from .metrics import wilson

LABELS = {
    "NerveLite": "Nerve (lite)",
    "NerveFull": "Nerve (full)",
    "Radiometry": "Radiometry",
    "Mahalanobis": "Mahalanobis",
    "FeatureBounds": "Feature bounds",
    "LayerBounds": "Layer bounds",
    "Energy": "Energy",
    "MaxLogit": "Max logit",
    "MSP": "MSP",
    "DeepKNN": "Deep k-NN",
}
OURS = ("NerveLite", "NerveFull")
OUTPUT_BASELINES = ("Energy", "MaxLogit", "MSP")
FPR_COLUMNS = ("NerveLite", "NerveFull", "Mahalanobis", "Energy", "MSP", "DeepKNN")
WINDOW_DETECTORS = ("layer_bounds", "feature_bounds", "mahalanobis")
WLABEL = {"layer_bounds": "layer bounds", "feature_bounds": "feature bounds", "mahalanobis": "Mahalanobis"}

# Reference palette of the dataviz method (light theme).
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d: int = 1) -> str:
    return "–" if _nan(x) else f"{100 * x:.{d}f} %"


def num(x, d: int = 3) -> str:
    return "–" if _nan(x) else f"{x:.{d}f}"


def mean(values) -> float:
    v = [float(x) for x in values]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------- #
# Out-of-distribution summaries
# --------------------------------------------------------------------------- #
def harmful_conditions(res: dict, k: str = "int8") -> list[str]:
    """Shifts that cost the classifier at least `harmful_drop_points` accuracy points."""
    drop = res["config"]["harmful_drop_points"] / 100
    clean = res["classification"][k]["accuracy"]
    return [
        c for c, e in res["ood"][k].items() if "accuracy_under_shift" in e and clean - e["accuracy_under_shift"] >= drop
    ]


def ood_summary(res: dict, k: str = "int8") -> dict:
    o = res["ood"][k]
    harm = harmful_conditions(res, k)
    shifts = [c for c in o if not c.startswith("far_ood")]
    far = [c for c in o if c.startswith("far_ood")]
    by = {}
    for d in res["detectors"]:
        a = {c: o[c]["detectors"][d]["auroc"] for c in o}
        f = {c: o[c]["detectors"][d]["fpr95"] for c in o}
        by[d] = {
            "far_ood_auroc": mean(a[c] for c in far),
            "far_ood_fpr95": mean(f[c] for c in far),
            "mean_auroc_all_shifts": mean(a[c] for c in shifts),
            "mean_auroc_harmful_shifts": mean(a[c] for c in harm),
            "min_auroc_harmful_shifts": min((a[c] for c in harm), default=float("nan")),
            "mean_fpr95_harmful_shifts": mean(f[c] for c in harm),
        }
    return {"harmful_conditions": harm, "by_detector": by}


# --------------------------------------------------------------------------- #
# Fault summaries
# --------------------------------------------------------------------------- #
def layer_names(res: dict) -> dict[str, str]:
    """ONNX node name -> conv1..convN / fc, in graph order.

    The order comes from the multiply-accumulate count of the FP32 graph, which
    walks the nodes in graph order; the INT8 graph keeps the same node names.
    (The parameter tensors themselves are listed in the order of their
    DequantizeLinear nodes, which is not the order of the layers.)
    """
    names, conv = {}, 0
    for n, _ in res["export"]["macs"]["per_layer"]:
        if "gemm" in n.lower() or "matmul" in n.lower():
            names[n] = "fc"
        else:
            conv += 1
            names[n] = f"conv{conv}"
    for t in res["faults"]["tensors"]:
        names.setdefault(t["layer"], t["layer"])
    return names


def bias_involvement(res: dict) -> list[dict]:
    """Uniform campaign: impact of the trials that flip at least one bias bit vs those that spare the biases."""
    kinds = [t["kind"] for t in res["faults"]["tensors"]]
    major = res["config"]["major_change"]
    rows = []
    tr = [t for t in res["faults"]["trials"] if t["campaign"] == "uniform"]
    for k in sorted({t["k"] for t in tr}):
        hit, spare = [], []
        for t in tr:
            if t["k"] == k:
                (hit if any(kinds[f[0]] == "bias" for f in t["flips"]) else spare).append(t)
        rows.append(
            {
                "k": k,
                "trials_hitting_a_bias": len(hit),
                "major_hitting_a_bias": int(sum(t["change"] > major for t in hit)),
                "trials_sparing_biases": len(spare),
                "major_sparing_biases": int(sum(t["change"] > major for t in spare)),
                "any_change_sparing_biases": mean(t["change"] > 0 for t in spare),
            }
        )
    return rows


def summarise(trials: list[dict], cfg: dict) -> dict:
    major = cfg["major_change"]
    big = [t for t in trials if t["change"] > major]
    n, nb = len(trials), len(big)
    row = {
        "trials": n,
        "any_change": mean(t["change"] > 0 for t in trials),
        "n_major": nb,
        "major": nb / n if n else float("nan"),
        "major_ci95": list(wilson(nb, n)),
        "worst_accuracy": min((t["accuracy"] for t in trials), default=float("nan")),
        "mean_accuracy_major": mean(t["accuracy"] for t in big),
        "nonfinite_outputs": int(sum(t["nonfinite"] for t in trials)),
        "crc": mean(t["crc"] for t in trials),
        "crc_major": mean(t["crc"] for t in big),
    }
    for K in cfg["canary_k"]:
        row[f"canary_{K}"] = mean(t["canary"][str(K)] for t in trials)
        row[f"canary_{K}_major"] = mean(t["canary"][str(K)] for t in big)
    for d in WINDOW_DETECTORS:
        for w in cfg["windows"]:
            row[f"{d}_{w}"] = mean(t[d][str(w)] for t in trials)
            row[f"{d}_{w}_major"] = mean(t[d][str(w)] for t in big)
    return row


def fault_table(res: dict) -> list[dict]:
    """Uniform campaign, by number of flipped bits."""
    tr = [t for t in res["faults"]["trials"] if t["campaign"] == "uniform"]
    return [{"k": k, **summarise([t for t in tr if t["k"] == k], res["config"])} for k in sorted({t["k"] for t in tr})]


def targeted_table(res: dict) -> list[dict]:
    rows = []
    for name, spec in res["config"]["targeted"].items():
        tr = [t for t in res["faults"]["trials"] if t["campaign"] == name]
        rows.append({"campaign": name, "kind": spec["kind"], "bits": spec["bits"], **summarise(tr, res["config"])})
    return rows


def bit_table(res: dict) -> list[dict]:
    """Single flips of the uniform campaign, by parameter kind and bit position."""
    tensors = res["faults"]["tensors"]
    single = [t for t in res["faults"]["trials"] if t["campaign"] == "uniform" and t["k"] == 1]
    groups: dict[tuple[str, str], list] = {}
    for t in single:
        ti, _, bit, _ = t["flips"][0]
        if tensors[ti]["kind"] == "weight":
            key = ("weight", str(bit))
        else:
            key = ("bias", "0-15" if bit < 16 else "16-23" if bit < 24 else "24-31")
        groups.setdefault(key, []).append(t)
    order = [("weight", str(b)) for b in range(8)] + [("bias", g) for g in ("0-15", "16-23", "24-31")]
    return [{"kind": key[0], "bits": key[1], **summarise(groups[key], res["config"])} for key in order if key in groups]


def layer_table(res: dict, campaign: str) -> list[dict]:
    tensors = res["faults"]["tensors"]
    names = layer_names(res)
    tr = [t for t in res["faults"]["trials"] if t["campaign"] == campaign]
    groups: dict[str, list] = {}
    for t in tr:
        groups.setdefault(names[tensors[t["flips"][0][0]]["layer"]], []).append(t)
    order = [n for n in dict.fromkeys(names.values()) if n in groups]
    return [{"layer": n, **summarise(groups[n], res["config"])} for n in order]


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _ratio(a: float, b: float) -> float:
    return a / b - 1.0


def markdown(res: dict) -> str:
    cfg = res["config"]
    c = res["classification"]
    b = res["bench"]
    lat1 = b["latency_batch1"]
    latb_key = next(k for k in b if k.startswith("latency_per_chip_batch"))
    latb = b[latb_key]
    nb = int(latb_key.rsplit("batch", 1)[1])
    dec1 = b["decision_step_batch1"]
    decb = b[f"decision_step_per_chip_batch{nb}"]
    mem = b["memory_bytes"]
    ops = b["operation_counts"]
    d = res["data"]
    fz = res["faults"]
    o = res["ood"]["int8"]
    summ = ood_summary(res, "int8")
    by = summ["by_detector"]
    harm = summ["harmful_conditions"]
    dets = res["detectors"]
    ft, tt = fault_table(res), targeted_table(res)
    tmap = {r["campaign"]: r for r in tt}
    k1 = next((r for r in ft if r["k"] == 1), None)
    lite_state = mem["radiometry_state_fp32"] + mem["mahalanobis_state_fp32"] + mem["feature_bounds_state_fp32"] + 12
    lite_state16 = mem["radiometry_state_fp32"] + mem["mahalanobis_state_fp16"] + mem["feature_bounds_state_uint8"] + 12

    L = ["# Nerve Edge Decision v0.1 — results\n"]
    if cfg.get("quick"):
        L.append(
            "> **Smoke-test run (`--quick`): one training epoch and a handful of fault trials. The numbers below are meaningless.**\n"
        )
    L.append(
        f"Generated by `python -m nerve_edge.report` from `results.json` (seed {cfg['seed']}). "
        f"Host: {b['host']['cpu']}, ONNX Runtime {b['host']['onnxruntime']}, one thread. "
        f"Pipeline run time: {res['runtime_seconds'] / 60:.0f} min.\n"
    )

    # ---------------------------------------------------------------- glance
    best_out = max(OUTPUT_BASELINES, key=lambda x: by[x]["mean_auroc_harmful_shifts"])
    missed = []
    for cn in harm:
        best_det = max(dets, key=lambda x: o[cn]["detectors"][x]["auroc"])
        if o[cn]["detectors"][best_det]["auroc"] < 0.7:
            missed.append((cn, best_det, o[cn]["detectors"][best_det]["auroc"]))
    L.append("## At a glance\n")
    L.append(
        f"- **Classifier.** INT8 accuracy {pct(c['int8']['accuracy'], 2)} on the scene-held-out test split "
        f"(FP32 {pct(c['fp32']['accuracy'], 2)}, {c['int8_minus_fp32_accuracy_points']:+.2f} points); the two agree on "
        f"{pct(c['fp32_int8_agreement'], 2)} of the {c['int8']['n']:,} test chips. Parameters: "
        f"{mem['int8_initializers']:,} bytes in INT8 vs {mem['fp32_initializers']:,} in FP32."
    )
    line = (
        f"- **Unusual inputs.** {len(harm)} of the {len([x for x in o if not x.startswith('far_ood')])} simulated sensor "
        f"faults cost the INT8 classifier at least {cfg['harmful_drop_points']:.0f} accuracy points. On those, scores "
        f"built on the softmax output do worse than chance (mean AUROC {num(by[best_out]['mean_auroc_harmful_shifts'])} "
        f"for {LABELS[best_out]}, the best of the three): a corrupted chip tends to get a more confident output. The full "
        f"Nerve Edge Decision score reaches {num(by['NerveFull']['mean_auroc_harmful_shifts'])} and the lite score "
        f"{num(by['NerveLite']['mean_auroc_harmful_shifts'])}; radiometry alone, the simplest component, reaches "
        f"{num(by['Radiometry']['mean_auroc_harmful_shifts'])} because exposure faults weigh heavily in this benchmark, "
        f"and Mahalanobis alone {num(by['Mahalanobis']['mean_auroc_harmful_shifts'])}. Far-OOD inputs (CIFAR-10): "
        f"{num(by['NerveLite']['far_ood_auroc'])} (lite)."
    )
    if missed:
        parts = []
        for cn, bd, v in missed:
            name, sev = cn.split("/")
            parts.append(f"{sev} {name.replace('_', ' ')} (best AUROC {v:.2f}, {LABELS[bd]})")
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]
        line += f" No signal reaches 0.7 on {joined}."
    L.append(line)
    bi = bias_involvement(res)
    if k1 is not None and "bias_high" in tmap:
        bh = tmap["bias_high"]
        n_major = sum(r["major_hitting_a_bias"] + r["major_sparing_biases"] for r in bi)
        n_major_hit = sum(r["major_hitting_a_bias"] for r in bi)
        n_spare = sum(r["trials_sparing_biases"] for r in bi)
        n_major_spare = sum(r["major_sparing_biases"] for r in bi)
        pm = fz["parameter_memory"]
        line = (
            f"- **Bit flips.** {pct(1 - k1['any_change'])} of single random flips change no prediction at all. "
            f"{'None' if n_major_spare == 0 else n_major_spare} of the {n_spare:,} uniform trials whose flips all fell "
            f"in the INT8 weights had a major impact (more than {pct(cfg['major_change'], 0)} of predictions changed), "
            f"with up to {max(cfg['uniform_k'])} flipped bits; "
            f"{'all' if n_major_hit == n_major else f'{n_major_hit} of the'} {n_major} major-impact trials flipped at "
            f"least one bit of an INT32 bias. The {pm['biases_bytes']:,} bytes of biases "
            f"({pct(pm['biases_bytes'] / pm['bytes'])} of the parameter memory) are where hardening pays in this network. "
            f"A flip in bits 24–31 of a bias has a major impact in {pct(bh['major'])} of cases "
            f"(accuracy down to {pct(bh['worst_accuracy'])}); in those cases the online layer bounds flag "
            f"{pct(bh['layer_bounds_1_major'])} of single inferences without a reference copy of the weights, one golden "
            f"input catches {pct(bh['canary_1_major'])}, and a CRC-32 scrub catches every flip in protected memory by "
            f"construction."
        )
        if "weight_sign" in tmap:
            ws = tmap["weight_sign"]
            line += f" Flipping the sign bit of an INT8 weight had a major impact in {ws['n_major']} of {ws['trials']} trials."
        L.append(line)
    L.append(
        f"- **Cost.** INT8 inference is {lat1['fp32']['median_us'] / lat1['int8']['median_us']:.1f}× faster than FP32 "
        f"at batch 1 on this CPU. The lite monitor reads only the input chip and the pooled features the network computes "
        f"anyway; its effect on latency is within measurement noise "
        f"({_ratio(lat1['int8_lite']['median_us'], lat1['int8']['median_us']) * 100:+.1f} % at batch 1, "
        f"{_ratio(latb['int8_lite']['median_us'], latb['int8']['median_us']) * 100:+.1f} % per chip at batch {nb}). Its "
        f"decision step is {ops['mahalanobis_macs']:,} multiply-accumulates "
        f"({pct(ops['mahalanobis_share_of_network'], 2)} of the network), {ops['feature_bounds_comparisons']} bound checks "
        f"and one pass over the {ops['radiometry_input_values']:,} input values; its state is {lite_state:,} bytes "
        f"({lite_state16:,} with a float16 precision matrix and 8-bit bounds). The full monitor's "
        f"{ops['layer_bounds_channels']} per-channel maxima cost "
        f"{_ratio(lat1['int8_full']['median_us'], lat1['int8']['median_us']) * 100:+.1f} % at batch 1 and "
        f"{_ratio(latb['int8_full']['median_us'], latb['int8']['median_us']) * 100:+.1f} % per chip at batch {nb} on ONNX "
        f"Runtime, whose separate ReduceMax operators re-read the {ops['layer_tap_input_values']:,} values the "
        f"convolutions write; computed inside the convolution kernels they would be {ops['layer_tap_input_values']:,} "
        f"comparisons per chip ({pct(ops['layer_tap_share_of_network'], 2)} of the network's multiply-accumulates; not "
        f"measured here).\n"
    )

    # ---------------------------------------------------------------- data
    L.append("## Data\n")
    L.append("| Split | Chips | Planes | Scenes |\n|---|---:|---:|---:|")
    for k in ("train", "val", "test"):
        s = d["splits"][k]
        L.append(f"| {k} | {s['chips']:,} | {s['planes']:,} ({pct(s['plane_share'])}) | {s['scenes']} |")
    ov = d["scene_overlap"]
    L.append(
        f"\nScenes shared between splits: train/test {ov['train_test']}, train/val {ov['train_val']}, "
        f"val/test {ov['val_test']}. Far-OOD set: {d['far_ood_cifar10']:,} CIFAR-10 test images resized to 20 x 20. "
        f"Training: {res['training']['params']:,} parameters, best validation accuracy "
        f"{pct(res['training']['best_val_acc'], 2)} at epoch {res['training']['best_epoch']} of {cfg['epochs']}.\n"
    )

    # ---------------------------------------------------------------- classification
    L.append("## Classification, FP32 vs INT8 (test split)\n")
    L.append("| Model | Accuracy | Balanced accuracy | Plane recall | False-alarm rate |\n|---|---:|---:|---:|---:|")
    for k in ("fp32", "int8"):
        r = c[k]
        L.append(
            f"| {k.upper()} | {pct(r['accuracy'], 2)} | {pct(r['balanced_accuracy'], 2)} | "
            f"{pct(r['plane_recall'], 2)} | {pct(r['false_alarm_rate'], 2)} |"
        )
    q = res["export"]["quantization"]
    eq = res["equivalence"]
    L.append(
        f"\nINT8: static quantisation with ONNX Runtime ({q['quant_format']}, per-channel symmetric INT8 weights, "
        f"asymmetric UINT8 activations, {q['calibrate_method']} calibration on {q['calibration_samples']:,} training chips). "
        f"Adding the monitor outputs leaves the logits bit-identical (FP32: {eq['fp32_logits_bit_identical']}, "
        f"INT8: {eq['int8_logits_bit_identical']}); ONNX FP32 vs PyTorch max |Δlogit| = "
        f"{eq['onnx_fp32_vs_torch_max_abs_diff']:.1e}.\n"
    )

    # ---------------------------------------------------------------- OOD
    L.append("## Out-of-distribution inputs (INT8 model, clean test split as in-distribution)\n")
    L.append(
        "AUROC, higher is better (0.5 = chance). The best detector of each row is in bold. 'Accuracy' is the classifier's "
        f"accuracy on the shifted chips (clean: {pct(c['int8']['accuracy'])}); rows marked * cost at least "
        f"{cfg['harmful_drop_points']:.0f} points. Nerve (lite) fuses {', '.join(LABELS[x] for x in res['fusion']['lite'])}; "
        f"the full variant adds {', '.join(LABELS[x] for x in res['fusion']['full'] if x not in res['fusion']['lite'])}.\n"
    )
    L.append("| Condition | Accuracy | " + " | ".join(LABELS[x] for x in dets) + " |")
    L.append("|---|---:|" + "---:|" * len(dets))
    for cname, e in o.items():
        acc = pct(e["accuracy_under_shift"]) if "accuracy_under_shift" in e else "–"
        vals = [e["detectors"][x]["auroc"] for x in dets]
        best = max(vals)
        cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
        mark = " *" if cname in harm else ""
        L.append(f"| {cname}{mark} | {acc} | " + " | ".join(cells) + " |")
    L.append(
        "| **mean, all shifts** | | "
        + " | ".join(f"{by[x]['mean_auroc_all_shifts']:.3f}" for x in dets)
        + " |\n| **mean, harmful shifts (*)** | | "
        + " | ".join(f"**{by[x]['mean_auroc_harmful_shifts']:.3f}**" for x in dets)
        + " |"
    )
    L.append(
        "\nFPR at 95 % TPR (share of shifted chips still accepted when 95 % of clean chips are accepted; lower is better):\n"
    )
    L.append("| Condition | " + " | ".join(LABELS[x] for x in FPR_COLUMNS) + " |")
    L.append("|---|" + "---:|" * len(FPR_COLUMNS))
    for cname, e in o.items():
        mark = " *" if cname in harm else ""
        L.append(f"| {cname}{mark} | " + " | ".join(f"{e['detectors'][x]['fpr95']:.2f}" for x in FPR_COLUMNS) + " |")
    L.append(
        "| **mean, harmful shifts (*)** | "
        + " | ".join(f"**{by[x]['mean_fpr95_harmful_shifts']:.2f}**" for x in FPR_COLUMNS)
        + " |"
    )
    f32 = res["ood"]["fp32"]
    delta = {
        x: [o[cn]["detectors"][x]["auroc"] - f32[cn]["detectors"][x]["auroc"] for cn in o]
        for x in ("NerveLite", "Mahalanobis", "Energy")
    }
    L.append(
        f"\nQuantisation and the monitor: INT8 minus FP32 AUROC over the {len(o)} conditions — "
        + "; ".join(
            f"{LABELS[x]} mean {np.mean(v):+.3f} (min {np.min(v):+.3f}, max {np.max(v):+.3f})" for x, v in delta.items()
        )
        + ".\n"
    )

    # ---------------------------------------------------------------- faults
    pm = fz["parameter_memory"]
    L.append("## Bit flips in the INT8 parameter memory\n")
    L.append(
        f"Fault model: bit flips in the memory holding the quantised parameters, {pm['bytes']:,} bytes "
        f"({pm['weights_bytes']:,} of INT8 weights, {pm['biases_bytes']:,} of INT32 biases, {pm['bits']:,} bits). "
        f"Each trial flips bits, rebuilds the model and runs {fz['eval_chips']:,} test chips plus {fz['canary_chips']} "
        f"golden inputs. 'Major' means more than {pct(cfg['major_change'], 0)} of the predictions change. Detection rates of "
        f"the online detectors are the share of windows of W consecutive inferences that raise an alarm, with thresholds set "
        f"for {pct(cfg['window_far'], 0)} false alarms per window on the clean model ({fz['window_calibration_chips']:,} test "
        f"chips not used in the campaign). Measured false-alarm rates on the campaign chips: "
        + "; ".join(
            f"{WLABEL[x]} " + ", ".join(f"W={w}: {pct(v)}" for w, v in fz["clean_false_alarm_rate"][x].items())
            for x in WINDOW_DETECTORS
        )
        + ". CRC-32 and golden inputs are reference-based checks; the bounds and Mahalanobis need no reference.\n"
    )
    L.append("### Uniform random flips\n")
    L.append(
        "Every bit of the parameter memory equally likely. Detection columns: share of trials with major impact that are "
        "detected (golden inputs) or share of windows flagged on those trials (online detectors).\n"
    )
    L.append(
        "| Bits flipped | Trials | Any change | Major (95 % CI) | CRC-32 | Golden K=1 | Golden K=8 | Layer bounds W=1 | "
        "Layer bounds W=32 | Feature bounds W=32 | Mahalanobis W=32 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in ft:
        lo, hi = r["major_ci95"]
        L.append(
            f"| {r['k']} | {r['trials']:,} | {pct(r['any_change'])} | {pct(r['major'])} ({pct(lo)}–{pct(hi)}), n={r['n_major']} | "
            f"{pct(r['crc'])} | {pct(r['canary_1_major'])} | {pct(r['canary_8_major'])} | {pct(r['layer_bounds_1_major'])} | "
            f"{pct(r['layer_bounds_32_major'])} | {pct(r['feature_bounds_32_major'])} | {pct(r['mahalanobis_32_major'])} |"
        )
    L.append(
        "\nWhere the damage comes from: trials that flip at least one bit of an INT32 bias vs trials that spare the biases.\n"
    )
    L.append(
        "| Bits flipped | Trials hitting a bias | Major among them | Trials sparing the biases | Major among them | "
        "Any change among them |\n|---:|---:|---:|---:|---:|---:|"
    )
    for r in bias_involvement(res):
        L.append(
            f"| {r['k']} | {r['trials_hitting_a_bias']:,} | {r['major_hitting_a_bias']} | {r['trials_sparing_biases']:,} | "
            f"{r['major_sparing_biases']} | {pct(r['any_change_sparing_biases'])} |"
        )
    L.append("\n### Targeted single flips (the most damaging bits)\n")
    L.append(
        "| Campaign | Trials | Any change | Major (95 % CI) | Worst accuracy | Golden K=1 | Golden K=8 | Layer bounds W=1 | "
        "Layer bounds W=32 | Feature bounds W=32 | Mahalanobis W=32 | Layer bounds W=1, all trials |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    names = {"weight_sign": "INT8 weight, sign bit (bit 7)", "bias_high": "INT32 bias, bits 24–31"}
    for r in tt:
        lo, hi = r["major_ci95"]
        L.append(
            f"| {names.get(r['campaign'], r['campaign'])} | {r['trials']:,} | {pct(r['any_change'])} | "
            f"{pct(r['major'])} ({pct(lo)}–{pct(hi)}), n={r['n_major']} | {pct(r['worst_accuracy'])} | "
            f"{pct(r['canary_1_major'])} | {pct(r['canary_8_major'])} | {pct(r['layer_bounds_1_major'])} | "
            f"{pct(r['layer_bounds_32_major'])} | {pct(r['feature_bounds_32_major'])} | {pct(r['mahalanobis_32_major'])} | "
            f"{pct(r['layer_bounds_1'])} |"
        )
    for camp in ("weight_sign", "bias_high"):
        if camp not in tmap:
            continue
        L.append(f"\n{names[camp]}, by layer:\n")
        L.append(
            "| Layer | Trials | Major | Layer bounds W=32 on major | Golden K=8 on major |\n|---|---:|---:|---:|---:|"
        )
        for r in layer_table(res, camp):
            L.append(
                f"| {r['layer']} | {r['trials']} | {pct(r['major'])} (n={r['n_major']}) | "
                f"{pct(r['layer_bounds_32_major'])} | {pct(r['canary_8_major'])} |"
            )
    L.append("\n### Single uniform flips by bit position\n")
    L.append(
        "| Parameter | Bit(s) | Trials | Any change | Major (95 % CI) | Layer bounds W=32 on major |\n|---|---|---:|---:|---:|---:|"
    )
    for r in bit_table(res):
        lo, hi = r["major_ci95"]
        L.append(
            f"| {r['kind']} | {r['bits']} | {r['trials']} | {pct(r['any_change'])} | "
            f"{pct(r['major'])} ({pct(lo)}–{pct(hi)}) | {pct(r['layer_bounds_32_major'])} |"
        )
    fv = fz["false_alarm_rate_validation_thresholds_on_test"]
    L.append(
        "\nThreshold transfer across scenes: with window thresholds calibrated on the validation scenes instead, the "
        "false-alarm rates on the clean test scenes become "
        + "; ".join(f"{WLABEL[x]} " + ", ".join(f"W={w}: {pct(v)}" for w, v in fv[x].items()) for x in WINDOW_DETECTORS)
        + f" (target {pct(cfg['window_far'], 0)}).\n"
    )

    # ---------------------------------------------------------------- cost
    L.append("## Cost\n")
    L.append(
        f"Host: {b['host']['cpu']}, ONNX Runtime {b['host']['onnxruntime']}, one thread; median of interleaved runs.\n"
    )
    L.append("| | FP32 | INT8 |\n|---|---:|---:|")
    L.append(f"| Parameters in the graph (bytes) | {mem['fp32_initializers']:,} | {mem['int8_initializers']:,} |")
    L.append(f"| ONNX file (bytes) | {mem['fp32_onnx_file']:,} | {mem['int8_onnx_file']:,} |")
    for v, lab in (
        ("", "network only"),
        ("_lite", "with lite monitor outputs"),
        ("_full", "with full monitor outputs"),
    ):
        L.append(
            f"| Latency, batch 1, {lab} (µs) | {lat1['fp32' + v]['median_us']:.1f} | {lat1['int8' + v]['median_us']:.1f} |"
        )
    for v, lab in (("", "network only"), ("_lite", "lite"), ("_full", "full")):
        L.append(
            f"| Per chip at batch {nb}, {lab} (µs) | {latb['fp32' + v]['median_us']:.2f} | {latb['int8' + v]['median_us']:.2f} |"
        )
    L.append(
        f"\nDecision step of the monitor (INT8): {ops['mahalanobis_macs']:,} MACs for Mahalanobis "
        f"({pct(ops['mahalanobis_share_of_network'], 2)} of the network's {ops['network_macs']:,}), "
        f"{ops['feature_bounds_comparisons']} feature-bound checks, {ops['layer_bounds_comparisons']} layer-bound checks (full only), "
        f"one pass over {ops['radiometry_input_values']:,} input values, and one threshold per fused component. "
        f"Python reference timings: {decb['lite']['median_us']:.2f} µs (lite) / {decb['full']['median_us']:.2f} µs (full) per "
        f"chip vectorised over {nb} chips; {dec1['lite']['median_us']:.1f} / {dec1['full']['median_us']:.1f} µs at batch 1, "
        f"where NumPy call overhead dominates. The per-channel maxima of the full monitor read "
        f"{ops['layer_tap_input_values']:,} activation values per chip; ONNX Runtime does it with separate ReduceMax "
        f"operators (the latency rows above).\n"
    )
    L.append(
        f"Monitor state: radiometry {mem['radiometry_state_fp32']} bytes, Mahalanobis {mem['mahalanobis_state_fp32']:,} bytes in "
        f"float32 ({mem['mahalanobis_state_fp16']:,} in float16), feature bounds {mem['feature_bounds_state_fp32']:,} bytes "
        f"({mem['feature_bounds_state_uint8']} as INT8 codes), layer bounds {mem['layer_bounds_state_uint8']} bytes, fusion "
        f"thresholds {mem['fusion_thresholds_fp32']} bytes. Reference-based checks: CRC table {mem['crc_table']} bytes, "
        f"8 golden inputs {mem['canaries_k8_inputs_uint8'] + mem['canaries_k8_reference_logits']:,} bytes.\n"
    )
    L.append(
        f"Integrity checks: full CRC-32 scrub of the parameter memory {b['crc_full_scan_us']:.0f} µs; golden-input check "
        + ", ".join(f"K={k}: {v['median_us']:.0f} µs" for k, v in b["canary_check_us"].items())
        + ".\n"
    )
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(res: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    paths = []
    dets = res["detectors"]
    summ = ood_summary(res, "int8")
    harm = set(summ["harmful_conditions"])

    # 1. OOD AUROC heatmap (INT8)
    o = res["ood"]["int8"]
    conds = list(o)
    mat = np.array([[o[cn]["detectors"][x]["auroc"] for x in dets] for cn in conds])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(conds) + 2.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(dets)), [LABELS[x] for x in dets], fontsize=9, color=INK2, rotation=30, ha="right")
    ax.set_yticks(
        range(len(conds)),
        [cn.replace("/", " · ").replace("_", " ") + (" *" if cn in harm else "") for cn in conds],
        fontsize=9,
        color=INK2,
    )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(conds)):
        for j in range(len(dets)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.82 else INK)
    ax.set_title(
        "AUROC on the INT8 model (0.5 = chance; values below 0.5 shown at the lightest step)\n"
        "* shift costs the classifier at least "
        f"{res['config']['harmful_drop_points']:.0f} accuracy points",
        fontsize=10,
        color=INK,
        loc="left",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK2)
    fig.tight_layout()
    p = out / "ood_auroc_int8.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 2. Mean AUROC on harmful shifts, per detector
    by = summ["by_detector"]
    order = sorted(dets, key=lambda x: by[x]["mean_auroc_harmful_shifts"])
    vals = [by[x]["mean_auroc_harmful_shifts"] for x in order]
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(order) + 1.3))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(order))
    ax.barh(ys, vals, height=0.62, color=[SERIES[0] if x in OURS else MUTED for x in order])
    for y, v in zip(ys, vals, strict=True):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, len(order) - 0.45, "chance", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(ys, [LABELS[x] for x in order], fontsize=9, color=INK2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("mean AUROC over the shifts that cost ≥ 3 accuracy points", fontsize=9, color=INK2)
    ax.set_title(
        f"Detecting the harmful sensor faults ({len(harm)} conditions, INT8)", fontsize=10, color=INK, loc="left"
    )
    fig.tight_layout()
    p = out / "ood_harmful_summary.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 3. Faults: impact and detection vs number of flipped bits
    rows = fault_table(res)
    ks = [r["k"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax in (a1, a2):
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
        ax.set_xlabel("bits flipped at random in the parameter memory", fontsize=9, color=INK2)
    a1.plot(
        ks,
        [100 * r["any_change"] for r in rows],
        color=SERIES[0],
        lw=2,
        marker="o",
        ms=6,
        label="any prediction changes",
    )
    a1.plot(
        ks,
        [100 * r["major"] for r in rows],
        color=SERIES[1],
        lw=2,
        marker="o",
        ms=6,
        label="> 1 % of predictions change",
    )
    a1.set_ylabel("% of trials", fontsize=9, color=INK2)
    a1.set_title("Impact on the classifier", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    # Reference checks (CRC-32, golden inputs) are left out of the plot: they are stated in the title.
    ref_full = all(r["crc_major"] == 1.0 and r["canary_8_major"] == 1.0 for r in rows if r["n_major"])
    series = [
        ("layer bounds, single inference", "layer_bounds_1_major"),
        ("layer bounds, W = 32", "layer_bounds_32_major"),
        ("feature bounds, W = 32", "feature_bounds_32_major"),
        ("Mahalanobis, W = 32", "mahalanobis_32_major"),
    ]
    for i, (label, key) in enumerate(series):
        a2.plot(ks, [100 * r[key] for r in rows], color=SERIES[i], lw=2, marker="o", ms=6, label=label)
    a2.set_xticks(ks, [f"{r['k']}\nn={r['n_major']}" for r in rows])
    a2.set_xlabel("bits flipped (n = trials with major impact)", fontsize=9, color=INK2)
    a2.set_ylabel("% of inferences or windows flagged", fontsize=9, color=INK2)
    a2.set_title(
        "Online detection on trials with major impact"
        + ("\nCRC-32 scrub and 8 golden inputs: 100 % at every point" if ref_full else ""),
        fontsize=10,
        color=INK,
        loc="left",
    )
    a2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    p = out / "faults_vs_flips.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # 4. Where the damage comes from: major impact with and without a flipped bias bit
    bi = bias_involvement(res)
    if bi:
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        _style(ax)
        ax.grid(axis="x", visible=False)
        xs = np.arange(len(bi))
        wdt = 0.36
        hit = [
            100 * r["major_hitting_a_bias"] / r["trials_hitting_a_bias"] if r["trials_hitting_a_bias"] else 0
            for r in bi
        ]
        spare = [
            100 * r["major_sparing_biases"] / r["trials_sparing_biases"] if r["trials_sparing_biases"] else 0
            for r in bi
        ]
        ax.bar(xs - wdt / 2 - 0.01, hit, width=wdt, color=SERIES[1], label="flips hit at least one INT32 bias")
        ax.bar(xs + wdt / 2 + 0.01, spare, width=wdt, color=SERIES[0], label="all flips in INT8 weights")
        top = max(hit + [10.0])
        for x, r, h, s in zip(xs, bi, hit, spare, strict=True):
            ax.text(
                x - wdt / 2 - 0.01,
                h + top * 0.02,
                f"{r['major_hitting_a_bias']}/{r['trials_hitting_a_bias']}",
                ha="center",
                fontsize=7,
                color=INK2,
            )
            ax.text(
                x + wdt / 2 + 0.01,
                s + top * 0.02,
                f"{r['major_sparing_biases']}/{r['trials_sparing_biases']:,}",
                ha="center",
                fontsize=7,
                color=INK2,
            )
        ax.set_xticks(xs, [str(r["k"]) for r in bi])
        ax.set_xlabel("bits flipped at random in the parameter memory", fontsize=9, color=INK2)
        ax.set_ylabel("% of trials with major impact", fontsize=9, color=INK2)
        ax.set_ylim(0, top * 1.2)
        ax.set_title("Where the damage comes from (uniform campaign)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
        fig.tight_layout()
        p = out / "damage_by_memory.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("runs/v0.1"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    res = json.loads((a.work / "results.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "figures").mkdir(exist_ok=True)
    (a.out / "summary.md").write_text(markdown(res))
    figs = figures(res, a.out / "figures")
    slim = {k: v for k, v in res.items() if k != "faults"}
    slim["faults"] = {k: v for k, v in res["faults"].items() if k != "trials"}
    slim["faults"]["uniform_by_flips"] = fault_table(res)
    slim["faults"]["targeted"] = targeted_table(res)
    slim["faults"]["targeted_by_layer"] = {cn: layer_table(res, cn) for cn in res["config"]["targeted"]}
    slim["faults"]["uniform_single_by_bit"] = bit_table(res)
    slim["faults"]["uniform_bias_involvement"] = bias_involvement(res)
    slim["ood_summary"] = {k: ood_summary(res, k) for k in ("fp32", "int8")}
    (a.out / "metrics.json").write_text(json.dumps(slim, indent=1) + "\n")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="fault_trials.jsonl", mode="wb", fileobj=buf, mtime=0) as gz:
        for t in res["faults"]["trials"]:
            gz.write((json.dumps(t, separators=(",", ":")) + "\n").encode())
    (a.out / "fault_trials.jsonl.gz").write_bytes(buf.getvalue())
    print(
        "\n".join(
            str(p) for p in [a.out / "summary.md", a.out / "metrics.json", a.out / "fault_trials.jsonl.gz", *figs]
        )
    )


if __name__ == "__main__":
    main()
