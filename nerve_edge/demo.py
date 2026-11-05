"""Interactive demo: degrade the sensor or flip bits in the INT8 weights, and watch the monitor decide.

    python -m nerve_edge.demo            # then open http://127.0.0.1:7860

Needs the `demo` extra (`pip install -e ".[demo]"`). PlanesNet is downloaded on
first start; the monitors are fitted in a few seconds.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import CORRUPTIONS
from .live import CLASS_NAMES, COMPONENTS, FAR, WINDOW, LiveMonitor

LABELS = {
    "Radiometry": "Radiometry (input statistics)",
    "Mahalanobis": "Mahalanobis (feature space)",
    "FeatureBounds": "Feature bounds",
    "LayerBounds": "Layer bounds (full only)",
}
FAULTS = ["none"] + list(CORRUPTIONS)
FAULT_LABELS = {
    "none": "No fault",
    "noise": "Sensor noise",
    "dead_pixels": "Dead / hot pixels",
    "stripes": "Column stripes",
    "saturation": "Saturation (gain)",
    "low_light": "Low light",
    "blur": "Defocus blur",
    "band_shift": "Band misregistration",
    "cloud": "Cloud",
}
GREEN, RED, INK2 = "#1baf7a", "#e34948", "#52514e"

CSS = """
.ne-cards {display:flex; gap:12px; flex-wrap:wrap}
.ne-card {flex:1 1 200px; border-radius:12px; padding:14px 16px; border:1px solid rgba(127,127,127,.25)}
.ne-card h4 {margin:0 0 4px 0; font-size:13px; font-weight:600; opacity:.75}
.ne-card .v {font-size:26px; font-weight:700; letter-spacing:.5px}
.ne-card .d {font-size:13px; opacity:.8; margin-top:4px}
.ne-ok {background:rgba(27,175,122,.12); border-color:rgba(27,175,122,.55)}
.ne-ok .v {color:#14895f}
.ne-bad {background:rgba(227,73,72,.12); border-color:rgba(227,73,72,.6)}
.ne-bad .v {color:#c43534}
.ne-neutral .v {font-size:22px}
.ne-bar {display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px}
.ne-bar .lab {width:230px}
.ne-bar .track {flex:1; height:12px; border-radius:6px; background:rgba(127,127,127,.18); position:relative; overflow:hidden}
.ne-bar .fill {height:100%; border-radius:6px}
.ne-bar .thr {position:absolute; top:-2px; bottom:-2px; width:2px; background:#0b0b0b; left:33.3%}
.ne-bar .num {width:62px; text-align:right; font-variant-numeric:tabular-nums}
.ne-table {width:100%; border-collapse:collapse; font-size:13px}
.ne-table th, .ne-table td {padding:6px 8px; border-bottom:1px solid rgba(127,127,127,.25); text-align:left}
"""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _chip(x01: np.ndarray, scale: int = 7) -> Image.Image:
    img = Image.fromarray((np.clip(x01, 0, 1) * 255).astype(np.uint8))
    return img.resize((x01.shape[1] * scale, x01.shape[0] * scale), Image.NEAREST)


def mosaic(clean: np.ndarray, faulty: np.ndarray, pred: np.ndarray, y: np.ndarray, n: int = 8) -> Image.Image:
    """Two rows of chips (clean, then as the sensor delivers them); a red frame marks a wrong prediction."""
    scale, pad, border, head = 7, 10, 4, 22
    side = 20 * scale + 2 * border
    w = n * side + (n + 1) * pad
    h = head + 2 * (side + pad) + pad + head
    canvas = Image.new("RGB", (w, h), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 4), "Clean chips (ground truth)", fill=(82, 81, 78))
    d.text(
        (pad, head + side + pad + 6),
        "What the sensor delivers - frame = classifier right (green) or wrong (red)",
        fill=(82, 81, 78),
    )
    for i in range(n):
        x0 = pad + i * (side + pad)
        canvas.paste(_chip(clean[i], scale), (x0 + border, head + border))
        lx, ly = x0 + border, head + side - border - 13
        d.rectangle([lx, ly, lx + 6 * len(CLASS_NAMES[int(y[i])]) + 6, ly + 12], fill=(11, 11, 11))
        d.text((lx + 3, ly), CLASS_NAMES[int(y[i])], fill=(255, 255, 255))
        y0 = 2 * head + side + pad
        col = (27, 175, 122) if pred[i] == y[i] else (227, 73, 72)
        d.rectangle([x0, y0, x0 + side - 1, y0 + side - 1], fill=col)
        canvas.paste(_chip(faulty[i], scale), (x0 + border, y0 + border))
    return canvas


def card(title: str, value: str, detail: str, state: str) -> str:
    return (
        f'<div class="ne-card ne-{state}"><h4>{html.escape(title)}</h4>'
        f'<div class="v">{html.escape(value)}</div><div class="d">{detail}</div></div>'
    )


def bars(ratio: dict[str, float], fired: list[str]) -> str:
    """Score / threshold per component; the black tick is the threshold (ratio 1, a third of the track)."""
    rows = []
    for k in COMPONENTS:
        r = float(ratio[k])
        width = min(r, 3.0) / 3.0 * 100
        col = RED if k in fired else GREEN
        rows.append(
            f'<div class="ne-bar"><div class="lab">{LABELS[k]}</div><div class="track">'
            f'<div class="fill" style="width:{width:.1f}%;background:{col}"></div><div class="thr"></div></div>'
            f'<div class="num">{r:.2f}×</div></div>'
        )
    return (
        "<div style='font-size:13px;opacity:.8;margin-bottom:4px'>Window mean of each score, divided by its "
        "threshold (black tick). Above the tick, that component raises the alarm.</div>" + "".join(rows)
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def make_callbacks(lm: LiveMonitor):
    def sensor(fault: str, severity: str, seed: float):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(lm.y), size=WINDOW, replace=False)
        clean, y = lm.x01[idx], lm.y[idx]
        faulty = lm.corrupt(clean, fault, severity.lower(), seed=int(seed))
        r = lm.check_stream(faulty)
        clean_pred = lm.clean_out["logits"][idx].argmax(1)
        acc, acc_clean = float((r["pred"] == y).mean()), float((clean_pred == y).mean())

        def verdict(name, alarm, fired):
            why = (
                "fired: " + ", ".join(LABELS[k].split(" (")[0] for k in fired)
                if fired
                else "all scores under threshold"
            )
            return card(name, "ALARM" if alarm else "TRUST", why, "bad" if alarm else "ok")

        cards = (
            '<div class="ne-cards">'
            + card(
                "Classifier on this window",
                f"{acc:.0%}",
                f"accuracy on {WINDOW} chips (clean sensor: {acc_clean:.0%})",
                "neutral",
            )
            + verdict("Nerve lite", r["alarm_lite"], r["fired_lite"])
            + verdict("Nerve full", r["alarm_full"], r["fired_full"])
            + card(
                "Baseline: softmax confidence",
                "ALARM" if r["alarm_msp"] else "TRUST",
                "same window and false-alarm rate",
                "bad" if r["alarm_msp"] else "ok",
            )
            + "</div>"
        )
        return mosaic(clean, faulty, r["pred"], y), cards, bars(r["ratio"], r["fired_full"])

    def flips(mode_label: str, layer: str, k: float, seed: float):
        mode = {"Random bits, anywhere": "uniform", "Sign bit of an INT8 weight": "weight_sign"}.get(
            mode_label, "bias_high"
        )
        rng = np.random.default_rng(int(seed))
        try:
            fl = lm.sample_flips(rng, mode, layer, int(k))
        except ValueError as e:
            return f"<p>{html.escape(str(e))}</p>", ""
        r = lm.inject(fl)
        harm = r["changed"] > 0.01
        rows = "".join(
            f"<tr><td>{f['layer']}</td><td>{f['kind']}</td><td>{f['element']}</td><td>{f['bit']}</td>"
            f"<td>{f['old']:,}</td><td>{f['new']:,}</td></tr>"
            for f in r["flips"][:8]
        )
        more = (
            f"<p style='font-size:12px;opacity:.7'>… and {len(r['flips']) - 8} more flips</p>"
            if len(r["flips"]) > 8
            else ""
        )
        table = (
            "<table class='ne-table'><tr><th>Layer</th><th>Tensor</th><th>Element</th><th>Bit</th>"
            f"<th>Stored value before</th><th>after</th></tr>{rows}</table>{more}"
        )
        crc = r["crc_tensors"]
        cards = (
            '<div class="ne-cards">'
            + card(
                "Impact on the classifier",
                f"{r['changed']:.1%}",
                f"of {r['n']} predictions changed · accuracy {r['accuracy_clean']:.1%} → {r['accuracy_faulty']:.1%}",
                "bad" if harm else "ok",
            )
            + card(
                "Layer bounds · no reference",
                f"{r['layer_alarm_rate']:.0%}",
                f"of inferences flagged (clean model: about {FAR:.0%})",
                "bad" if r["layer_alarm_rate"] > 0.05 else "ok",
            )
            + card(
                "Feature bounds · no reference",
                f"{r['feature_alarm_rate']:.0%}",
                "of inferences flagged",
                "bad" if r["feature_alarm_rate"] > 0.05 else "ok",
            )
            + card(
                "CRC-32 scrub · reference",
                "MISMATCH" if crc else "OK",
                f"{len(crc)} tensor(s) differ from the validated copy",
                "bad" if crc else "ok",
            )
            + card(
                "Golden inputs · reference",
                "CHANGED" if r["golden_changed"] else "SAME",
                "8 stored chips, logits compared bit for bit",
                "bad" if r["golden_changed"] else "ok",
            )
            + "</div>"
        )
        return table, cards

    return sensor, flips


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def build_app(lm: LiveMonitor):
    import gradio as gr

    sensor, flips = make_callbacks(lm)
    layers = ["any"] + sorted(set(lm.layers.values()), key=lambda s: (s == "fc", s))
    with gr.Blocks(title="Nerve Edge Decision") as app:
        gr.Markdown(
            "# Nerve Edge Decision\n"
            "A runtime monitor for an INT8 Earth-observation network (plane detection on 3 m PlanetScope chips, "
            "ONNX Runtime). The network always returns a confident answer; the monitor decides whether that answer "
            "can still be trusted. Every number below is computed live on test chips the monitor was not tuned on. "
            f"Thresholds: {FAR:.0%} false alarms on clean data."
        )
        with gr.Tab("Sensor faults"):
            gr.Markdown(
                f"The sensor degrades and sends {WINDOW} consecutive chips. The monitor averages its scores over the "
                "window: a persistent fault leaves evidence on every chip."
            )
            with gr.Row():
                fault = gr.Dropdown(
                    [(FAULT_LABELS[f], f) for f in FAULTS], value="low_light", label="Sensor fault", scale=3
                )
                severity = gr.Radio(["Mild", "Strong"], value="Strong", label="Severity", scale=2)
                seed = gr.Number(value=0, precision=0, label="Window seed", scale=1)
                run = gr.Button("Send window", variant="primary", scale=1)
            image = gr.Image(label="Window", type="pil", show_label=False)
            verdict = gr.HTML()
            detail = gr.HTML()
            run.click(sensor, [fault, severity, seed], [image, verdict, detail])
            app.load(sensor, [fault, severity, seed], [image, verdict, detail])
        with gr.Tab("Bit flips"):
            gr.Markdown(
                "Flip bits in the INT8 parameter memory (weights and INT32 biases), rebuild the model and run it on "
                "512 test chips. Reference-based checks compare with the validated copy; the bounds need no reference."
            )
            with gr.Row():
                mode = gr.Radio(
                    ["Random bits, anywhere", "Sign bit of an INT8 weight", "High bit (24-31) of an INT32 bias"],
                    value="High bit (24-31) of an INT32 bias",
                    label="What flips",
                    scale=3,
                )
                layer = gr.Dropdown(layers, value="conv3", label="Layer (targeted flips)", scale=1)
                k = gr.Slider(1, 256, value=16, step=1, label="Bits (random mode)", scale=2)
                fseed = gr.Number(value=1, precision=0, label="Seed", scale=1)
                go = gr.Button("Flip", variant="primary", scale=1)
            fcards = gr.HTML()
            ftable = gr.HTML()
            go.click(flips, [mode, layer, k, fseed], [ftable, fcards])
        gr.Markdown(
            "Design, protocol and limits: [DESIGN.md](https://github.com/JolanMc/nerve-edge-decision/blob/main/DESIGN.md) · "
            "full benchmark: [results/summary.md](https://github.com/JolanMc/nerve-edge-decision/blob/main/results/summary.md)"
        )
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/edgenet_int8_monitor_full.onnx"))
    ap.add_argument("--raw", type=Path, default=Path("data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    import gradio as gr

    print("Loading PlanesNet and fitting the monitors…", flush=True)
    lm = LiveMonitor(a.model, a.raw)
    build_app(lm).launch(server_name=a.host, server_port=a.port, theme=gr.themes.Soft(primary_hue="blue"), css=CSS)


if __name__ == "__main__":
    main()
