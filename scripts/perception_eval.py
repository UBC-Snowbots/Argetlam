"""Automatic sim scorer for the camera perception path: camera-derived object positions vs the sim
GROUND TRUTH, so the whole pipeline is graded in sim before it ever touches the real arm.

It scores the SAME CameraPerception the app uses (perception.source: camera), built from the merged
config (camera pose + intrinsics from the arm profile `camera:` block).  Pick the detector with
`--backend` (default: the config's): ground_truth (sim oracle), aruco (markers), owlv2 / gdino (need
a torch python).  Metrics vs ground truth: position error per object (median / p90 / max, cm),
detection rate, and false positives.  Gate: median < 2 cm, p90 < 3 cm.

Runs HEADLESS, no Isaac, two frame sources:
  * --frames <file.jsonl|dir>   recorded frames: one CameraFrame-ish JSON per line (or per .json file),
                                with `objects` = sim ground-truth {name: [x, y, z_bottom]} and, for the
                                image detectors, `jpeg_b64`. The Isaac bridge writes exactly this (the
                                ground-truth objects ride in the same packet), so a recorded sim session
                                is scored directly.
  * --synth N                   generate N random in-workspace layouts in memory (default when no
                                --frames). ground_truth needs no image; aruco renders synthetic markers;
                                owlv2/gdino render flat-shaded objects (perception/synth.py).

    PYTHONPATH=src OMP_NUM_THREADS=4 .venv/bin/python scripts/perception_eval.py --arm dev_arm \
        --backend aruco --synth 40
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rover_intent.config import load_config  # noqa: E402
from rover_intent.perception.source import build_from_config  # noqa: E402

GATE_MED_CM, GATE_P90_CM = 2.0, 3.0
FP_GATE_CM = 10.0  # a "detection" of object X more than this from X's truth is a false positive, not a hit


def random_layouts(lib, ws, z_table, n, seed, min_sep=0.06):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pts = {}
        for name, o in lib.items():
            for _ in range(200):
                x = rng.uniform(ws[0][0] + o.radius, ws[1][0] - o.radius)
                y = rng.uniform(ws[0][1] + o.radius, ws[1][1] - o.radius)
                if all(np.hypot(x - px, y - py) > min_sep + o.radius + lib[m].radius
                       for m, (px, py) in pts.items()):
                    pts[name] = (x, y)
                    break
        out.append(pts)
    return out


def synth_frames(perc, cfg, layouts, backend, z_table):
    """layout {name:(x,y)} -> (image_rgb_or_None, gt_positions). Renders only what the detector needs."""
    from rover_intent.perception import synth
    frames = []
    for layout in layouts:
        gt = {n: [float(x), float(y), z_table] for n, (x, y) in layout.items()}
        img = None
        if backend == "aruco":
            a = cfg["perception"]["detector"].get("aruco", {})
            img = synth.render_aruco_scene(perc.cam, layout, a.get("markers", {}), z_table,
                                           marker_len=a.get("marker_length", 0.04),
                                           dictionary=a.get("dictionary", "DICT_4X4_50"))
        elif backend in ("owlv2", "gdino"):
            img = synth.render_scene_view(perc.cam, perc.lib, layout, z_table)
        frames.append((img, gt))
    return frames


def load_recorded(path):
    """A .jsonl of CameraFrame-ish dicts, or a dir of .json frames -> [(image_rgb_or_None, gt)]."""
    from PIL import Image
    p = Path(path)
    records = []
    if p.is_dir():
        for f in sorted(p.glob("*.json")):
            records.append(json.loads(f.read_text()))
    else:
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    frames = []
    for r in records:
        img = None
        if r.get("jpeg_b64"):
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(r["jpeg_b64"]))).convert("RGB"))
        gt = {n: list(map(float, v)) for n, v in (r.get("objects") or {}).items()}
        frames.append((img, gt))
    return frames


def score(perc, frames, backend):
    errs, n_gt, n_tp, n_fp = [], 0, 0, 0
    per_obj = {}
    for img, gt in frames:
        dets = perc.detect(img, gt_positions=gt)
        seen = set()
        for d in dets:
            truth = gt.get(d.name)
            err = 100.0 * np.linalg.norm(d.xyz[:2] - np.array(truth[:2])) if truth is not None else None
            if truth is not None and err <= FP_GATE_CM:
                n_tp += 1
                errs.append(err)
                per_obj.setdefault(d.name, []).append(err)
                seen.add(d.name)
            else:
                n_fp += 1  # a name not in this frame, or a hit too far from its truth = phantom
        n_gt += len(gt)
    errs = np.array(errs) if errs else np.array([np.nan])
    return {
        "backend": backend,
        "frames": len(frames),
        "gt_instances": int(n_gt),
        "true_positives": int(n_tp),
        "false_positives": int(n_fp),
        "detection_rate": round(n_tp / n_gt, 4) if n_gt else 0.0,
        "false_positive_rate": round(n_fp / max(n_tp + n_fp, 1), 4),
        "median_cm": round(float(np.median(errs)), 3),
        "p90_cm": round(float(np.percentile(errs, 90)), 3),
        "max_cm": round(float(np.max(errs)), 3),
        "mean_cm": round(float(np.mean(errs)), 3),
        "per_object_median_cm": {k: round(float(np.median(v)), 3) for k, v in per_obj.items()},
        "pass_gate": bool(np.median(errs) < GATE_MED_CM and np.percentile(errs, 90) < GATE_P90_CM),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dev_arm")
    ap.add_argument("--backend", default=None, help="override perception.detector.backend")
    ap.add_argument("--frames", default=None, help="recorded frames: .jsonl file or dir of .json")
    ap.add_argument("--synth", type=int, default=40, help="synthetic layouts if no --frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "logs/perception/eval.json"))
    args = ap.parse_args()

    cfg = load_config(arm=args.arm)
    cfg.setdefault("perception", {})["source"] = "camera"
    if args.backend:
        cfg["perception"].setdefault("detector", {})["backend"] = args.backend
    backend = cfg["perception"].get("detector", {}).get("backend", "aruco")
    perc = build_from_config(cfg, with_detector=True)  # ground_truth/aruco need no torch; owlv2/gdino do
    z_table = perc.z_table

    if args.frames:
        frames = load_recorded(args.frames)
        src = f"recorded:{args.frames}"
    else:
        ws = perc.workspace
        layouts = random_layouts(perc.lib, ws, z_table, args.synth, args.seed)
        frames = synth_frames(perc, cfg, layouts, backend, z_table)
        src = f"synth:{args.synth}"

    r = score(perc, frames, backend)
    r["source"] = src
    r["arm"] = args.arm
    r["gate"] = {"median_cm": GATE_MED_CM, "p90_cm": GATE_P90_CM}
    r["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2))
    print(f"{backend:12s} {src:14s} det_rate={r['detection_rate']:.2f} fp={r['false_positives']} "
          f"median={r['median_cm']:.2f}cm p90={r['p90_cm']:.2f}cm max={r['max_cm']:.2f}cm "
          f"{'PASS' if r['pass_gate'] else 'FAIL'}")
    print("per-object median cm:", r["per_object_median_cm"])
    print("wrote", args.out)
    return 0 if r["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
