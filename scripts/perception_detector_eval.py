"""Detector-in-the-loop sim validation: render the known objects, run an open-vocab detector,
project its boxes through the table-plane path, and measure VRAM / latency / detection rate /
false positives / end-to-end (x, y) error vs ground truth.

Run on a python with torch + transformers + cv2 (the Isaac env), NOT the lightweight control-host
.venv.  Short interactive GPU inference; it only loads one small detector at a time and unloads it.

    PYTHONPATH=src OMP_NUM_THREADS=4 \
      ~/projects/ubc-rover-arm/isaac/isaacenv6/bin/python scripts/perception_detector_eval.py \
      --arm dev_arm --layouts 30 --backends owlv2 gdino --det-conf 0.1
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rover_intent.config import load_config  # noqa: E402
from rover_intent.perception.camera import PinholeCamera, intrinsics_from_fov, look_at  # noqa: E402
from rover_intent.perception.objectlib import library_from_scene, set_support  # noqa: E402
from rover_intent.perception.projection import project_detection  # noqa: E402
from rover_intent.perception.synth import render_scene_view  # noqa: E402
from rover_intent.perception.detector import Detector  # noqa: E402

GATE_MED_CM, GATE_P90_CM = 2.0, 3.0


def silhouette_bbox(cam, obj, xy, z_table):
    px = cam.project(obj.silhouette_points(xy, z_table))
    return np.array([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def random_layout(lib, ws, rng, min_sep=0.06):
    pts = {}
    for name, o in lib.items():
        for _ in range(200):
            x = rng.uniform(ws[0][0] + o.radius, ws[1][0] - o.radius)
            y = rng.uniform(ws[0][1] + o.radius, ws[1][1] - o.radius)
            if all(np.hypot(x - px, y - py) > min_sep + o.radius + lib[n].radius for n, (px, py) in pts.items()):
                pts[name] = (x, y)
                break
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dev_arm")
    ap.add_argument("--layouts", type=int, default=30)
    ap.add_argument("--backends", nargs="+", default=["owlv2", "gdino"])
    ap.add_argument("--det-conf", type=float, default=0.1)
    ap.add_argument("--proj-conf", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-first", action="store_true", help="save the first rendered frame to logs/perception/")
    ap.add_argument("--out", default=str(ROOT / "logs/perception/detector_eval.json"))
    args = ap.parse_args()

    import torch

    cfg = load_config(arm=args.arm)
    z_table = cfg.get("table", {}).get("top", 0.0)
    lib = library_from_scene(cfg["scene"])
    set_support(lib, z_table)
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    W, H = 640, 480
    v = cfg["sim"]["views"][cfg["sim"].get("default_view", "behind")]
    R, t = look_at(np.array(v["eye"], float), np.array(v["target"], float))
    cam = PinholeCamera(K=intrinsics_from_fov(W, H, 60.0), dist=np.zeros(5), R_cw=R, t_cw=t, width=W, height=H)
    prompts = {o.desc: name for name, o in lib.items()}           # "short red cup" -> "cup"
    prompt_list = list(prompts)

    # pre-generate layouts + render + ground-truth boxes (CPU)
    rng = np.random.default_rng(args.seed)
    frames = []
    for i in range(args.layouts):
        layout = random_layout(lib, ws, rng)
        img = render_scene_view(cam, lib, layout, z_table)  # BGR uint8
        gt = {n: silhouette_bbox(cam, lib[n], xy, z_table) for n, xy in layout.items()}
        frames.append((img, layout, gt))
    if args.save_first:
        import cv2
        cv2.imwrite(str(ROOT / "logs/perception/detector_eval_frame0.png"), frames[0][0])

    report = {"arm": args.arm, "layouts": args.layouts, "det_conf": args.det_conf,
              "camera_center": np.round(cam.center(), 4).tolist(),
              "gate": {"median_cm": GATE_MED_CM, "p90_cm": GATE_P90_CM},
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "backends": {}}

    for backend in args.backends:
        det = Detector(backend=backend, device="cuda", dtype="float32")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_load = time.time()
        det.load()
        load_s = time.time() - t_load
        # warmup
        det.detect(frames[0][0][:, :, ::-1], prompt_list, min_conf=args.det_conf)
        lat, errs, n_tp, n_obj, n_fp = [], [], 0, 0, 0
        for img, layout, gt in frames:
            rgb = img[:, :, ::-1]
            t0 = time.time()
            raw = det.detect(rgb, prompt_list, min_conf=args.det_conf)
            lat.append(time.time() - t0)
            # best detection per object name
            by_name = {}
            for r in raw:
                if r.label not in prompts:
                    n_fp += 1
                    continue
                nm = prompts[r.label]
                if nm not in by_name or r.score > by_name[nm].score:
                    by_name[nm] = r
            for nm, (x, y) in layout.items():
                n_obj += 1
                r = by_name.get(nm)
                if r is None or iou(r.bbox, gt[nm]) < 0.2:
                    continue  # miss (or a box far from the true object)
                n_tp += 1
                d = project_detection(cam, nm, r.bbox, r.score, lib[nm], z_table,
                                      workspace=ws, min_conf=args.proj_conf, method="refine")
                if d is not None:
                    errs.append(100.0 * np.linalg.norm(d.xyz[:2] - np.array([x, y])))
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        errs = np.array(errs) if errs else np.array([np.nan])
        rb = {
            "model_id": det.model_id,
            "vram_peak_mb": round(peak_mb, 1),
            "load_s": round(load_s, 2),
            "latency_ms_median": round(1e3 * float(np.median(lat)), 1),
            "latency_ms_p90": round(1e3 * float(np.percentile(lat, 90)), 1),
            "detection_rate": round(n_tp / n_obj, 4) if n_obj else 0.0,
            "false_positives": int(n_fp),
            "objects_total": int(n_obj),
            "true_positives": int(n_tp),
            "endto_end_median_cm": round(float(np.median(errs)), 3),
            "endto_end_p90_cm": round(float(np.percentile(errs, 90)), 3),
            "endto_end_max_cm": round(float(np.max(errs)), 3),
            "pass_gate": bool(np.median(errs) < GATE_MED_CM and np.percentile(errs, 90) < GATE_P90_CM),
        }
        report["backends"][backend] = rb
        print(f"[{backend}] {det.model_id}")
        print(f"   VRAM peak {rb['vram_peak_mb']:.0f} MB  load {rb['load_s']}s  "
              f"latency {rb['latency_ms_median']}/{rb['latency_ms_p90']} ms (med/p90)")
        print(f"   det_rate {rb['detection_rate']:.2f}  FP {rb['false_positives']}  "
              f"end-to-end median {rb['endto_end_median_cm']}cm p90 {rb['endto_end_p90_cm']}cm  "
              f"{'PASS' if rb['pass_gate'] else 'FAIL'}")
        det.unload()
        del det
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
