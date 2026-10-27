"""Sim validation of the table-plane perception geometry (calibration + projection).

A single fixed RGB camera is modelled with a synthetic pinhole at the sim camera pose (and a couple
of alternative mounts).  For many random table layouts the known objects are placed at random (x, y)
inside the workspace; the exact silhouette bounding box is computed by projecting each object's 3D
shell (this is the ground-truth box a perfect detector would output), optionally perturbed by
Gaussian pixel noise to model detector localisation error; the box is fed through the SAME
projection path the app uses, and the estimated (x, y) is compared with ground truth.

This isolates the geometry gate (median < 2 cm, 90th pct < 3 cm).  Detector-in-the-loop behaviour
(real boxes, detection rate, false positives, VRAM, latency) is measured separately on rendered
images by scripts/perception_detector_eval.py (needs torch/transformers).

    PYTHONPATH=src OMP_NUM_THREADS=4 .venv/bin/python scripts/perception_sim_eval.py --arm dev_arm \
        --layouts 40 --noise-px 2.0
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

GATE_MED_CM, GATE_P90_CM = 2.0, 3.0


def silhouette_bbox(cam, obj, xy, z_table):
    px = cam.project(obj.silhouette_points(xy, z_table))
    return np.array([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])


def in_frame(cam, bbox, margin=2):
    return (bbox[0] >= margin and bbox[1] >= margin and
            bbox[2] <= cam.width - margin and bbox[3] <= cam.height - margin)


def random_layout(lib, ws, z_table, rng, min_sep=0.06):
    """Random non-overlapping (x, y) per object inside the workspace (with footprint margin)."""
    pts = {}
    for name, o in lib.items():
        for _ in range(200):
            x = rng.uniform(ws[0][0] + o.radius, ws[1][0] - o.radius)
            y = rng.uniform(ws[0][1] + o.radius, ws[1][1] - o.radius)
            if all(np.hypot(x - px, y - py) > min_sep + o.radius + lib[n].radius for n, (px, py) in pts.items()):
                pts[name] = (x, y)
                break
    return pts


def eval_mount(name, cam, lib, ws, z_table, layouts, noise_px, method, rng):
    errs, n_layout_pts, n_est, n_rejected = [], 0, 0, 0
    for _ in range(layouts):
        layout = random_layout(lib, ws, z_table, rng)
        for obj_name, (x, y) in layout.items():
            o = lib[obj_name]
            bbox = silhouette_bbox(cam, o, (x, y), z_table)
            if not in_frame(cam, bbox):
                continue  # object out of this camera's view
            n_layout_pts += 1
            if noise_px:
                bbox = bbox + rng.normal(0, noise_px, 4)
            det = project_detection(cam, obj_name, tuple(bbox), 0.9, o, z_table,
                                    workspace=ws, min_conf=0.0, method=method)
            if det is None:
                n_rejected += 1
                continue
            n_est += 1
            errs.append(100.0 * np.linalg.norm(det.xyz[:2] - np.array([x, y])))
    errs = np.array(errs) if errs else np.array([np.nan])
    return {
        "mount": name,
        "camera_center": np.round(cam.center(), 4).tolist(),
        "layouts": layouts,
        "visible_object_instances": int(n_layout_pts),
        "estimated": int(n_est),
        "rejected_out_of_workspace": int(n_rejected),
        "detection_rate": round(n_est / n_layout_pts, 4) if n_layout_pts else 0.0,
        "median_cm": round(float(np.median(errs)), 3),
        "p90_cm": round(float(np.percentile(errs, 90)), 3),
        "max_cm": round(float(np.max(errs)), 3),
        "mean_cm": round(float(np.mean(errs)), 3),
        "pass_gate": bool(np.median(errs) < GATE_MED_CM and np.percentile(errs, 90) < GATE_P90_CM),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dev_arm")
    ap.add_argument("--layouts", type=int, default=40)
    ap.add_argument("--noise-px", type=float, default=2.0)
    ap.add_argument("--method", default="refine", choices=["refine", "pushback"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "logs/perception/sim_eval.json"))
    args = ap.parse_args()

    cfg = load_config(arm=args.arm)
    z_table = cfg.get("table", {}).get("top", 0.0)
    lib = library_from_scene(cfg["scene"])
    set_support(lib, z_table)
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    ctr = (ws[0] + ws[1]) / 2
    W, H = 640, 480

    # the sim behind view + two realistic alternative mounts (high near-overhead, moderate oblique)
    sim_view = cfg.get("sim", {}).get("views", {}).get(cfg.get("sim", {}).get("default_view", "behind"))
    mounts = {
        "sim_behind": (np.array(sim_view["eye"], float), np.array(sim_view["target"], float), 60.0),
        "high_overhead_1.2m": (np.array([ctr[0], ctr[1], 1.2]), np.array([ctr[0], ctr[1], z_table]), 55.0),
        "oblique_front_45deg": (np.array([ctr[0] - 0.55, ctr[1], 0.55]), np.array([ctr[0], ctr[1], z_table]), 60.0),
    }
    results = []
    for name, (eye, target, hfov) in mounts.items():
        R, t = look_at(eye, target)
        cam = PinholeCamera(K=intrinsics_from_fov(W, H, hfov), dist=np.zeros(5), R_cw=R, t_cw=t, width=W, height=H)
        for noise in sorted({0.0, args.noise_px}):
            rng = np.random.default_rng(args.seed)
            r = eval_mount(name, cam, lib, ws, z_table, args.layouts, noise, args.method, rng)
            r["noise_px"] = noise
            r["hfov_deg"] = hfov
            results.append(r)
            print(f"{name:20s} noise={noise:>4}px  det_rate={r['detection_rate']:.2f}  "
                  f"median={r['median_cm']:.2f}cm  p90={r['p90_cm']:.2f}cm  max={r['max_cm']:.2f}cm  "
                  f"{'PASS' if r['pass_gate'] else 'FAIL'}")

    out = {"arm": args.arm, "layouts": args.layouts, "method": args.method,
           "gate": {"median_cm": GATE_MED_CM, "p90_cm": GATE_P90_CM},
           "note": "synthetic pinhole at the sim camera pose; exact silhouette boxes + pixel noise; "
                   "false-detection rate is 0 by construction (only real objects generated) - it is "
                   "measured with a real detector in perception_detector_eval.py",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
