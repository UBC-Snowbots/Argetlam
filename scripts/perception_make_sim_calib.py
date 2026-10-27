"""Write a camera calibration YAML for the SIM table camera, from the Isaac view geometry.

This stands in for a real ChArUco calibration so the camera perception path and the sim validation
can run end to end without a physical camera.  The real setup replaces this file with the output of
scripts/perception_calibrate.py (see docs/TABLE_CAM.md).

    PYTHONPATH=src .venv/bin/python scripts/perception_make_sim_calib.py --arm dev_arm --hfov 60
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rover_intent.config import load_config  # noqa: E402
from rover_intent.perception.camera import PinholeCamera, intrinsics_from_fov, look_at  # noqa: E402
from rover_intent.perception.calibration import save_calibration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dev_arm")
    ap.add_argument("--view", default=None, help="sim view name (default: profile default_view)")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=str(ROOT / "configs/calib/table_cam.yaml"))
    args = ap.parse_args()

    cfg = load_config(arm=args.arm)
    sim = cfg.get("sim", {})
    view = args.view or sim.get("default_view", "behind")
    v = sim.get("views", {}).get(view)
    if v is None:
        raise SystemExit(f"view {view!r} not in profile sim.views ({list(sim.get('views', {}))})")
    eye, target = np.array(v["eye"], float), np.array(v["target"], float)
    K = intrinsics_from_fov(args.width, args.height, args.hfov)
    R, t = look_at(eye, target)
    cam = PinholeCamera(K=K, dist=np.zeros(5), R_cw=R, t_cw=t, width=args.width, height=args.height)
    z_table = cfg.get("table", {}).get("top", 0.0)
    save_calibration(args.out, cam, z_table=z_table,
                     meta={"source": "sim", "arm": args.arm, "view": view, "hfov_deg": args.hfov,
                           "eye": eye.tolist(), "target": target.tolist(),
                           "note": "synthetic sim calibration; replace with a real ChArUco calibration"})
    print(f"wrote {args.out}")
    print(f"  view={view} eye={eye.tolist()} target={target.tolist()} hfov={args.hfov} z_table={z_table}")
    print(f"  camera centre (base) = {np.round(cam.center(), 4).tolist()}")


if __name__ == "__main__":
    main()
