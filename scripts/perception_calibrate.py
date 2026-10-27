"""Calibrate the real table camera from ChArUco images and write configs/calib/table_cam.yaml.

Two steps (see docs/TABLE_CAM.md for the physical procedure):

  INTRINSICS: ~15 photos of the board held at many angles/distances filling the frame.
      python scripts/perception_calibrate.py intrinsics --images 'calib_shots/*.jpg' \
          --squares-x 5 --squares-y 7 --square-len 0.035 --marker-len 0.026 --out-intr intr.yaml

  EXTRINSICS: ONE photo of the board lying flat on the table at a measured pose (its origin (x, y)
  and yaw in the arm base frame), reusing the intrinsics. No arm motion needed.
      python scripts/perception_calibrate.py extrinsics --image table.jpg --intr intr.yaml \
          --origin 0.40 -0.12 --yaw-deg 0 --z-table 0.0 --out configs/calib/table_cam.yaml

  ALL-IN-ONE: intrinsics images + one extrinsics image in a single call.
      python scripts/perception_calibrate.py full --images 'calib_shots/*.jpg' \
          --table-image table.jpg --origin 0.40 -0.12 --yaw-deg 0 --out configs/calib/table_cam.yaml

The board origin is the board's (0,0) corner; +X along the squares-x direction, +Y along squares-y,
lying flat on the table (z = z-table).  Mark those axes on the table, measured from the arm base.
"""
import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rover_intent.perception.calibration import (BoardSpec, calibrate_intrinsics, estimate_extrinsics,  # noqa: E402
                                                 save_calibration)


def _spec(a):
    return BoardSpec(a.squares_x, a.squares_y, a.square_len, a.marker_len, a.dictionary)


def _read(pattern):
    import cv2
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no images matched {pattern!r}")
    imgs = [cv2.imread(p) for p in paths]
    return [im for im in imgs if im is not None], paths


def _add_board_args(ap):
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-len", type=float, default=0.035)
    ap.add_argument("--marker-len", type=float, default=0.026)
    ap.add_argument("--dictionary", default="DICT_4X4_50")


def do_intrinsics(a):
    imgs, paths = _read(a.images)
    K, dist, rms, n = calibrate_intrinsics(imgs, _spec(a))
    print(f"intrinsics from {n}/{len(paths)} views, reproj rms {rms:.3f} px")
    print("K=", np.round(K, 2).tolist())
    yaml.safe_dump({"K": K.tolist(), "dist": dist.tolist(), "rms": rms}, open(a.out_intr, "w"))
    print("wrote", a.out_intr)


def _extrinsics(image_path, K, dist, a):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    cam = estimate_extrinsics(img, _spec(a), K, dist, tuple(a.origin), a.z_table, np.radians(a.yaw_deg))
    return cam


def do_extrinsics(a):
    intr = yaml.safe_load(open(a.intr))
    cam = _extrinsics(a.image, np.array(intr["K"]), np.array(intr["dist"]), a)
    print("camera centre (base) =", np.round(cam.center(), 4).tolist())
    save_calibration(a.out, cam, spec=_spec(a), z_table=a.z_table,
                     meta={"source": "real", "origin_xy": a.origin, "yaw_deg": a.yaw_deg})
    print("wrote", a.out)


def do_full(a):
    imgs, paths = _read(a.images)
    K, dist, rms, n = calibrate_intrinsics(imgs, _spec(a))
    print(f"intrinsics from {n}/{len(paths)} views, reproj rms {rms:.3f} px")
    cam = _extrinsics(a.table_image, K, dist, a)
    print("camera centre (base) =", np.round(cam.center(), 4).tolist())
    save_calibration(a.out, cam, spec=_spec(a), z_table=a.z_table,
                     meta={"source": "real", "intr_rms": rms, "origin_xy": a.origin, "yaw_deg": a.yaw_deg})
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("intrinsics"); _add_board_args(pi)
    pi.add_argument("--images", required=True); pi.add_argument("--out-intr", default="intr.yaml")
    pi.set_defaults(func=do_intrinsics)

    pe = sub.add_parser("extrinsics"); _add_board_args(pe)
    pe.add_argument("--image", required=True); pe.add_argument("--intr", required=True)
    pe.add_argument("--origin", type=float, nargs=2, required=True, metavar=("X", "Y"))
    pe.add_argument("--yaw-deg", type=float, default=0.0); pe.add_argument("--z-table", type=float, default=0.0)
    pe.add_argument("--out", default=str(ROOT / "configs/calib/table_cam.yaml"))
    pe.set_defaults(func=do_extrinsics)

    pf = sub.add_parser("full"); _add_board_args(pf)
    pf.add_argument("--images", required=True); pf.add_argument("--table-image", required=True)
    pf.add_argument("--origin", type=float, nargs=2, required=True, metavar=("X", "Y"))
    pf.add_argument("--yaw-deg", type=float, default=0.0); pf.add_argument("--z-table", type=float, default=0.0)
    pf.add_argument("--out", default=str(ROOT / "configs/calib/table_cam.yaml"))
    pf.set_defaults(func=do_full)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
