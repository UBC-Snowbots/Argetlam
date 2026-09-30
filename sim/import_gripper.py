"""URDF -> USD for the arm + gripper (Isaac Sim 6.0.1 importer API). Headless, ~1 min.

    tsp gpu-run bash -c 'cd ~/projects/ubc-rover-arm/rover-intent && OMNI_KIT_ACCEPT_EULA=YES ~/projects/ubc-rover-arm/isaac/isaacenv6/bin/python sim/import_gripper.py > logs/import_gripper.log 2>&1'

Output: assets/rover2026/usd/rover2026_gripper/rover2026_gripper.usda (links are nested prims).
Checks a6_rotation is NOT locked and the finger joints exist; writes a report to logs/import_gripper.out.
"""
import glob
import os
import traceback
from pathlib import Path

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", default="assets/rover2026/rover2026_gripper.urdf")
ap.add_argument("--out", default="assets/rover2026/usd")
ap.add_argument("--check", nargs="*", default=["gripper_base", "finger_left", "finger_right", "tcp", "a6_rotation",
                                                "finger_left_joint", "finger_right_joint"])
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / args.urdf
OUT_DIR = ROOT / args.out
REPORT = ROOT / "logs/import_gripper.out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
out = []
res = None
try:
    cfg = URDFImporterConfig(urdf_path=str(SRC), usd_path=str(OUT_DIR), merge_fixed_joints=False,
                             fix_base=True, allow_self_collision=False)
    res = URDFImporter(cfg).import_urdf()
    out.append(f"import result: {res}")
except Exception:  # noqa: BLE001
    out.append("IMPORT_FAIL\n" + traceback.format_exc())

files = sorted(glob.glob(f"{OUT_DIR}/**/*.usd*", recursive=True))
out.append("files: " + ", ".join(os.path.relpath(f, OUT_DIR) for f in files))
path = res if isinstance(res, str) and os.path.exists(res) else (files[0] if files else None)
if path:
    st = Usd.Stage.Open(path)
    prims = {p.GetName(): p for p in st.Traverse()}
    for want in args.check:
        out.append(f"has {want}: {want in prims}")
    for jn in [c for c in args.check if "joint" in c or "rotation" in c]:
        p = prims.get(jn)
        if p:
            lo, hi = p.GetAttribute("physics:lowerLimit"), p.GetAttribute("physics:upperLimit")
            out.append(f"{jn} type={p.GetTypeName()} lower={lo.Get() if lo else None} upper={hi.Get() if hi else None}")
REPORT.write_text("\n".join(out) + "\n")
print("\n".join(out))
os._exit(0)
