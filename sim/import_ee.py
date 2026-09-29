import glob
import os
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd  # noqa: E402

HOME = os.path.expanduser("~")
SRC = f"{HOME}/projects/ubc-rover-arm/isaac/rover2026_import/rover2026_ee.urdf"
OUT_DIR = f"{HOME}/projects/ubc-rover-arm/isaac/rover2026_import/ee_usd"
os.makedirs(OUT_DIR, exist_ok=True)
out = []
try:
    cfg = URDFImporterConfig(urdf_path=SRC, usd_path=OUT_DIR, merge_fixed_joints=False,
                             fix_base=True, allow_self_collision=False)
    res = URDFImporter(cfg).import_urdf()
    out.append(f"import result: {res}")
except Exception:  # noqa: BLE001
    out.append("IMPORT_FAIL\n" + traceback.format_exc())
    res = None

files = sorted(glob.glob(f"{OUT_DIR}/**/*.usd*", recursive=True))
out.append("files: " + ", ".join(os.path.relpath(f, OUT_DIR) for f in files))
path = res if isinstance(res, str) and os.path.exists(res) else (files[0] if files else None)
if path:
    st = Usd.Stage.Open(path)
    names = {p.GetName(): p for p in st.Traverse()}
    for want in ("a6_EE_holder", "ee_presser", "ee_tip", "a6_rotation"):
        out.append(f"has {want}: {want in names}")
    p = names.get("a6_rotation")
    if p:
        lo, hi = p.GetAttribute("physics:lowerLimit"), p.GetAttribute("physics:upperLimit")
        out.append(f"a6_rotation type={p.GetTypeName()} lower={lo.Get() if lo else None} upper={hi.Get() if hi else None}")
open("/tmp/import_ee.out", "w").write("\n".join(out))
os._exit(0)
