"""Camera perception source: one RGB frame -> object positions in the arm base frame.

Ties together the calibrated camera (camera.py), the known-object library (objectlib.py), the
open-vocab detector (detector.py) and the table-plane projection (projection.py).  The app uses
`as_scene_updates()` to fill the Scene's object positions when `perception.source: camera`.

The detector step is optional at construction time so the geometry (camera + projection) can be
unit-tested and validated with synthetic boxes, with no torch/transformers import.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .objectlib import ObjInfo, library_from_scene, set_support
from .projection import Detection, project_detection


class CameraPerception:
    def __init__(self, cam: PinholeCamera, lib: dict[str, ObjInfo], z_table: float,
                 workspace: tuple[np.ndarray, np.ndarray] | None = None, min_conf: float = 0.25,
                 method: str = "refine", loc_noise_px: float = 2.0,
                 prompts: dict[str, str] | None = None, detector=None):
        self.cam = cam
        self.lib = lib
        self.z_table = z_table
        self.workspace = workspace
        self.min_conf = min_conf
        self.method = method
        self.loc_noise_px = loc_noise_px
        # prompt text -> object name.  Default: the object's description if present, else its name.
        self.prompts = prompts or {(o.desc or name): name for name, o in lib.items()}
        self.detector = detector

    def prompt_list(self) -> list[str]:
        return list(self.prompts)

    def project_boxes(self, boxes: list[tuple[str, tuple, float]]) -> list[Detection]:
        """`boxes` = [(object_name, (u0,v0,u1,v1), score)] -> accepted Detections (best score per object)."""
        best: dict[str, Detection] = {}
        for name, bbox, score in boxes:
            obj = self.lib.get(name)
            if obj is None:
                continue
            det = project_detection(self.cam, name, bbox, score, obj, self.z_table,
                                    workspace=self.workspace, min_conf=self.min_conf,
                                    loc_noise_px=self.loc_noise_px, method=self.method)
            if det is not None and (name not in best or det.confidence > best[name].confidence):
                best[name] = det
        return list(best.values())

    def detect(self, image_rgb) -> list[Detection]:
        """Run the detector on an image and project every box. Requires a detector was supplied."""
        if self.detector is None:
            raise RuntimeError("CameraPerception has no detector; pass one or use project_boxes()")
        raw = self.detector.detect(image_rgb, self.prompt_list(), min_conf=self.min_conf)
        boxes = [(self.prompts[r.label], r.bbox, r.score) for r in raw if r.label in self.prompts]
        return self.project_boxes(boxes)

    def as_scene_updates(self, detections: list[Detection]) -> dict[str, np.ndarray]:
        """{name: bottom-centre xyz} for Scene.update_from_bottom (z pinned to the table)."""
        return {d.name: d.xyz for d in detections}


def build_from_config(cfg: dict, with_detector: bool = False) -> CameraPerception:
    """Construct a CameraPerception from the merged config (reads the saved calibration YAML)."""
    from .calibration import load_calibration
    root = Path(__file__).resolve().parents[3]
    p = cfg.get("perception", {})
    calib_path = p.get("calib", "configs/calib/table_cam.yaml")
    cam, z_table = load_calibration(root / calib_path)
    lib = library_from_scene(cfg["scene"])
    set_support(lib, cfg.get("table", {}).get("top", 0.0))
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    det = None
    if with_detector:
        from .detector import Detector
        d = p.get("detector", {})
        det = Detector(backend=d.get("backend", "owlv2"), model=d.get("model"),
                       device=d.get("device", "cuda"), dtype=d.get("dtype", "float16"))
    return CameraPerception(cam, lib, z_table, workspace=ws, min_conf=p.get("min_conf", 0.25),
                            method=p.get("method", "refine"), loc_noise_px=p.get("loc_noise_px", 2.0),
                            detector=det)
