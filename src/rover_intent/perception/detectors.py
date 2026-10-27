"""Detectors: image (+ camera) -> list[Detection] with a 3D position in the arm base frame.

One `Detector` protocol, four implementations that share the camera model (camera.py), the known
object library (objectlib.py) and the table-plane lift (locator.py):

  * GroundTruthDetector - SIM ONLY oracle.  Projects the known scene objects through the camera model
    and reports their TRUE positions (optionally + localisation noise + FOV / occlusion dropout).  The
    baseline the scorer compares everything else against, and what makes the camera path runnable with
    no detector model and no Isaac.
  * ArucoDetector - OpenCV ArUco markers.  marker id -> object name; the tag CENTRE pixel is lifted to
    the table plane (TablePlaneLocator.locate_pixel).  Works on real and sim images, no GPU, no training
    - the reliable baseline.  (Sim marker TEXTURES are a TODO; see docs/PERCEPTION.md.)
  * OpenVocabDetector - wraps the text-promptable OWLv2 / Grounding DINO box detector (detector.py) and
    lifts each box with the known object size (TablePlaneLocator.locate_bbox).  Needs the `vision`
    extra / a torch python.
  * NullDetector - the plugin slot: returns nothing.  Stand-in for an open-vocab model to be filled in
    later (NanoOWL etc.) without a torch dependency.

`detect(image_rgb, gt_positions=None)` is the common entry point.  GroundTruthDetector uses
`gt_positions` ({name: [x, y, z_bottom]}, the sim truth from the bridge) and ignores the image; the
others use the image and ignore `gt_positions`.  Confidence / workspace / best-per-object filtering is
done by the CameraPerception source, so a detector just reports what it saw.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .camera import PinholeCamera
from .locator import TablePlaneLocator
from .objectlib import ObjInfo
from .projection import Detection, _pixel_sensitivity


@runtime_checkable
class Detector(Protocol):
    """image (+ optional sim ground truth) -> detections with a 3D base-frame position."""
    name: str

    def detect(self, image_rgb, gt_positions: dict[str, list[float]] | None = None) -> list[Detection]:
        ...


def _silhouette_bbox(cam: PinholeCamera, obj: ObjInfo, xy, z_table: float) -> np.ndarray:
    px = cam.project(obj.silhouette_points(xy, z_table))
    return np.array([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])


def _in_frame(cam: PinholeCamera, bbox, margin: float = 1.0) -> bool:
    return bool(bbox[0] >= margin and bbox[1] >= margin and
                bbox[2] <= cam.width - margin and bbox[3] <= cam.height - margin)


class GroundTruthDetector:
    """SIM oracle: the known scene objects projected through the camera, reported at their true pos.

    `pos_noise_cm` adds isotropic Gaussian noise to (x, y) to model a detector's localisation error.
    `drop_out_of_frame` skips an object whose silhouette box leaves the image (models FOV misses, so the
    detection rate is < 1 when the camera cannot see everything).  `occlusion` additionally drops a
    farther object whose centroid is covered by a nearer object's box.
    """
    name = "ground_truth"

    def __init__(self, camera: PinholeCamera, lib: dict[str, ObjInfo], z_table: float,
                 pos_noise_cm: float = 0.0, drop_out_of_frame: bool = True, occlusion: bool = False,
                 confidence: float = 1.0, seed: int | None = 0):
        self.camera = camera
        self.lib = lib
        self.z_table = float(z_table)
        self.pos_noise_cm = float(pos_noise_cm)
        self.drop_out_of_frame = drop_out_of_frame
        self.occlusion = occlusion
        self.confidence = float(confidence)
        self.rng = np.random.default_rng(seed)

    def detect(self, image_rgb=None, gt_positions: dict[str, list[float]] | None = None) -> list[Detection]:
        if gt_positions is None:
            raise ValueError("GroundTruthDetector needs gt_positions (the sim ground-truth object bottoms)")
        cam = self.camera
        cand = []  # (name, xy, bbox, centroid, dist_to_cam)
        cxy = cam.center()[:2]
        for name, bottom in gt_positions.items():
            obj = self.lib.get(name)
            if obj is None:
                continue
            xy = np.array([float(bottom[0]), float(bottom[1])])
            bbox = _silhouette_bbox(cam, obj, xy, self.z_table)
            if self.drop_out_of_frame and not _in_frame(cam, bbox):
                continue
            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            cand.append((name, xy, bbox, centroid, float(np.linalg.norm(xy - cxy))))
        cand.sort(key=lambda c: c[4])  # nearest first
        out, kept_boxes = [], []
        for name, xy, bbox, centroid, _ in cand:
            if self.occlusion and any(b[0] <= centroid[0] <= b[2] and b[1] <= centroid[1] <= b[3]
                                      for b in kept_boxes):
                continue  # centroid hidden behind a nearer object's box
            kept_boxes.append(bbox)
            noisy = xy + (self.rng.normal(0, self.pos_noise_cm / 100.0, 2) if self.pos_noise_cm else 0.0)
            out.append(Detection(name=name, xyz=np.array([noisy[0], noisy[1], self.z_table]),
                                 confidence=self.confidence, px_err_cm=self.pos_noise_cm,
                                 bbox=tuple(float(b) for b in bbox), in_workspace=True,
                                 source=self.name, centroid_px=centroid))
        return out


class ArucoDetector:
    """OpenCV ArUco markers -> object positions via the table-plane trick.

    `marker_map` maps a marker id (int) to an object name.  A detected marker's four corners give a
    centre pixel, lifted to the table plane (locate_pixel).  If a marker sits at the object's footprint
    centre on the table, that IS the object's bottom-centre.  confidence is 1.0 (a decoded marker is a
    certain, id'd detection) - that reliability is the point of the ArUco baseline.
    """
    name = "aruco"

    def __init__(self, camera: PinholeCamera, marker_map: dict[int, str], z_table: float,
                 dictionary: str = "DICT_4X4_50", marker_length: float = 0.03):
        self.locator = TablePlaneLocator(camera, z_table)
        self.camera = camera
        self.z_table = float(z_table)
        self.marker_map = {int(k): v for k, v in marker_map.items()}
        self.dictionary = dictionary
        self.marker_length = float(marker_length)

    def detect(self, image_rgb, gt_positions=None) -> list[Detection]:
        import cv2
        img = np.asarray(image_rgb)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        aru = cv2.aruco
        d = aru.getPredefinedDictionary(getattr(aru, self.dictionary))
        corners, ids, _ = aru.ArucoDetector(d, aru.DetectorParameters()).detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for quad, mid in zip(corners, ids.reshape(-1)):
            name = self.marker_map.get(int(mid))
            if name is None:
                continue  # an unknown tag id: ignore, never a phantom object
            pts = quad.reshape(-1, 2)
            centre = tuple(pts.mean(axis=0))
            xyz = self.locator.locate_pixel(centre)
            if xyz is None:
                continue  # ray does not meet the table in front of the camera
            bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max()))
            out.append(Detection(name=name, xyz=np.asarray(xyz, float), confidence=1.0,
                                 px_err_cm=float("nan"), bbox=bbox, in_workspace=True,
                                 source=self.name, centroid_px=(float(centre[0]), float(centre[1]))))
        return out


class OpenVocabDetector:
    """Text-promptable OWLv2 / Grounding DINO box detector (detector.py) + table-plane box lift.

    Needs the `vision` extra / a torch python; lazy-loaded.  `prompts` maps a prompt string (the object
    description) to an object name.  Each box is lifted with the known object size (locate_bbox).
    """
    def __init__(self, camera: PinholeCamera, lib: dict[str, ObjInfo], z_table: float,
                 backend: str = "owlv2", model: str | None = None, device: str = "cuda",
                 dtype: str = "float32", method: str = "refine", loc_noise_px: float = 2.0,
                 prompts: dict[str, str] | None = None):
        from .detector import Detector as BoxDetector  # lazy: avoids importing the torch wrapper path
        self.name = backend
        self.camera = camera
        self.lib = lib
        self.z_table = float(z_table)
        self.locator = TablePlaneLocator(camera, z_table)
        self.method = method
        self.loc_noise_px = float(loc_noise_px)
        self.prompts = prompts or {(o.desc or n): n for n, o in lib.items()}
        self._box = BoxDetector(backend=backend, model=model, device=device, dtype=dtype)

    def detect(self, image_rgb, gt_positions=None) -> list[Detection]:
        raw = self._box.detect(image_rgb, list(self.prompts), min_conf=0.0)
        out = []
        for r in raw:
            name = self.prompts.get(r.label)
            obj = self.lib.get(name) if name else None
            if obj is None:
                continue
            xyz = self.locator.locate_bbox(r.bbox, obj, method=self.method)
            if xyz is None:
                continue
            err = _pixel_sensitivity(self.camera, r.bbox, obj, self.z_table, self.loc_noise_px, self.method)
            out.append(Detection(name=name, xyz=np.asarray(xyz, float), confidence=float(r.score),
                                 px_err_cm=err, bbox=tuple(float(b) for b in r.bbox), in_workspace=True,
                                 source=self.name,
                                 centroid_px=((r.bbox[0] + r.bbox[2]) / 2.0, (r.bbox[1] + r.bbox[3]) / 2.0)))
        return out


class NullDetector:
    """Plugin slot: a no-op open-vocab stand-in (NanoOWL / Grounding DINO to be dropped in later)."""
    name = "null"

    def detect(self, image_rgb, gt_positions=None) -> list[Detection]:
        return []


def build_detector(backend: str, camera: PinholeCamera, lib: dict[str, ObjInfo], z_table: float,
                   cfg: dict | None = None):
    """Construct the detector named by `perception.detector.backend`.

    ground_truth | aruco need only numpy / opencv (the control-host .venv); owlv2 | gdino need torch.
    `cfg` is the `perception.detector` block (for aruco markers, open-vocab model/device, etc.).
    """
    cfg = cfg or {}
    if backend == "ground_truth":
        g = cfg.get("ground_truth", {})
        return GroundTruthDetector(camera, lib, z_table, pos_noise_cm=g.get("pos_noise_cm", 0.0),
                                   drop_out_of_frame=g.get("drop_out_of_frame", True),
                                   occlusion=g.get("occlusion", False), seed=g.get("seed", 0))
    if backend == "aruco":
        a = cfg.get("aruco", {})
        return ArucoDetector(camera, a.get("markers", {}), z_table,
                             dictionary=a.get("dictionary", "DICT_4X4_50"),
                             marker_length=a.get("marker_length", 0.03))
    if backend in ("owlv2", "gdino"):
        return OpenVocabDetector(camera, lib, z_table, backend=backend, model=cfg.get("model"),
                                 device=cfg.get("device", "cuda"), dtype=cfg.get("dtype", "float32"))
    if backend in ("null", "none"):
        return NullDetector()
    raise ValueError(f"unknown perception detector backend {backend!r}")
