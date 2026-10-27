"""Camera perception source: one camera frame (+ optional sim ground truth) -> object positions in
the arm base frame, with staleness / confidence / smoothing tracking and provenance for the HUD.

Ties together the camera model (camera.py), the known-object library (objectlib.py), one of the
pluggable detectors (detectors.py: ground_truth | aruco | owlv2 | gdino | null) and the table-plane
lift (locator.py, inside the detectors).  The app calls `ingest()` when a frame arrives and
`tracker` for what is currently seen and where each object came from.

`perception.source: sim | camera` is the top-level switch (default sim = unchanged behaviour;
`ground_truth` is accepted as an alias of sim).  Within `camera`, `perception.detector.backend`
chooses HOW positions are found: `ground_truth` is the in-sim oracle (needs no model, no Isaac),
`aruco` is the reliable marker baseline, `owlv2`/`gdino` are the open-vocab models (need torch).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .camera import PinholeCamera, camera_from_config
from .detectors import build_detector
from .objectlib import ObjInfo, library_from_scene, set_support
from .projection import Detection


class PerceptionTracker:
    """Per-object track with staleness, confidence gate and exponential smoothing.

    A detection updates a track only if its confidence is >= `min_conf`; the stored position is an EMA
    of the raw positions (alpha = weight of the newest frame).  An object is "seen" only while its last
    accepted detection is within `stale_s`.  The app reads `updates()` for fresh scene positions,
    `seen()` to gate a skill, and `provenance()` for the HUD.
    """

    def __init__(self, min_conf: float = 0.25, stale_s: float = 1.0, alpha: float = 0.5):
        self.min_conf = float(min_conf)
        self.stale_s = float(stale_s)
        self.alpha = float(alpha)
        self.tracks: dict[str, dict] = {}  # name -> {xyz, t, confidence, source}

    def update(self, detections: list[Detection], now: float | None = None) -> None:
        now = time.time() if now is None else now
        for d in detections:
            if d.confidence < self.min_conf:
                continue
            tr = self.tracks.get(d.name)
            xyz = np.asarray(d.xyz, float)
            if tr is None or (now - tr["t"]) > self.stale_s:
                sm = xyz  # no fresh history: snap
            else:
                sm = self.alpha * xyz + (1 - self.alpha) * tr["xyz"]
            self.tracks[d.name] = {"xyz": sm, "t": now, "confidence": float(d.confidence),
                                   "source": d.source}

    def seen(self, name: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        tr = self.tracks.get(name)
        return tr is not None and (now - tr["t"]) <= self.stale_s

    def updates(self, now: float | None = None) -> dict[str, np.ndarray]:
        """{name: bottom-centre xyz} for every object currently seen (fresh + confident)."""
        now = time.time() if now is None else now
        return {n: tr["xyz"] for n, tr in self.tracks.items() if (now - tr["t"]) <= self.stale_s}

    def provenance(self, now: float | None = None) -> dict[str, dict]:
        """{name: {source, age_s, confidence, fresh}} for the HUD / status."""
        now = time.time() if now is None else now
        return {n: {"source": tr["source"], "age_s": round(now - tr["t"], 3),
                    "confidence": round(tr["confidence"], 3), "fresh": (now - tr["t"]) <= self.stale_s}
                for n, tr in self.tracks.items()}


class CameraPerception:
    def __init__(self, cam: PinholeCamera, lib: dict[str, ObjInfo], z_table: float, detector,
                 workspace: tuple[np.ndarray, np.ndarray] | None = None, min_conf: float = 0.25,
                 stale_s: float = 1.0, alpha: float = 0.5):
        self.cam = cam
        self.lib = lib
        self.z_table = z_table
        self.detector = detector
        self.workspace = workspace
        self.min_conf = min_conf
        self.tracker = PerceptionTracker(min_conf=min_conf, stale_s=stale_s, alpha=alpha)

    def _accept(self, det: Detection) -> bool:
        if det.confidence < self.min_conf:
            return False
        if self.workspace is not None:
            lo, hi = self.workspace
            if not (lo[0] - 1e-6 <= det.xyz[0] <= hi[0] + 1e-6 and lo[1] - 1e-6 <= det.xyz[1] <= hi[1] + 1e-6):
                return False  # a point outside the reachable box is almost surely a false positive
        return True

    def detect(self, image_rgb, gt_positions: dict[str, list[float]] | None = None) -> list[Detection]:
        """Run the detector, keep accepted detections, best confidence per object."""
        raw = self.detector.detect(image_rgb, gt_positions=gt_positions)
        best: dict[str, Detection] = {}
        for d in raw:
            if not self._accept(d):
                continue
            if d.name not in best or d.confidence > best[d.name].confidence:
                best[d.name] = d
        return list(best.values())

    def ingest(self, image_rgb, gt_positions=None, now: float | None = None) -> dict[str, np.ndarray]:
        """Detect + update the tracker; return the fresh scene updates {name: bottom-centre xyz}."""
        self.tracker.update(self.detect(image_rgb, gt_positions=gt_positions), now)
        return self.tracker.updates(now)


def build_from_config(cfg: dict, with_detector: bool = True) -> CameraPerception:
    """Construct a CameraPerception from the merged config.

    Camera: prefer the arm-profile `camera:` block (pose + intrinsics, so the sim camera matches the
    real mount); fall back to a saved ChArUco calibration YAML (perception.calib) if there is no block.
    Detector: `perception.detector.backend` (default owlv2 for back-compat; ground_truth / aruco need
    no torch).  `with_detector=False` skips building a torch backend (geometry-only construction).
    """
    root = Path(__file__).resolve().parents[3]
    p = cfg.get("perception", {})
    cam_cfg = cfg.get("camera")
    table_top = cfg.get("table", {}).get("top", 0.0)
    if cam_cfg:
        cam, z_table = camera_from_config(cam_cfg, z_table_default=table_top)
    else:
        from .calibration import load_calibration
        cam, z_table = load_calibration(root / p.get("calib", "configs/calib/table_cam.yaml"))
    lib = library_from_scene(cfg["scene"])
    set_support(lib, table_top)
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    dcfg = p.get("detector", {})
    backend = dcfg.get("backend", "owlv2")
    if backend in ("owlv2", "gdino") and not with_detector:
        detector = None
    else:
        detector = build_detector(backend, cam, lib, z_table, dcfg)
    return CameraPerception(cam, lib, z_table, detector, workspace=ws,
                            min_conf=p.get("min_conf", 0.25), stale_s=p.get("stale_s", 1.0),
                            alpha=p.get("smoothing", 0.5))
