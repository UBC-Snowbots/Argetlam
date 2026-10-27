"""Camera perception: object positions in the arm base frame from a camera, so the pick/place/move
skills can run from CAMERA-derived positions instead of sim ground truth - and be scored against it.

Pipeline: a calibrated pinhole camera (camera.py; pose + intrinsics from the arm profile `camera:`
block, or a ChArUco calibration) + a detector (detectors.py) + the table-plane lift (locator.py) ->
Detections with a 3D base-frame position, tracked with staleness / confidence / smoothing (source.py).

Detectors: GroundTruthDetector (sim oracle baseline), ArucoDetector (reliable marker baseline),
OpenVocabDetector (OWLv2 / Grounding DINO, needs torch), NullDetector (plugin slot).

Default config keeps `perception.source: sim`, so nothing changes unless camera perception is turned
on.  See docs/PERCEPTION.md (architecture, config, sim test, matching the real camera) and
docs/TABLE_CAM.md (the fixed-table-camera variant and the ChArUco calibration workflow).
"""
from .camera import PinholeCamera, camera_from_config, intrinsics_from_fov, look_at, quat_to_R
from .detectors import (ArucoDetector, Detector, GroundTruthDetector, NullDetector, OpenVocabDetector,
                        build_detector)
from .locator import TablePlaneLocator
from .objectlib import ObjInfo, library_from_scene, set_support
from .projection import Detection, estimate_xy, project_detection
from .source import CameraPerception, PerceptionTracker, build_from_config

__all__ = ["PinholeCamera", "camera_from_config", "intrinsics_from_fov", "look_at", "quat_to_R",
           "ObjInfo", "library_from_scene", "set_support", "Detection", "estimate_xy",
           "project_detection", "TablePlaneLocator", "Detector", "GroundTruthDetector",
           "ArucoDetector", "OpenVocabDetector", "NullDetector", "build_detector",
           "CameraPerception", "PerceptionTracker", "build_from_config"]
