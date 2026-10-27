"""Single-RGB-camera table-plane perception: get object positions on the table from one fixed
webcam, so the pick/place/move skills work without a depth sensor and without sim ground truth.

See docs/TABLE_CAM.md.  Default config keeps `perception.source: sim`, so nothing changes unless
camera perception is explicitly turned on.
"""
from .camera import PinholeCamera, intrinsics_from_fov, look_at
from .objectlib import ObjInfo, library_from_scene, set_support
from .projection import Detection, estimate_xy, project_detection
from .source import CameraPerception, build_from_config

__all__ = ["PinholeCamera", "intrinsics_from_fov", "look_at", "ObjInfo", "library_from_scene",
           "set_support", "Detection", "estimate_xy", "project_detection", "CameraPerception",
           "build_from_config"]
