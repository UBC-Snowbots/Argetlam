"""TablePlaneLocator: lift a monocular pixel / box to a 3D point in the arm base frame.

A single RGB camera cannot measure depth, but if the object rests on a flat table at a KNOWN height
z_table (base frame), the viewing ray through a pixel meets that plane at one point - the table-plane
trick (docs/TABLE_CAM.md).  This helper is the one place monocular detectors (ArUco, open-vocab) turn
pixels into positions, so the geometry lives in exactly one spot.

Two entry points:
  * locate_pixel(uv)          - ray-plane intersection at z_table.  For a marker/centroid that lies ON
                                the table plane (e.g. an ArUco tag lying flat, or a tag on the object
                                footprint).  No object size needed.
  * locate_bbox(bbox, obj)    - the silhouette-fit estimator (projection.estimate_xy): fit the object
                                footprint (x, y) so its predicted silhouette box matches the detector
                                box, using the known object size.  For an open-vocab box around the
                                whole object.  z is pinned to z_table.

Both return a base-frame (x, y, z_table) point, or None if the ray does not meet the plane in front
of the camera.
"""
from __future__ import annotations

import numpy as np

from .camera import PinholeCamera
from .objectlib import ObjInfo
from .projection import estimate_xy


class TablePlaneLocator:
    def __init__(self, camera: PinholeCamera, z_table: float = 0.0):
        self.camera = camera
        self.z_table = float(z_table)

    def locate_pixel(self, uv) -> np.ndarray | None:
        """Pixel (u, v) -> (x, y, z_table) where its viewing ray meets the table plane (or None)."""
        return self.camera.ray_plane(uv, self.z_table)

    def locate_bbox(self, bbox, obj: ObjInfo, method: str = "refine") -> np.ndarray | None:
        """Detector box around a known object -> footprint (x, y, z_table) by silhouette fit (or None)."""
        return estimate_xy(self.camera, bbox, obj, self.z_table, method=method)
