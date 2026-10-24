"""bbox -> table-plane contact point -> object position in the arm base frame.

The table-plane trick: every object rests on a flat table at a known height z_table in the base
frame, and every known object has a known footprint radius and height.  A single RGB camera cannot
measure depth, but with those knowns a detector box is enough to place the object.

Two estimators (both documented; `refine` is the default):

* "pushback" (simple, fast, no iteration): the bottom-centre of the box is ~the image of the
  object's near-bottom rim, which lies ON the table plane.  Back-project it to z=z_table and push
  it away from the camera by the footprint radius to recover the centre.  BIAS: the raw
  back-projection sits ~1 radius in front of the true centre (toward the camera); the +radius
  push-back corrects the first-order part.  It is good for an oblique mount but DEGRADES for a
  near-top-down mount, where the push-back direction (camera azimuth) is ill-conditioned and the
  correction adds error instead of removing it.

* "refine" (default): fit the footprint centre (x, y) so the object's predicted silhouette box
  (project the known cylinder/box shell, take min/max) matches the observed box, by a few steps of
  Gauss-Newton seeded from the pushback estimate.  This uses the full known geometry, has no
  azimuth singularity, and is accurate for any view.  Pure numpy (no scipy), ~12 iters, 2 unknowns.

z is pinned to the table (known); only (x, y) is estimated.  Measured accuracy vs sim ground truth
is in docs/TABLE_CAM.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera
from .objectlib import ObjInfo


@dataclass
class Detection:
    name: str
    xyz: np.ndarray      # object BOTTOM-centre in base frame (x, y, z_table); scene adds grasp_z
    confidence: float    # detector score
    px_err_cm: float     # estimated spatial error from ~loc_noise_px detector localisation noise (cm)
    bbox: tuple[float, float, float, float]  # u0, v0, u1, v1
    in_workspace: bool


def _bottom_center(bbox) -> np.ndarray:
    u0, v0, u1, v1 = bbox
    return np.array([(u0 + u1) / 2.0, max(v0, v1)])  # max row = lowest in the image


def _predict_bbox(cam: PinholeCamera, obj: ObjInfo, xy, z_table: float) -> np.ndarray:
    px = cam.project(obj.silhouette_points(xy, z_table))
    return np.array([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])


def estimate_pushback(cam: PinholeCamera, bbox, obj: ObjInfo, z_table: float) -> np.ndarray | None:
    """Simple estimator: bottom-centre back-projection + footprint-radius push-back (see module docstring)."""
    p_front = cam.ray_plane(_bottom_center(bbox), z_table)
    if p_front is None:
        return None
    if obj.radius <= 0:
        return p_front
    cxy = cam.center()[:2]
    az = p_front[:2] - cxy
    n = np.linalg.norm(az)
    if n < 1e-9:
        return p_front
    centre_xy = p_front[:2] + obj.radius * (az / n)
    return np.array([centre_xy[0], centre_xy[1], z_table])


def estimate_refine(cam: PinholeCamera, bbox, obj: ObjInfo, z_table: float, iters: int = 15) -> np.ndarray | None:
    """Default estimator: Gauss-Newton fit of (x, y) so the predicted silhouette box matches `bbox`."""
    seed = estimate_pushback(cam, bbox, obj, z_table)
    if seed is None:
        seed = cam.ray_plane(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2), z_table)
    if seed is None:
        return None
    xy = seed[:2].astype(float).copy()
    target = np.asarray(bbox, float)
    h = 1e-4
    for _ in range(iters):
        r = _predict_bbox(cam, obj, xy, z_table) - target
        J = np.zeros((4, 2))
        for j in range(2):
            d = np.zeros(2); d[j] = h
            J[:, j] = (_predict_bbox(cam, obj, xy + d, z_table) - _predict_bbox(cam, obj, xy - d, z_table)) / (2 * h)
        try:
            step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(2), J.T @ r)
        except np.linalg.LinAlgError:
            break
        xy = xy - step
        if np.linalg.norm(step) < 1e-6:
            break
    return np.array([xy[0], xy[1], z_table])


def estimate_xy(cam: PinholeCamera, bbox, obj: ObjInfo, z_table: float,
                method: str = "refine") -> np.ndarray | None:
    """bbox + known object -> footprint-centre (x, y, z_table). method = 'refine' (default) | 'pushback'."""
    if method == "pushback":
        return estimate_pushback(cam, bbox, obj, z_table)
    return estimate_refine(cam, bbox, obj, z_table)


def _pixel_sensitivity(cam: PinholeCamera, bbox, obj: ObjInfo, z_table: float, px: float, method: str) -> float:
    """Spatial error (cm) in (x, y) for a `px`-pixel perturbation of the box (RMS over 4 directions)."""
    base = estimate_xy(cam, bbox, obj, z_table, method)
    if base is None:
        return float("nan")
    u0, v0, u1, v1 = bbox
    moves = []
    for du, dv in ((px, 0), (-px, 0), (0, px), (0, -px)):
        p = estimate_xy(cam, (u0 + du, v0 + dv, u1 + du, v1 + dv), obj, z_table, method)
        if p is not None:
            moves.append(np.linalg.norm(p[:2] - base[:2]))
    return 100.0 * float(np.sqrt(np.mean(np.square(moves)))) if moves else float("nan")


def project_detection(cam: PinholeCamera, name: str, bbox, score: float, obj: ObjInfo,
                      z_table: float, workspace: tuple[np.ndarray, np.ndarray] | None = None,
                      min_conf: float = 0.0, loc_noise_px: float = 2.0,
                      method: str = "refine") -> Detection | None:
    """One detector box -> a Detection (or None if rejected: low confidence, ray misses the table,
    or the point falls outside the workspace box; xy only, z is pinned to the table)."""
    if score < min_conf:
        return None
    xyz = estimate_xy(cam, bbox, obj, z_table, method)
    if xyz is None:
        return None
    in_ws = True
    if workspace is not None:
        lo, hi = workspace
        in_ws = bool(lo[0] - 1e-6 <= xyz[0] <= hi[0] + 1e-6 and lo[1] - 1e-6 <= xyz[1] <= hi[1] + 1e-6)
        if not in_ws:
            return None
    px_err = _pixel_sensitivity(cam, bbox, obj, z_table, loc_noise_px, method)
    return Detection(name=name, xyz=xyz, confidence=float(score), px_err_cm=px_err,
                     bbox=tuple(float(b) for b in bbox), in_workspace=in_ws)
