"""Pinhole camera model (OpenCV convention) with extrinsics in the arm base frame.

Frames
------
* camera frame: OpenCV optical frame (+x right, +y down, +z into the scene).
* base frame: the arm base frame used everywhere else (x forward, y left, z up, metres).

Calibration stores the OpenCV solvePnP result (rvec, tvec) which is the BASE -> CAMERA
transform: X_cam = R_cw @ X_base + t_cw.  From it we derive the camera centre in the base
frame C = -R_cw^T @ t_cw and the camera->base rotation R_wc = R_cw^T.

The whole class is pure numpy for the no-distortion path (so the control-host `.venv`, which has
no OpenCV, can import and run it).  Distorted pixels need OpenCV; `pixel_to_ray` raises a clear
error if a non-zero distortion is set and cv2 is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 matrix (numpy port of cv2.Rodrigues, so .venv needs no cv2)."""
    rvec = np.asarray(rvec, float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _inv_rodrigues(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> rotation vector."""
    R = np.asarray(R, float)
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if theta < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(theta)) * theta


@dataclass
class PinholeCamera:
    """Intrinsics K (3x3), distortion (k1,k2,p1,p2,k3), and base->camera extrinsics (R_cw, t_cw)."""
    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R_cw: np.ndarray = field(default_factory=lambda: np.eye(3))  # base -> camera rotation
    t_cw: np.ndarray = field(default_factory=lambda: np.zeros(3))  # base -> camera translation
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.dist = np.asarray(self.dist, float).reshape(-1)
        self.R_cw = np.asarray(self.R_cw, float).reshape(3, 3)
        self.t_cw = np.asarray(self.t_cw, float).reshape(3)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_rvec_tvec(cls, K, dist, rvec, tvec, width=0, height=0) -> "PinholeCamera":
        """rvec/tvec as returned by cv2.solvePnP (base points -> camera)."""
        return cls(K=K, dist=dist, R_cw=_rodrigues(rvec), t_cw=np.asarray(tvec, float).reshape(3),
                   width=width, height=height)

    @property
    def rvec(self) -> np.ndarray:
        return _inv_rodrigues(self.R_cw)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist) > 1e-9))

    # --- geometry ---------------------------------------------------------
    def center(self) -> np.ndarray:
        """Camera optical centre in the base frame."""
        return -self.R_cw.T @ self.t_cw

    def _normalized(self, uv: np.ndarray) -> np.ndarray:
        """Pixel(s) -> normalized camera coords (x', y') on z=1, undistorted."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        if self.has_distortion:
            try:
                import cv2
            except ImportError as e:  # pragma: no cover - exercised only on a distorted real cam w/o cv2
                raise RuntimeError("distorted intrinsics need OpenCV (pip install opencv-contrib-python) "
                                   "to undistort pixels") from e
            pts = uv.reshape(-1, 1, 2).astype(np.float64)
            und = cv2.undistortPoints(pts, self.K, self.dist)  # -> normalized coords
            return und.reshape(-1, 2)
        Kinv = np.linalg.inv(self.K)
        hom = np.hstack([uv, np.ones((len(uv), 1))])
        n = (Kinv @ hom.T).T
        return n[:, :2] / n[:, 2:3]

    def pixel_to_ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> (origin, direction) of the viewing ray in the base frame (direction normalized)."""
        xyn = self._normalized(uv)[0]
        d_cam = np.array([xyn[0], xyn[1], 1.0])
        d_base = self.R_cw.T @ d_cam
        d_base = d_base / np.linalg.norm(d_base)
        return self.center(), d_base

    def ray_plane(self, uv, z_plane: float) -> np.ndarray | None:
        """Intersect the viewing ray through `uv` with the horizontal plane z=z_plane (base frame).

        Returns the (x, y, z_plane) point, or None if the ray is parallel to the plane or the
        intersection is behind the camera (the plane is not actually seen through that pixel).
        """
        C, d = self.pixel_to_ray(uv)
        if abs(d[2]) < 1e-9:
            return None
        s = (z_plane - C[2]) / d[2]
        if s <= 0:  # intersection behind the camera
            return None
        return C + s * d

    def project(self, pts_base) -> np.ndarray:
        """Base-frame point(s) -> pixel(s). Uses the pinhole + distortion model."""
        pts = np.asarray(pts_base, float).reshape(-1, 3)
        cam = (self.R_cw @ pts.T).T + self.t_cw  # base -> camera
        z = np.clip(cam[:, 2:3], 1e-9, None)
        xn = cam[:, :2] / z
        if self.has_distortion:
            k1, k2, p1, p2 = self.dist[0], self.dist[1], self.dist[2], self.dist[3]
            k3 = self.dist[4] if self.dist.size > 4 else 0.0
            x, y = xn[:, 0], xn[:, 1]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            xn = np.stack([xd, yd], axis=1)
        u = self.fx * xn[:, 0] + self.K[0, 2]
        v = self.K[1, 1] * xn[:, 1] + self.K[1, 2]
        return np.stack([u, v], axis=1)

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "dist": self.dist.tolist(),
                "rvec": self.rvec.tolist(), "tvec": self.t_cw.tolist(),
                "width": int(self.width), "height": int(self.height),
                "frame": "base", "model": "opencv_pinhole"}

    @classmethod
    def from_dict(cls, d: dict) -> "PinholeCamera":
        return cls.from_rvec_tvec(d["K"], d.get("dist", np.zeros(5)), d["rvec"], d["tvec"],
                                  d.get("width", 0), d.get("height", 0))


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """A plain pinhole K from an image size and a horizontal field of view (square pixels, centred)."""
    fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1.0]])


def look_at(eye, target, up=(0, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    """Build (R_cw, t_cw) for a camera at `eye` looking at `target`, OpenCV optical frame.

    OpenCV optical axis is +z (forward into the scene), +x right, +y down.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f = f / np.linalg.norm(f)          # camera +z (forward)
    r = np.cross(f, up)                 # camera +x (right); image-up is world-ish up
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([0, 1.0, 0]))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)                  # camera +y (down)
    R_wc = np.stack([r, d, f], axis=1)  # columns = camera axes in base = base->? actually cam->base
    R_cw = R_wc.T
    t_cw = -R_cw @ eye
    return R_cw, t_cw
