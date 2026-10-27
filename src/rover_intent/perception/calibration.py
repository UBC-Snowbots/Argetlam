"""Camera calibration with a ChArUco board (OpenCV): intrinsics from several board views,
extrinsics (camera pose in the arm base frame) from one board view at a KNOWN table pose.

Needs OpenCV with aruco (opencv-contrib-python[-headless], the `vision` extra).  Kept out of the
hot path: the control host calibrates once and loads the saved YAML at run time via
`PinholeCamera.from_dict`, which is pure numpy.

Real-setup procedure (no arm motion required) is in docs/TABLE_CAM.md.  In short: print a ChArUco
board, lay it flat on the table against marked spots whose (x, y) and yaw you measured from the arm
base, snap one frame -> extrinsics; wave it around the workspace, snap ~15 frames -> intrinsics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .camera import PinholeCamera


@dataclass
class BoardSpec:
    squares_x: int = 5
    squares_y: int = 7
    square_len: float = 0.035   # m, chessboard square side (printed size)
    marker_len: float = 0.026   # m, aruco marker side
    dictionary: str = "DICT_4X4_50"

    def build(self):
        import cv2
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_len, self.marker_len, d)
        return board, d

    def detector(self):
        import cv2
        board, _ = self.build()
        return cv2.aruco.CharucoDetector(board), board


def detect_charuco(image, spec: BoardSpec):
    """Detect ChArUco corners in one image. Returns (charuco_corners Nx1x2, charuco_ids Nx1) or (None, None)."""
    det, _ = spec.detector()
    ch_corners, ch_ids, _, _ = det.detectBoard(image)
    if ch_ids is None or len(ch_ids) < 4:
        return None, None
    return ch_corners, ch_ids


def calibrate_intrinsics_from_corners(obj_pts_list, img_pts_list, image_size):
    """Core intrinsics solve: lists of per-view board-frame object points (Nx3) and image points
    (Nx2).  Returns (K, dist, rms_reproj_px, n_views_used).  Used by both the image path and the
    synthetic-corner tests (which project known corners instead of detecting them in an image)."""
    import cv2
    all_obj = [np.asarray(o, np.float32).reshape(-1, 1, 3) for o in obj_pts_list if len(o) >= 4]
    all_img = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in img_pts_list if len(p) >= 4]
    if len(all_obj) < 3:
        raise ValueError(f"need >=3 usable ChArUco views, got {len(all_obj)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(all_obj, all_img, tuple(image_size), None, None)
    return K, dist.reshape(-1), float(rms), len(all_obj)


def calibrate_intrinsics(images, spec: BoardSpec, image_size=None):
    """Estimate K, dist from several ChArUco views. `images` = list of grayscale/BGR frames.

    Returns (K, dist, rms_reproj_px, n_views_used).
    """
    _, board = spec.detector()
    all_obj, all_img = [], []
    size = image_size
    for img in images:
        size = size or (img.shape[1], img.shape[0])
        ch_corners, ch_ids = detect_charuco(img, spec)
        if ch_ids is None:
            continue
        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue
        all_obj.append(obj_pts.reshape(-1, 3))
        all_img.append(img_pts.reshape(-1, 2))
    return calibrate_intrinsics_from_corners(all_obj, all_img, size)


def extrinsics_from_corners(obj_base, img_pts, K, dist, width=0, height=0) -> PinholeCamera:
    """Core extrinsics solve: base-frame object points (Nx3) and their image points (Nx2) ->
    camera pose in the base frame (a PinholeCamera).  Shared by the image path and the tests."""
    import cv2
    obj_base = np.asarray(obj_base, np.float32).reshape(-1, 3)
    img_pts = np.asarray(img_pts, np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_base, img_pts, np.asarray(K, float), np.asarray(dist, float),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed for extrinsics")
    return PinholeCamera.from_rvec_tvec(K, dist, rvec, tvec, width=width, height=height)


def _board_pose_matrix(origin_xy, z_table, yaw_rad) -> np.ndarray:
    """4x4 board-frame -> base-frame transform: board lies flat on z=z_table, rotated by yaw about +z,
    board origin at (origin_xy[0], origin_xy[1], z_table)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T[:3, 3] = [origin_xy[0], origin_xy[1], z_table]
    return T


def estimate_extrinsics(image, spec: BoardSpec, K, dist, origin_xy, z_table, yaw_rad=0.0) -> PinholeCamera:
    """One board view at a known table pose -> camera pose in the base frame (a PinholeCamera).

    The board lies flat on the table; its origin (x, y) and yaw are measured from the arm base.
    Each detected ChArUco corner has a known board-frame coordinate -> known base-frame coordinate;
    solvePnP(base points, image points) gives the base->camera transform.
    """
    _, board = spec.detector()
    ch_corners, ch_ids = detect_charuco(image, spec)
    if ch_ids is None:
        raise ValueError("no ChArUco board detected for extrinsics")
    obj_board, img_pts = board.matchImagePoints(ch_corners, ch_ids)  # obj in board frame
    T = _board_pose_matrix(origin_xy, z_table, yaw_rad)
    obj_base = (T[:3, :3] @ obj_board.reshape(-1, 3).T).T + T[:3, 3]
    h, w = image.shape[:2]
    return extrinsics_from_corners(obj_base, img_pts, K, dist, width=w, height=h)


def save_calibration(path, cam: PinholeCamera, spec: BoardSpec | None = None,
                     z_table: float = 0.0, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"camera": cam.to_dict(), "z_table": float(z_table)}
    if spec is not None:
        doc["board"] = spec.__dict__
    if meta:
        doc["meta"] = meta
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def load_calibration(path) -> tuple[PinholeCamera, float]:
    doc = yaml.safe_load(Path(path).read_text())
    return PinholeCamera.from_dict(doc["camera"]), float(doc.get("z_table", 0.0))
