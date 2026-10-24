"""Synthetic renderers for offline testing / validation (OpenCV, lazy-imported).

* render_board_view: render a ChArUco board lying flat on the table, as seen by a given
  PinholeCamera, via a planar homography.  Used by the calibration round-trip test and the
  sim-less calibration validation.
* render_scene_view: render the known objects (coloured cylinders/boxes on a plane) as seen by a
  camera.  A crude rasteriser good enough to exercise a real open-vocab detector when Isaac is not
  available; NOT physically accurate lighting.
"""
from __future__ import annotations

import numpy as np

from .camera import PinholeCamera
from .objectlib import ObjInfo


def _board_pose_matrix(origin_xy, z_table, yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T[:3, 3] = [origin_xy[0], origin_xy[1], z_table]
    return T


def render_board_view(cam: PinholeCamera, spec, origin_xy, z_table, yaw_rad=0.0, bg=210):
    """Render `spec` (a BoardSpec) lying flat at the given base-frame pose, as camera sees it.

    The board image is generated at ~1.3x its on-screen pixel size, so the homography warp
    up/slightly-downsamples rather than heavily downsamples (cv2.warpPerspective point-samples and
    would alias the markers into noise otherwise).
    """
    import cv2
    board, _ = spec.build()
    aspect = spec.squares_y / spec.squares_x
    # on-screen board size: project the board's 4 outer corners
    bw_m, bh_m = spec.squares_x * spec.square_len, spec.squares_y * spec.square_len
    T = _board_pose_matrix(origin_xy, z_table, yaw_rad)
    corners_board = np.array([[0, 0, 0], [bw_m, 0, 0], [bw_m, bh_m, 0], [0, bh_m, 0.0]])
    corners_base = (T[:3, :3] @ corners_board.T).T + T[:3, 3]
    cpx = cam.project(corners_base)
    screen_w = max(np.ptp(cpx[:, 0]), np.ptp(cpx[:, 1]), 60.0)
    # generate at a good resolution for clean markers, then INTER_AREA-resize to ~1.3x the
    # on-screen size so the homography warp (which point-samples) does not alias the markers.
    # Generate at a good resolution for crisp markers, then INTER_AREA-resize to ~1.3x the
    # on-screen size so the homography warp (which point-samples) does not alias the markers.
    gen_w = 1000
    board_img = board.generateImage((gen_w, int(round(gen_w * aspect))))
    target_w = int(np.clip(1.3 * screen_w, 320, gen_w))
    if target_w < gen_w:
        board_img = cv2.resize(board_img, (target_w, int(round(target_w * aspect))),
                               interpolation=cv2.INTER_AREA)
    # Homography from the board image's OWN detected ChArUco corners (OpenCV board-frame coords from
    # matchImagePoints) to their camera pixels.  Using matchImagePoints keeps the rendered board
    # self-consistent with estimate_extrinsics, which uses the same board-frame convention.
    det = cv2.aruco.CharucoDetector(board)
    ch_c, ch_ids, _, _ = det.detectBoard(board_img)
    if ch_ids is None or len(ch_ids) < 4:
        raise RuntimeError("could not detect the generated board (adjust board size)")
    obj_board, gen_px = board.matchImagePoints(ch_c, ch_ids)
    base = (T[:3, :3] @ obj_board.reshape(-1, 3).T).T + T[:3, 3]
    H, _ = cv2.findHomography(gen_px.reshape(-1, 2), cam.project(base).reshape(-1, 2))
    canvas = np.full((cam.height, cam.width), bg, np.uint8)
    warped = cv2.warpPerspective(board_img, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(board_img, 255), H, (cam.width, cam.height))
    canvas[mask > 127] = warped[mask > 127]
    return canvas


_COLORS_BGR = {"cup": (50, 60, 210), "bottle": (210, 120, 50), "block": (30, 190, 240)}


def render_scene_view(cam: PinholeCamera, lib: dict, layout: dict, z_table: float,
                      bg=(120, 120, 120), n_facets=40):
    """Render coloured cylinders/boxes standing at `layout` {name: (x, y)} as camera sees them.

    Painter's algorithm back-to-front; flat-shaded.  For exercising an open-vocab detector, not
    for metric accuracy (the detector's box is what feeds the metric path).
    """
    import cv2
    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    img[:] = bg
    order = sorted(layout, key=lambda n: -np.linalg.norm(cam.center()[:2] - np.asarray(layout[n])))
    for name in order:
        o: ObjInfo = lib[name]
        x, y = layout[name]
        col = _COLORS_BGR.get(name, (200, 200, 200))
        ang = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)
        if o.shape == "box":
            ring = [(x + sx * o.radius, y + sy * o.radius) for sx in (-1, 1) for sy in (1, -1)]
        else:
            ring = [(x + o.radius * np.cos(a), y + o.radius * np.sin(a)) for a in ang]
        bottom = np.array([[rx, ry, z_table] for rx, ry in ring])
        top = np.array([[rx, ry, z_table + o.height] for rx, ry in ring])
        bpx = cam.project(bottom).astype(np.int32)
        tpx = cam.project(top).astype(np.int32)
        # side wall as a filled polygon per facet
        shade = tuple(int(c * 0.8) for c in col)
        m = len(ring)
        for i in range(m):
            j = (i + 1) % m
            quad = np.array([bpx[i], bpx[j], tpx[j], tpx[i]])
            cv2.fillConvexPoly(img, quad, shade)
        cv2.fillConvexPoly(img, tpx, col)  # top cap
    return img
