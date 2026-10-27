"""Perception tests: projection maths (synthetic pinhole), calibration round-trip (projected ChArUco
corners), the perception config switch default, and CameraFrame transport parsing.

All run in the lightweight .venv (numpy / scipy / PIL / opencv / yaml / pydantic); none need torch.
"""
import base64
import io
import sys

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _real_cv2():
    """Other tests (test_link_relay / test_presets) stub `cv2` into sys.modules to load the laptop
    clients without OpenCV; test_link_relay does not clean up. Drop a stub (no `aruco`) so our
    calibration code re-imports the real OpenCV, regardless of test order."""
    mod = sys.modules.get("cv2")
    if mod is not None and not hasattr(mod, "aruco"):
        del sys.modules["cv2"]
    yield

from rover_intent.config import load_config
from rover_intent.perception.camera import PinholeCamera, intrinsics_from_fov, look_at
from rover_intent.perception.objectlib import ObjInfo, library_from_scene, set_support
from rover_intent.perception.projection import estimate_xy, project_detection


def _sim_cam(hfov=60.0, W=640, H=480, eye=(-0.5, -0.5, 0.75), target=(0.5, 0.0, 0.05)):
    R, t = look_at(eye, target)
    return PinholeCamera(K=intrinsics_from_fov(W, H, hfov), dist=np.zeros(5), R_cw=R, t_cw=t, width=W, height=H)


def _silhouette_bbox(cam, obj, xy, z):
    px = cam.project(obj.silhouette_points(xy, z))
    return (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max())


# --- camera model -----------------------------------------------------------
def test_project_backproject_roundtrip_exact():
    cam = _sim_cam()
    pts = np.array([[0.45, 0.2, 0.0], [0.45, -0.2, 0.0], [0.56, 0.0, 0.0]])
    for p in pts:
        uv = cam.project(p)[0]
        q = cam.ray_plane(uv, 0.0)
        assert np.linalg.norm(q - p) < 1e-9


def test_project_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    cam = _sim_cam()
    pts = np.array([[0.4, -0.1, 0.0], [0.6, 0.1, 0.05], [0.5, 0.0, 0.1]])
    mine = cam.project(pts)
    ocv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), cam.rvec, cam.t_cw, cam.K, cam.dist)
    assert np.allclose(mine, ocv.reshape(-1, 2), atol=1e-6)


def test_ray_plane_rejects_behind_camera():
    cam = _sim_cam()
    # a pixel whose ray goes up and away from the table should not hit z=0 in front of the camera
    assert cam.ray_plane((cam.width / 2, 0.0), 2.0) is None or cam.ray_plane((0, 0), 5.0) is not None


# --- projection accuracy (the gate) -----------------------------------------
def test_projection_accuracy_gate_synthetic():
    """Exact silhouette boxes + 2 px noise through the projection path must meet the gate."""
    cfg = load_config(arm="dev_arm")
    lib = library_from_scene(cfg["scene"])
    set_support(lib, cfg["table"]["top"])
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    z = cfg["table"]["top"]
    cam = _sim_cam()
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(40):
        for name, o in lib.items():
            x = rng.uniform(ws[0][0] + o.radius, ws[1][0] - o.radius)
            y = rng.uniform(ws[0][1] + o.radius, ws[1][1] - o.radius)
            bbox = np.array(_silhouette_bbox(cam, o, (x, y), z)) + rng.normal(0, 2.0, 4)
            est = estimate_xy(cam, tuple(bbox), o, z, method="refine")
            errs.append(100 * np.linalg.norm(est[:2] - np.array([x, y])))
    errs = np.array(errs)
    assert np.median(errs) < 2.0
    assert np.percentile(errs, 90) < 3.0


def test_projection_z_pinned_to_table():
    cam = _sim_cam()
    o = ObjInfo("cup", radius=0.035, height=0.09, grasp_z=0.055, shape="cylinder")
    bbox = _silhouette_bbox(cam, o, (0.5, 0.0), 0.0)
    est = estimate_xy(cam, bbox, o, 0.0)
    assert abs(est[2] - 0.0) < 1e-9


def test_workspace_rejection():
    cfg = load_config(arm="dev_arm")
    lib = library_from_scene(cfg["scene"])
    set_support(lib, 0.0)
    ws = (np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"]))
    cam = _sim_cam()
    o = lib["cup"]
    # an object placed well outside the workspace box must be rejected
    bbox = _silhouette_bbox(cam, o, (0.5, 0.0), 0.0)
    inside = project_detection(cam, "cup", bbox, 0.9, o, 0.0, workspace=ws)
    assert inside is not None and inside.in_workspace
    tiny = (0.0, 0.0, 1.0, 1.0)  # a box in the image corner -> table point outside the workspace
    assert project_detection(cam, "cup", tiny, 0.9, o, 0.0, workspace=ws) is None


def test_low_confidence_rejected():
    cam = _sim_cam()
    o = ObjInfo("cup", 0.035, 0.09, 0.055)
    bbox = _silhouette_bbox(cam, o, (0.5, 0.0), 0.0)
    assert project_detection(cam, "cup", bbox, 0.1, o, 0.0, min_conf=0.5) is None


# --- calibration round-trip (projected ChArUco corners) ---------------------
def test_calibration_extrinsics_roundtrip():
    cv2 = pytest.importorskip("cv2")
    from rover_intent.perception.calibration import BoardSpec, extrinsics_from_corners, _board_pose_matrix
    spec = BoardSpec(5, 7, 0.045, 0.034)
    board, _ = spec.build()
    corners3d = board.getChessboardCorners()
    W, H = 640, 480
    K = intrinsics_from_fov(W, H, 58)
    rng = np.random.default_rng(2)
    for s in range(4):
        eye = (0.5 + rng.uniform(-0.25, 0.25), rng.uniform(-0.35, 0.1), rng.uniform(0.5, 0.8))
        R, t = look_at(eye, (0.52, 0.0, 0.03))
        cam = PinholeCamera(K=K, dist=np.zeros(5), R_cw=R, t_cw=t, width=W, height=H)
        ox, yaw = (0.40, -0.12), rng.uniform(-0.5, 0.5)
        T = _board_pose_matrix(ox, 0.0, yaw)
        base = (T[:3, :3] @ corners3d.T).T + T[:3, 3]
        px = cam.project(base) + rng.normal(0, 0.2, (len(base), 2))
        rec = extrinsics_from_corners(base, px, K, np.zeros(5), W, H)
        assert np.linalg.norm(rec.center() - cam.center()) < 0.02  # < 2 cm


def test_calibration_intrinsics_roundtrip():
    cv2 = pytest.importorskip("cv2")
    from rover_intent.perception.calibration import (BoardSpec, calibrate_intrinsics_from_corners,
                                                     _board_pose_matrix)
    spec = BoardSpec(5, 7, 0.045, 0.034)
    board, _ = spec.build()
    corners3d = board.getChessboardCorners()
    W, H = 640, 480
    Ktrue = intrinsics_from_fov(W, H, 58)
    rng = np.random.default_rng(6)
    objs, imgs = [], []
    for _ in range(24):  # diverse camera poses (a near-fronto-parallel set is degenerate for fx)
        eye = (0.5 + rng.uniform(-0.35, 0.35), rng.uniform(-0.45, 0.25), rng.uniform(0.38, 0.85))
        R, t = look_at(eye, (0.5 + rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 0.0))
        T = _board_pose_matrix((0.4 + rng.uniform(-0.06, 0.06), -0.15 + rng.uniform(-0.06, 0.06)), 0.0,
                               rng.uniform(-1.2, 1.2))
        base = (T[:3, :3] @ corners3d.T).T + T[:3, 3]
        cam = PinholeCamera(K=Ktrue, R_cw=R, t_cw=t, width=W, height=H)
        px = cam.project(base) + rng.normal(0, 0.1, (len(base), 2))
        vis = (px[:, 0] > 0) & (px[:, 0] < W) & (px[:, 1] > 0) & (px[:, 1] < H)
        if vis.sum() >= 12:
            objs.append(corners3d[vis]); imgs.append(px[vis])
    K, dist, rms, n = calibrate_intrinsics_from_corners(objs, imgs, (W, H))
    assert n >= 15 and rms < 1.0
    assert abs(K[0, 0] - Ktrue[0, 0]) / Ktrue[0, 0] < 0.02   # fx within 2 %


def test_calibration_yaml_roundtrip(tmp_path):
    pytest.importorskip("cv2")
    from rover_intent.perception.calibration import save_calibration, load_calibration
    cam = _sim_cam()
    p = tmp_path / "calib.yaml"
    save_calibration(p, cam, z_table=0.0, meta={"source": "test"})
    cam2, z = load_calibration(p)
    assert z == 0.0
    assert np.allclose(cam.center(), cam2.center(), atol=1e-6)
    assert np.allclose(cam.K, cam2.K, atol=1e-6)


# --- config switch default --------------------------------------------------
def test_perception_source_defaults_to_sim():
    for arm in ("dev_arm", "dev_arm_real", "rover2026"):
        cfg = load_config(arm=arm)
        assert cfg.get("perception", {}).get("source", "sim") == "sim"


# --- transport message parsing ----------------------------------------------
def test_cameraframe_parse_roundtrip():
    from rover_intent.types import CameraFrame
    from pydantic import TypeAdapter
    from rover_intent.types import ArmPreset, BodyPose, ClutchMsg, EEGState, GripToggle, Query, Utterance
    adapter = TypeAdapter(BodyPose | Utterance | EEGState | GripToggle | ArmPreset | Query | ClutchMsg | CameraFrame)
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    msg = CameraFrame(t=1.0, jpeg_b64=b64, width=8, height=6, seq=3)
    import json
    parsed = adapter.validate_python(json.loads(msg.model_dump_json()))
    assert isinstance(parsed, CameraFrame)
    assert parsed.seq == 3 and parsed.width == 8
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(parsed.jpeg_b64))).convert("RGB"))
    assert img.shape == (6, 8, 3)
