"""Rover-camera perception tests (additions to test_perception.py): camera-from-config round trips,
the GroundTruth and ArUco detectors, the TablePlaneLocator, the staleness/smoothing tracker, the
automatic scorer on a synthetic frame set, and app integration (sim default unchanged; camera source
drives a pick; an unseen object is refused).

All run in the lightweight .venv (numpy / scipy / PIL / opencv / yaml / pydantic); none need torch.
"""
import base64
import io
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _real_cv2():
    """Other tests stub a (no-aruco) `cv2` into sys.modules without cleanup; drop such a stub so the
    real OpenCV is re-imported regardless of test order (same guard as test_perception.py)."""
    mod = sys.modules.get("cv2")
    if mod is not None and not hasattr(mod, "aruco"):
        del sys.modules["cv2"]
    yield


from rover_intent.config import load_config  # noqa: E402
from rover_intent.perception.camera import PinholeCamera, camera_from_config, intrinsics_from_fov, look_at  # noqa: E402
from rover_intent.perception.detectors import ArucoDetector, GroundTruthDetector, NullDetector  # noqa: E402
from rover_intent.perception.locator import TablePlaneLocator  # noqa: E402
from rover_intent.perception.objectlib import library_from_scene, set_support  # noqa: E402
from rover_intent.perception.projection import Detection, estimate_xy  # noqa: E402
from rover_intent.perception.source import PerceptionTracker, build_from_config  # noqa: E402


def _cam(hfov=60.0, W=640, H=480, eye=(-0.35, -0.35, 0.8), target=(0.5, 0.0, 0.0)):
    R, t = look_at(eye, target)
    return PinholeCamera(K=intrinsics_from_fov(W, H, hfov), dist=np.zeros(5), R_cw=R, t_cw=t, width=W, height=H)


def _lib_z(arm="dev_arm"):
    cfg = load_config(arm=arm)
    z = cfg["table"]["top"]
    lib = library_from_scene(cfg["scene"])
    set_support(lib, z)
    return cfg, lib, z


# --- camera from config ----------------------------------------------------------------------------
def test_camera_from_config_look_at_matches_direct():
    cfg = {"eye": [-0.35, -0.35, 0.8], "look_at": [0.5, 0.0, 0.0], "width": 640, "height": 480, "hfov_deg": 60}
    cam, z = camera_from_config(cfg, z_table_default=0.0)
    ref = _cam()
    assert np.allclose(cam.center(), ref.center(), atol=1e-9)
    assert np.allclose(cam.R_cw, ref.R_cw, atol=1e-9)
    # project / unproject round trip on the table plane
    for p in ([0.45, 0.2, 0.0], [0.56, 0.0, 0.0], [0.45, -0.2, 0.0]):
        uv = cam.project(p)[0]
        q = cam.ray_plane(uv, 0.0)
        assert np.linalg.norm(q - p) < 1e-9


def test_camera_from_config_quat_round_trip():
    ref = _cam()
    # feed the reference camera's own pose back in as pos + quat (camera->base rotation R_wc = R_cw.T)
    from rover_intent.perception.camera import quat_to_R
    R_wc = ref.R_cw.T
    # build a quaternion from R_wc (shepperd), then check quat_to_R recovers it and the camera matches
    m = R_wc
    t = np.trace(m)
    w = np.sqrt(max(0.0, 1 + t)) / 2
    x = (m[2, 1] - m[1, 2]) / (4 * w); y = (m[0, 2] - m[2, 0]) / (4 * w); z = (m[1, 0] - m[0, 1]) / (4 * w)
    cam, _ = camera_from_config({"pos": ref.center().tolist(), "quat": [x, y, z, w],
                                 "width": 640, "height": 480, "hfov_deg": 60})
    assert np.allclose(cam.center(), ref.center(), atol=1e-6)
    assert np.allclose(cam.R_cw, ref.R_cw, atol=1e-6)


def test_camera_from_config_needs_pose():
    with pytest.raises(ValueError):
        camera_from_config({"width": 640, "height": 480, "hfov_deg": 60})


# --- TablePlaneLocator -----------------------------------------------------------------------------
def test_table_plane_locator_pixel_and_bbox():
    cfg, lib, z = _lib_z()
    cam = _cam()
    loc = TablePlaneLocator(cam, z)
    p = np.array([0.5, 0.0, z])
    uv = cam.project(p)[0]
    assert np.linalg.norm(loc.locate_pixel(uv) - p) < 1e-9     # ray-plane == camera.ray_plane
    o = lib["cup"]
    px = cam.project(o.silhouette_points((0.5, 0.0), z))
    bbox = (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max())
    assert np.allclose(loc.locate_bbox(bbox, o), estimate_xy(cam, bbox, o, z))  # == projection.estimate_xy


# --- GroundTruthDetector ---------------------------------------------------------------------------
def test_ground_truth_detector_exact():
    cfg, lib, z = _lib_z()
    cam = _cam()
    gt = {n: [o["pos"][0], o["pos"][1], z] for n, o in cfg["scene"].items()}
    dets = GroundTruthDetector(cam, lib, z, pos_noise_cm=0.0).detect(gt_positions=gt)
    assert {d.name for d in dets} == set(gt) and all(d.source == "ground_truth" for d in dets)
    for d in dets:
        assert np.linalg.norm(d.xyz[:2] - np.array(gt[d.name][:2])) < 1e-9
        assert abs(d.xyz[2] - z) < 1e-12 and d.centroid_px is not None


def test_ground_truth_detector_noise_bounded():
    cfg, lib, z = _lib_z()
    cam = _cam()
    gt = {n: [o["pos"][0], o["pos"][1], z] for n, o in cfg["scene"].items()}
    errs = []
    for s in range(30):
        for d in GroundTruthDetector(cam, lib, z, pos_noise_cm=1.0, seed=s).detect(gt_positions=gt):
            errs.append(100 * np.linalg.norm(d.xyz[:2] - np.array(gt[d.name][:2])))
    errs = np.array(errs)
    assert 0.3 < np.mean(errs) < 3.0   # ~1 cm sigma per axis -> mean |err| around 1.25 cm


def test_ground_truth_detector_out_of_frame_dropped():
    cfg, lib, z = _lib_z()
    cam = _cam()
    gt = {"cup": [0.45, 0.2, z], "bottle": [5.0, 5.0, z]}   # bottle far off to the side -> out of frame
    names = {d.name for d in GroundTruthDetector(cam, lib, z, drop_out_of_frame=True).detect(gt_positions=gt)}
    assert "cup" in names and "bottle" not in names


def test_ground_truth_detector_occlusion():
    cfg, lib, z = _lib_z()
    cam = _cam()
    # two objects nearly on the same viewing line: the farther one's centroid falls in the nearer box
    gt = {"cup": [0.47, 0.0, z], "block": [0.60, 0.0, z]}
    base = {d.name for d in GroundTruthDetector(cam, lib, z, occlusion=False).detect(gt_positions=gt)}
    occ = {d.name for d in GroundTruthDetector(cam, lib, z, occlusion=True).detect(gt_positions=gt)}
    assert base == {"cup", "block"} and len(occ) <= len(base)


# --- ArucoDetector on a synthetic rendered marker --------------------------------------------------
def test_aruco_detector_synthetic_scene():
    pytest.importorskip("cv2")
    from rover_intent.perception import synth
    cfg, lib, z = _lib_z()
    cam = _cam()
    marker_map = {0: "cup", 1: "bottle", 2: "block"}
    layout = {"cup": (0.45, 0.2), "bottle": (0.45, -0.2), "block": (0.56, 0.0)}
    img = synth.render_aruco_scene(cam, layout, marker_map, z, marker_len=0.05)
    dets = ArucoDetector(cam, marker_map, z, marker_length=0.05).detect(img)  # img is RGB
    found = {d.name: d.xyz for d in dets}
    assert set(found) == set(layout) and all(d.source == "aruco" for d in dets)
    for n, (x, y) in layout.items():
        assert 100 * np.linalg.norm(found[n][:2] - np.array([x, y])) < 2.0  # < 2 cm


def test_aruco_detector_ignores_unknown_id():
    pytest.importorskip("cv2")
    from rover_intent.perception import synth
    cfg, lib, z = _lib_z()
    cam = _cam()
    img = synth.render_aruco_scene(cam, {"cup": (0.45, 0.2)}, {7: "cup"}, z, marker_len=0.05)
    # the detector's map knows id 0..2, not 7 -> the rendered tag is ignored, no phantom object
    assert ArucoDetector(cam, {0: "cup"}, z).detect(img) == []


# --- tracker: staleness / confidence / smoothing ---------------------------------------------------
def _det(name, xyz, conf=1.0):
    return Detection(name=name, xyz=np.asarray(xyz, float), confidence=conf, px_err_cm=0.0,
                     bbox=(0, 0, 1, 1), in_workspace=True, source="test")


def test_tracker_staleness():
    tr = PerceptionTracker(min_conf=0.25, stale_s=1.0, alpha=1.0)
    tr.update([_det("cup", [0.4, 0.2, 0.0])], now=100.0)
    assert tr.seen("cup", 100.5) and "cup" in tr.updates(100.5)
    assert not tr.seen("cup", 101.5) and "cup" not in tr.updates(101.5)  # went stale
    assert not tr.seen("bottle", 100.5)                                   # never seen


def test_tracker_confidence_gate():
    tr = PerceptionTracker(min_conf=0.5)
    tr.update([_det("cup", [0.4, 0.2, 0.0], conf=0.3)], now=0.0)
    assert not tr.seen("cup", 0.0)   # below min_conf: not tracked


def test_tracker_smoothing_ema():
    tr = PerceptionTracker(min_conf=0.0, stale_s=10.0, alpha=0.5)
    tr.update([_det("cup", [0.0, 0.0, 0.0])], now=0.0)
    tr.update([_det("cup", [1.0, 0.0, 0.0])], now=0.1)
    assert abs(tr.updates(0.1)["cup"][0] - 0.5) < 1e-9   # EMA of 0 and 1 at alpha 0.5
    prov = tr.provenance(0.1)["cup"]
    assert prov["source"] == "test" and prov["fresh"]


def test_null_detector_is_plugin_slot():
    assert NullDetector().detect(np.zeros((4, 4, 3), np.uint8)) == []


# --- automatic scorer on a small synthetic frame set -----------------------------------------------
def test_perception_eval_gate_ground_truth():
    spec = _load_eval()
    cfg = load_config(arm="dev_arm")
    cfg.setdefault("perception", {})["source"] = "camera"
    cfg["perception"].setdefault("detector", {})["backend"] = "ground_truth"
    perc = build_from_config(cfg, with_detector=True)
    layouts = spec.random_layouts(perc.lib, perc.workspace, perc.z_table, 10, seed=0)
    frames = spec.synth_frames(perc, cfg, layouts, "ground_truth", perc.z_table)
    r = spec.score(perc, frames, "ground_truth")
    assert r["pass_gate"] and r["detection_rate"] == 1.0 and r["false_positives"] == 0
    assert r["median_cm"] < 2.0 and r["p90_cm"] < 3.0


def test_perception_eval_gate_aruco():
    pytest.importorskip("cv2")
    spec = _load_eval()
    cfg = load_config(arm="dev_arm")
    cfg.setdefault("perception", {})["source"] = "camera"
    cfg["perception"].setdefault("detector", {})["backend"] = "aruco"
    perc = build_from_config(cfg, with_detector=True)
    layouts = spec.random_layouts(perc.lib, perc.workspace, perc.z_table, 10, seed=1)
    frames = spec.synth_frames(perc, cfg, layouts, "aruco", perc.z_table)
    r = spec.score(perc, frames, "aruco")
    assert r["pass_gate"] and r["false_positives"] == 0 and r["median_cm"] < 2.0


def _load_eval():
    import importlib.util
    spec = importlib.util.spec_from_file_location("perception_eval_t", ROOT / "scripts/perception_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- app integration -------------------------------------------------------------------------------
def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _tiny_jpeg_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 30, 30)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _camera_app(source, backend):
    cfg = load_config(arm="dev_arm")
    cfg["perception"]["source"] = source
    cfg["perception"]["detector"]["backend"] = backend
    listen, video, hudp = _free_port(), _free_port(), _free_port()
    cfg["net"]["listen_port"] = listen
    cfg["net"]["hud_host"] = "127.0.0.1"
    cfg["net"]["hud_video_port"] = video
    cfg["net"]["hud_port"] = hudp
    return cfg, listen


def test_app_camera_source_pick_and_refusal():
    """camera source + ground_truth detector: a seen object drives a pick; an unseen one is refused."""
    from rover_intent.app import run
    cfg, listen = _camera_app("camera", "ground_truth")
    th = threading.Thread(target=run, kwargs=dict(cfg=cfg, backend="mock", seconds=6.0, record=False),
                          daemon=True)
    th.start()
    time.sleep(1.0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    app_addr = ("127.0.0.1", listen)

    def send(d):
        client.sendto(json.dumps(d).encode(), app_addr)

    def recv(match, tries=300):
        for _ in range(tries):
            try:
                m = json.loads(client.recvfrom(65536)[0])
            except socket.timeout:
                return None
            if match(m):
                return m
        return None

    z = cfg["table"]["top"]
    # ground-truth objects ride in the CameraFrame packet (as the bridge sends them). Only the cup here.
    send({"kind": "camera_frame", "t": time.time(), "jpeg_b64": _tiny_jpeg_b64(), "width": 640,
          "height": 480, "seq": 1, "source": "rover_cam",
          "objects": {"cup": [0.45, 0.20, z]}})
    time.sleep(0.5)  # let the detection be ingested by the tracker

    # the cup IS seen -> a pick is accepted (not a "don't see" refusal)
    send({"kind": "utterance", "t": time.time(), "text": "pick up the cup"})
    r_cup = recv(lambda m: m.get("kind") == "reply" and "cup" in (m.get("text") or ""))
    assert r_cup is not None and "don't see" not in (r_cup.get("say") or "").lower()

    # the bottle is NOT in the frame -> refused with a clear spoken reason
    send({"kind": "utterance", "t": time.time(), "text": "pick up the bottle"})
    r_bot = recv(lambda m: m.get("kind") == "reply" and "bottle" in (m.get("text") or ""))
    assert r_bot is not None and r_bot.get("ok") is False and "don't see" in (r_bot.get("say") or "").lower()

    send({"kind": "utterance", "t": time.time(), "text": "stop"})
    client.close()
    th.join(timeout=5.0)


def test_app_sim_source_default_unchanged():
    """Default perception source (sim): objects come from the mock backend ground truth; a status query
    reports perception_source sim and the objects seen, exactly as before camera perception existed."""
    from rover_intent.app import run
    cfg = load_config(arm="dev_arm")   # default perception.source == sim
    listen, hudp = _free_port(), _free_port()
    cfg["net"]["listen_port"] = listen
    cfg["net"]["hud_port"] = hudp
    assert cfg["perception"]["source"] == "sim"
    th = threading.Thread(target=run, kwargs=dict(cfg=cfg, backend="mock", seconds=4.0, record=False),
                          daemon=True)
    th.start()
    time.sleep(1.0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    client.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), ("127.0.0.1", listen))
    for _ in range(50):
        try:
            m = json.loads(client.recvfrom(65536)[0])
        except socket.timeout:
            break
        if m.get("kind") == "state":
            assert m.get("perception_source") == "sim"
            assert set(m["objects"]) == {"cup", "bottle", "block"}
            assert all(o.get("src") == "sim" and o.get("seen") for o in m["objects"].values())
            break
    else:
        pytest.fail("no state reply")
    client.close()
    th.join(timeout=5.0)


# --- digital-twin mounts / models ------------------------------------------------------------------
def test_camera_mount_and_model_resolution():
    cfg = load_config(arm="dev_arm")["camera"]
    front, z = camera_from_config(dict(cfg, mount="rover_front", model="vga_default"), 0.0)
    assert front.width == 640 and front.height == 480
    assert np.allclose(front.center(), [-0.35, -0.35, 0.80], atol=1e-9)
    over, _ = camera_from_config(dict(cfg, mount="overhead", model="usb_1080p"), 0.0)
    assert over.width == 1920 and over.height == 1080 and over.center()[2] > front.center()[2]
    with pytest.raises(ValueError):
        camera_from_config(dict(cfg, mount="nope"), 0.0)


def test_camera_calib_intrinsics_override(tmp_path):
    pytest.importorskip("cv2")
    from rover_intent.perception.calibration import save_calibration
    cam = _cam()
    p = tmp_path / "real_calib.yaml"
    save_calibration(p, cam, z_table=0.0)
    out, _ = camera_from_config({"mount": "rover_front", "calib": str(p),
                                 "mounts": {"rover_front": {"eye": [-0.35, -0.35, 0.8], "look_at": [0.5, 0, 0]}}}, 0.0)
    assert np.allclose(out.K, cam.K, atol=1e-6)   # K came from the calibration file


# --- sim/rover_camera.py pure helpers --------------------------------------------------------------
def _rover_camera_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rover_camera_t", ROOT / "sim/rover_camera.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rover_camera_offset_and_boxes():
    rc = _rover_camera_mod()
    cfg, lib, z = _lib_z()
    pos, quat, K, W, H, zt = rc.camera_offset(cfg["camera"], z_table_default=z)
    assert len(pos) == 3 and len(quat) == 4 and abs(np.linalg.norm(quat) - 1.0) < 1e-6
    assert W == 640 and H == 480
    cam = _cam(eye=pos, target=cfg["camera"]["mounts"]["rover_front"]["look_at"])
    gt = {n: [o["pos"][0], o["pos"][1], z] for n, o in cfg["scene"].items()}
    boxes = rc.gt_boxes_from_scene(cam, lib, gt, z)
    assert set(boxes) == set(gt)
    for b in boxes.values():
        assert 0 <= b[0] < b[2] <= 640 and 0 <= b[1] < b[3] <= 480


def test_rover_camera_boxes_from_label_map_and_frame():
    rc = _rover_camera_mod()
    seg = np.zeros((20, 30), int)
    seg[5:10, 4:12] = 3
    boxes = rc.boxes_from_label_map(seg, {3: "cup", 9: "bottle"})
    assert boxes == {"cup": [4.0, 5.0, 11.0, 9.0]}   # id 9 absent -> no bottle
    frame = rc.build_frame(np.zeros((4, 4, 3), np.uint8), {"cup": [0.4, 0.2, 0.0]}, 1.0, 2, 4, 4,
                           source="sim:overhead")
    assert frame["kind"] == "camera_frame" and frame["source"] == "sim:overhead"
    assert frame["objects"] == {"cup": [0.4, 0.2, 0.0]} and frame["seq"] == 2 and frame["jpeg_b64"]


def test_rover_camera_depth_png_round_trip():
    pytest.importorskip("PIL")
    from PIL import Image
    rc = _rover_camera_mod()
    depth = np.array([[0.5, 1.0], [2.0, np.inf]])
    b64 = rc.encode_depth_png(depth)
    mm = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert mm[0, 0] == 500 and mm[0, 1] == 1000 and mm[1, 0] == 2000 and mm[1, 1] == 0  # inf -> 0


# --- camera_tx (device sender) pure helpers --------------------------------------------------------
def test_camera_tx_parse_source_and_frame():
    pytest.importorskip("cv2")
    from rover_intent.perception import camera_tx
    assert camera_tx.parse_source("0") == 0
    assert camera_tx.parse_source("/dev/video1") == "/dev/video1"
    assert camera_tx.parse_source("rtsp://x/y") == "rtsp://x/y"
    bgr = np.zeros((48, 64, 3), np.uint8)
    frame, nbytes = camera_tx.build_camera_frame(bgr, 5, source_label="rover_cam", width=32, height=24)
    assert frame["kind"] == "camera_frame" and frame["source"] == "rover_cam" and frame["seq"] == 5
    assert frame["width"] == 32 and frame["height"] == 24 and nbytes > 0


# --- viewer overlay (pure) -------------------------------------------------------------------------
def test_viewer_annotate_and_banner():
    pytest.importorskip("PIL")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sim_cam_viewer_t", ROOT / "scripts/sim_cam_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.source_banner("sim:rover_front") == "SIM CAMERA (rover_front)"
    assert mod.source_banner("rover_cam").startswith("REAL CAMERA")
    cfg, lib, z = _lib_z()
    cam = _cam()
    rgb = np.zeros((480, 640, 3), np.uint8)
    dets = [{"name": "cup", "xyz": [0.45, 0.2, z], "src": "aruco", "seen": True}]
    im = mod.annotate_frame(rgb, cam, {"cup": [100, 100, 140, 160]}, dets, "sim:rover_front")
    assert im.size == (640, 480)
    assert np.asarray(im).sum() > 0   # overlays drew something on the black frame
