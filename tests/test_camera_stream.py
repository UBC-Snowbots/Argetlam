"""Live webcam stream to the HUD: CameraFrame round trip + size, latest-only frame dropping, stale handling,
the app-side preset "busy" rule, that the video path never touches the pose/record path, and the client panel
layout at the smaller default window. cv2 / mediapipe are stubbed so the laptop client imports without them."""
import importlib.util
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from rover_intent.config import load_config
from rover_intent.session import SessionRecorder
from rover_intent.transport.udp import _MSG
from rover_intent.types import BodyPose, CameraFrame

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# --- message round trip + the recorder leaves the video out ----------------------------------------------------
def test_cameraframe_adapter_round_trip():
    raw = {"kind": "camera_frame", "t": 1.0, "jpeg_b64": "QUJD", "width": 480, "height": 360, "seq": 7,
           "source": "body_cam"}
    msg = _MSG.validate_python(raw)
    assert isinstance(msg, CameraFrame) and msg.seq == 7 and msg.jpeg_b64 == "QUJD"
    # a body pose still decodes to BodyPose: adding CameraFrame to the union did not shadow it
    body = _MSG.validate_python({"kind": "body", "t": 0.0, "wrist": [0, 0, 0], "shoulder": [0, 0, 0]})
    assert isinstance(body, BodyPose)


def test_recorder_skips_cameraframe_keeps_body(tmp_path):
    rec = SessionRecorder(tmp_path, enabled=True)
    rec.record(0.0, BodyPose(t=0.0, wrist=(0, 0, 0), shoulder=(0, 0, 0)))
    rec.record(0.1, CameraFrame(t=0.1, jpeg_b64="QUJD", width=4, height=4, seq=0))
    rec.close()
    lines = [json.loads(x) for x in rec.path.read_text().splitlines()]
    kinds = [x["msg"]["kind"] for x in lines]
    assert kinds == ["body"]  # the pose was recorded; the frame was not


# --- stale handling (pure helper from scripts/hud.py) ----------------------------------------------------------
def test_video_status_stale():
    spec = importlib.util.spec_from_file_location("hud_t", ROOT / "scripts/hud.py")
    hud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hud)
    assert hud.video_status(None, 0) == ("no camera", False)
    txt, stale = hud.video_status(0.2, 11)
    assert not stale and "11 fps" in txt
    txt, stale = hud.video_status(1.6, 11)
    assert stale and "stale 1.6s" in txt


# --- the laptop client: VideoStreamer (cv2 stubbed) + panel layout at the new size -----------------------------
@pytest.fixture(scope="module")
def body_client():
    stubs = {}
    cv2 = types.ModuleType("cv2")
    cv2.EVENT_LBUTTONDOWN, cv2.FONT_HERSHEY_SIMPLEX = 1, 0
    cv2.INTER_AREA, cv2.IMWRITE_JPEG_QUALITY = 3, 1
    cv2.rectangle = cv2.putText = lambda *a, **k: None
    cv2.resize = lambda frame, size, **k: ("resized", size)   # record the requested size; frame content unused
    cv2._nbytes = 1000
    cv2.imencode = lambda ext, img, params=None: (True, b"\x00" * cv2._nbytes)
    stubs["cv2"] = cv2
    mp = types.ModuleType("mediapipe")
    tasks, tpy = types.ModuleType("mediapipe.tasks"), types.ModuleType("mediapipe.tasks.python")
    tpy.BaseOptions, tpy.vision = object, types.SimpleNamespace()
    stubs.update({"mediapipe": mp, "mediapipe.tasks": tasks, "mediapipe.tasks.python": tpy})
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("body_client_t", ROOT / "laptop/body_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._cv2 = cv2
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_videostreamer_latest_only_and_seq(body_client):
    vs = body_client.VideoStreamer(width=480, height=360, quality=60, hz=10.0)
    m0 = vs.maybe("frame", 100.0)
    assert m0 is not None and m0["kind"] == "camera_frame" and m0["seq"] == 0
    assert m0["width"] == 480 and m0["height"] == 360 and m0["source"] == "body_cam"
    # a frame 50 ms later (< 100 ms period) is DROPPED, not queued: returns None, dropped counter climbs, seq held
    assert vs.maybe("frame", 100.05) is None
    assert vs.dropped == 1 and vs.sent == 1
    # past the period: a new frame is sent and the seq advances
    m1 = vs.maybe("frame", 100.2)
    assert m1 is not None and m1["seq"] == 1 and vs.sent == 2


def test_videostreamer_size_guard_warns_once(body_client, capsys):
    body_client._cv2._nbytes = 80000          # force an oversize JPEG for the UDP path
    vs = body_client.VideoStreamer(width=480, height=360, quality=60, hz=100.0, tunnel=False)
    vs.maybe("frame", 1000.0)
    out = capsys.readouterr().out
    assert "UDP safe size" in out and "--tunnel" in out
    # the tunnel path has no datagram limit, so it must NOT warn even for a big frame
    vt = body_client.VideoStreamer(width=480, height=360, quality=60, hz=100.0, tunnel=True)
    vt.maybe("frame", 1000.0)
    assert "UDP safe size" not in capsys.readouterr().out
    body_client._cv2._nbytes = 1000


def test_panel_inset_at_new_window_size(body_client):
    import numpy as np
    p = body_client.ButtonPanel(margin=40)
    frame = np.zeros((540, 960, 3), np.uint8)   # the new default window
    p.draw(frame, {"presets": ["return", "ready", "base"]}, 0.0)
    xs = [x for _, x, _, w, _ in p.rects]
    rights = [x + w for _, x, _, w, _ in p.rects]
    assert min(xs) == 960 - p.W - 40            # left edge honours the margin
    assert all(r <= 960 - 40 for r in rights)   # every button stays >= 40 px from the window's right edge


# --- app side: forwards frames to the HUD video port, and refuses a second preset while one is moving ----------
def _run_app(cfg, seconds):
    from rover_intent.app import run
    run(cfg, backend="mock", seconds=seconds, record=False)


def test_app_forwards_video_and_busy_rule():
    cfg = load_config(arm="dev_arm")
    listen = _free_port()
    video = _free_port()
    cfg["net"]["listen_port"] = listen
    cfg["net"]["hud_host"] = "127.0.0.1"
    cfg["net"]["hud_video_port"] = video
    cfg["net"]["hud_port"] = _free_port()   # keep test status off any live HUD on the default port
    th = threading.Thread(target=_run_app, args=(cfg, 4.0), daemon=True)
    th.start()
    time.sleep(1.0)  # let the app bind and reach its loop

    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", video))
    sink.settimeout(2.0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    app_addr = ("127.0.0.1", listen)

    def send(d):
        client.sendto(json.dumps(d).encode(), app_addr)

    def recv_reply(match, tries=200):
        for _ in range(tries):
            try:
                m = json.loads(client.recvfrom(65536)[0])
            except socket.timeout:
                break
            if match(m):
                return m
        return None

    # 1) a CameraFrame is forwarded verbatim to the HUD video port
    send({"kind": "camera_frame", "t": time.time(), "jpeg_b64": "SEVMTE8=", "width": 480, "height": 360, "seq": 3,
          "source": "body_cam"})
    fwd = json.loads(sink.recvfrom(200000)[0])
    assert fwd["kind"] == "camera_frame" and fwd["jpeg_b64"] == "SEVMTE8=" and fwd["seq"] == 3

    # 2) a preset starts a move (ok=True); a second preset while it is moving is refused as busy (not chain-replaced).
    #    "base" is far from home so the move lasts; "ready" == home would finish instantly, so drive "base" first.
    send({"kind": "preset", "t": time.time(), "name": "base", "confirm": False})
    r1 = recv_reply(lambda m: m.get("text") == "preset base")
    assert r1 is not None and r1.get("ok") is True
    send({"kind": "preset", "t": time.time(), "name": "ready", "confirm": False})
    r2 = recv_reply(lambda m: m.get("text") == "preset ready")
    assert r2 is not None and r2.get("ok") is False and r2.get("busy") is True
    assert "STOP" in (r2.get("say") or "")

    # 3) STOP always gets through and clears the move
    send({"kind": "utterance", "t": time.time(), "text": "stop"})
    time.sleep(0.3)
    sink.close()
    client.close()
    th.join(timeout=5.0)
