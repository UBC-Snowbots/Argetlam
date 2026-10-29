"""'grip is level': the app records the real gripper's level roll (real minus model) at the current pose, turns pick
levelling on, and saves the offset per arm profile."""
import json
import socket
import threading
import time
from pathlib import Path

from rover_intent.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_grip_is_level_records_and_saves_offset():
    from rover_intent.app import run
    cfg = load_config(arm="dev_arm_real")
    cfg["arm_profile"] = "pytest_levelcheck"
    cfg["perception"]["source"] = "sim"
    listen = _free_port()
    cfg["net"]["listen_port"], cfg["net"]["hud_port"] = listen, _free_port()
    saved = ROOT / "configs" / "arms" / "pytest_levelcheck_grip_level.json"
    saved.unlink(missing_ok=True)
    th = threading.Thread(target=run, kwargs=dict(cfg=cfg, backend="mock", seconds=4.0, record=False), daemon=True)
    th.start()
    time.sleep(1.0)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2.0)
    s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": "grip is level"}).encode(),
             ("127.0.0.1", listen))
    reply = None
    for _ in range(100):
        m = json.loads(s.recvfrom(65536)[0])
        if m.get("kind") == "reply":
            reply = m
            break
    th.join(timeout=6.0)
    try:
        assert reply is not None and reply["ok"] and "level" in reply["say"].lower()
        data = json.loads(saved.read_text())
        # at the base pose the model's level roll is ~36 deg from the real one (the base roll is called level here)
        assert abs(abs(data["offset_deg"]) - 35.8) < 1.0
    finally:
        saved.unlink(missing_ok=True)
