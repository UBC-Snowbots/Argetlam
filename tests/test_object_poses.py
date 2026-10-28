"""perception.source: external - a remote device (the laptop RealSense) sends ObjectPoses (already-located objects);
the app feeds its tracker, the voice skills use the positions, and the sim objects are moved to match."""
import json
import socket
import threading
import time

import pytest

from rover_intent.config import load_config
from rover_intent.robot.backends import MockArm
from rover_intent.transport.udp import _MSG
from rover_intent.types import ObjectPoses


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_object_poses_parses_on_the_transport():
    m = _MSG.validate_python({"kind": "object_poses", "t": 1.0, "seq": 3, "source": "realsense",
                              "objects": {"cup": {"pos": [0.45, 0.2, 0.0], "conf": 0.9, "desc": "red cup"}}})
    assert isinstance(m, ObjectPoses)
    assert m.objects["cup"].pos == [0.45, 0.2, 0.0] and m.objects["cup"].desc == "red cup"
    assert m.objects["cup"].err_cm == 0.0


def test_app_external_source_pick_refusal_and_sim_mirror(monkeypatch):
    placed = []
    orig = MockArm.place_objects

    def spy(self, objects):
        orig(self, objects)
        placed.append(dict(self.placed))

    monkeypatch.setattr(MockArm, "place_objects", spy)
    from rover_intent.app import run
    cfg = load_config(arm="dev_arm")
    cfg["perception"]["source"] = "external"
    listen, hudp = _free_port(), _free_port()
    cfg["net"]["listen_port"] = listen
    cfg["net"]["hud_port"] = hudp
    th = threading.Thread(target=run, kwargs=dict(cfg=cfg, backend="mock", seconds=6.0, record=False), daemon=True)
    th.start()
    time.sleep(1.0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    app = ("127.0.0.1", listen)

    def send(d):
        client.sendto(json.dumps(d).encode(), app)

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
    # the RealSense sees only the cup (moved from its config spot) plus something the scene does not know
    send({"kind": "object_poses", "t": time.time(), "seq": 1, "source": "realsense",
          "objects": {"cup": {"pos": [0.50, 0.10, z], "conf": 0.9}, "screwdriver": {"pos": [0.4, 0.0, z]}}})
    time.sleep(0.4)

    send({"kind": "query", "t": time.time()})
    st = recv(lambda m: m.get("kind") == "state")
    assert st is not None and st.get("perception_source") == "external"
    assert st["objects"]["cup"]["seen"] and st["objects"]["cup"]["src"] == "realsense"
    assert not st["objects"]["bottle"]["seen"]

    # the sim was told to put the cup where the camera saw it (and only the cup)
    assert placed and set(placed[-1]) == {"cup"}
    assert placed[-1]["cup"][:2] == pytest.approx([0.50, 0.10], abs=1e-6)

    # seen cup -> pick accepted; unseen bottle -> refused with a spoken reason
    send({"kind": "utterance", "t": time.time(), "text": "pick up the cup"})
    r = recv(lambda m: m.get("kind") == "reply" and "cup" in (m.get("text") or ""))
    assert r is not None and "don't see" not in (r.get("say") or "").lower()
    send({"kind": "utterance", "t": time.time(), "text": "pick up the bottle"})
    r = recv(lambda m: m.get("kind") == "reply" and "bottle" in (m.get("text") or ""))
    assert r is not None and r.get("ok") is False and "don't see" in (r.get("say") or "").lower()

    send({"kind": "utterance", "t": time.time(), "text": "stop"})
    client.close()
    th.join(timeout=5.0)


def test_object_poses_ignored_unless_external(monkeypatch):
    """In the default sim mode an object_poses message changes nothing (objects stay sim ground truth)."""
    placed = []
    monkeypatch.setattr(MockArm, "place_objects", lambda self, o: placed.append(o))
    from rover_intent.app import run
    cfg = load_config(arm="dev_arm")
    listen, hudp = _free_port(), _free_port()
    cfg["net"]["listen_port"] = listen
    cfg["net"]["hud_port"] = hudp
    th = threading.Thread(target=run, kwargs=dict(cfg=cfg, backend="mock", seconds=3.0, record=False), daemon=True)
    th.start()
    time.sleep(1.0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    client.sendto(json.dumps({"kind": "object_poses", "t": time.time(), "objects": {"cup": {"pos": [0.5, 0.1, 0]}}})
                  .encode(), ("127.0.0.1", listen))
    time.sleep(0.3)
    client.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), ("127.0.0.1", listen))
    for _ in range(50):
        m = json.loads(client.recvfrom(65536)[0])
        if m.get("kind") == "state":
            assert m.get("perception_source") == "sim" and all(o["src"] == "sim" for o in m["objects"].values())
            break
    assert not placed
    client.close()
    th.join(timeout=5.0)
