"""Real -> sim "mirror real" state (app side) and the sim->real follower's one-way loop safety.

App: a MirrorMsg enters/leaves the state; while mirroring the sim joint targets come straight from real_joints (no
IK), body/voice motion is refused, STOP leaves the mirror, and status/query expose mirror + the real-sample age. If
real samples go stale (> 0.5 s) the sim holds. Follower: SimQuery reads `mirror` from the app Query so the follower
can hold (zero velocity) and never close a real->sim->real loop.
"""
import importlib.util
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

from rover_intent import app
from rover_intent.config import load_config
from rover_intent.control.stepper_map import q_to_fw

ROOT = Path(__file__).resolve().parents[1]


class _App:
    def __init__(self, port, seconds=9.0):
        cfg = load_config(arm="dev_arm_real")
        cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
        cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
        self.cfg, self.home = cfg, np.array(cfg["arm"]["home"], float)
        self.th = threading.Thread(target=app.run, args=(cfg, "mock", seconds), kwargs={"record": False}, daemon=True)
        self.th.start(); time.sleep(0.5)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.settimeout(3)
        self.addr = ("127.0.0.1", port)

    def send(self, m):
        self.s.sendto(json.dumps(m).encode(), self.addr)

    def query(self):
        self.send({"kind": "query", "t": time.time()})
        while True:
            m = json.loads(self.s.recvfrom(65536)[0])
            if m.get("kind") == "state":
                return m

    def stream_real(self, q, secs, hz=30.0):
        fw = q_to_fw(q) + [0.0]
        t_end = time.time() + secs
        while time.time() < t_end:
            self.send({"kind": "real_joints", "t": time.time(), "fw": fw, "q": list(q)})
            time.sleep(1 / hz)

    def join(self):
        self.th.join(timeout=12)


def test_mirror_drives_joints_refuses_motion_and_stop_leaves():
    a = _App(47195, seconds=10.0)
    assert a.query()["mirror"] is False

    a.send({"kind": "mirror", "t": time.time(), "on": True})
    time.sleep(0.2)
    st = a.query()
    assert st["mirror"] is True

    target = (a.home + np.array([0.05, 0.03, -0.04, 0.0, 0.0, 0.0])).tolist()
    a.stream_real(target, 2.5)
    st = a.query()
    assert np.max(np.abs(np.asarray(st["q_meas"]) - np.asarray(target))) < 0.01   # sim followed the real joints
    assert st["mirror"] is True and st["mirror_age"] is not None and st["mirror_age"] < 0.5

    # body/voice motion is refused while mirroring: an AUTO command must not take over
    a.send({"kind": "utterance", "t": time.time(), "text": "pick up the cup", "offline": True})
    a.stream_real(target, 0.8)
    st = a.query()
    assert st["mode"] != "auto" and st["mirror"] is True
    assert np.max(np.abs(np.asarray(st["q_meas"]) - np.asarray(target))) < 0.02   # still mirroring, not grasping

    # STOP leaves the mirror
    a.send({"kind": "utterance", "t": time.time(), "text": "stop"})
    time.sleep(0.3)
    assert a.query()["mirror"] is False
    a.join()


def test_mirror_holds_when_real_samples_go_stale():
    a = _App(47197, seconds=8.0)
    a.send({"kind": "mirror", "t": time.time(), "on": True})
    target = (a.home + np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()
    a.stream_real(target, 2.0)
    q_at_pause = np.asarray(a.query()["q_meas"])
    time.sleep(1.2)                                   # no real samples for > 0.5 s -> sim holds
    st = a.query()
    assert st["mirror_age"] > 0.5
    assert np.max(np.abs(np.asarray(st["q_meas"]) - q_at_pause)) < 5e-3   # it did not drift while stale
    a.send({"kind": "utterance", "t": time.time(), "text": "stop"})
    a.join()


# ---- follower (scripts/sim_to_real.py) reads `mirror` from the app Query -------------------------
def _load_follower_with_stubbed_ros():
    """Import scripts/sim_to_real.py with rclpy/rover_msgs stubbed, so SimQuery (pure sockets) can be tested."""
    rclpy = types.ModuleType("rclpy")
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = type("Node", (), {"__init__": lambda self, *a, **k: None})
    rclpy.node = node_mod
    msg_mod = types.ModuleType("rover_msgs.msg")
    msg_mod.ArmCommand = type("ArmCommand", (), {})
    rover_msgs = types.ModuleType("rover_msgs"); rover_msgs.msg = msg_mod
    for name, mod in {"rclpy": rclpy, "rclpy.node": node_mod, "rover_msgs": rover_msgs,
                      "rover_msgs.msg": msg_mod}.items():
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("sim_to_real_mod", ROOT / "scripts/sim_to_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_follower_simquery_reads_mirror_flag():
    mod = _load_follower_with_stubbed_ros()
    # a fake app that answers a query with a state carrying mirror + q_meas
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0)); srv.settimeout(2)
    port = srv.getsockname()[1]
    reply = {"mirror": True}

    def serve():
        for _ in range(20):
            try:
                _, addr = srv.recvfrom(65536)
            except socket.timeout:
                return
            srv.sendto(json.dumps({"kind": "state", "q_meas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                   "hw_busy": 0.0, "mirror": reply["mirror"]}).encode(), addr)

    threading.Thread(target=serve, daemon=True).start()
    sim = mod.SimQuery(port=port)
    q = sim.joints()
    assert q == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert sim.mirror is True            # the follower sees mirror and (in its loop) holds at zero velocity
    reply["mirror"] = False
    sim.joints()
    assert sim.mirror is False

    # confirm the table came from stepper_map (single source of truth), not a stale inline copy
    from rover_intent.control.stepper_map import DEFAULT_TABLE
    assert mod.ZERO_RAD == list(DEFAULT_TABLE.zero_rad) and mod.FW_MAX_VEL == list(DEFAULT_TABLE.fw_max_vel)
