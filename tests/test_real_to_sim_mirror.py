"""real_to_sim re-asserts 'mirror on' while running (not just once at start).

The app can silently drop its mirror flag on an app/stack restart; a single start-only "mirror on" then stops
mirroring with no error (2026-10-23). scripts/real_to_sim.py now re-sends mirror on every ~2 s via a timer. This
test imports that module with rclpy/rover_msgs stubbed (same approach as test_mirror.py for the follower), points a
RealToSim at a real local UDP socket acting as the app, and checks that repeated reassert_mirror() calls land as
mirror-on datagrams. Pure sockets, no ROS.
"""
import importlib.util
import json
import socket
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_real_to_sim_with_stubbed_ros():
    rclpy = types.ModuleType("rclpy")
    node_mod = types.ModuleType("rclpy.node")
    # Node stub: no-op __init__, and create_subscription is a no-op (real_to_sim calls it in __init__).
    node_mod.Node = type("Node", (), {
        "__init__": lambda self, *a, **k: None,
        "create_subscription": lambda self, *a, **k: None,
        "create_timer": lambda self, *a, **k: None,
        "get_logger": lambda self: types.SimpleNamespace(info=lambda *a, **k: None),
    })
    rclpy.node = node_mod
    msg_mod = types.ModuleType("rover_msgs.msg")
    msg_mod.ArmCommand = type("ArmCommand", (), {})
    rover_msgs = types.ModuleType("rover_msgs")
    rover_msgs.msg = msg_mod
    for name, mod in {"rclpy": rclpy, "rclpy.node": node_mod,
                      "rover_msgs": rover_msgs, "rover_msgs.msg": msg_mod}.items():
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("real_to_sim_mod", ROOT / "scripts/real_to_sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recv_mirror_msgs(srv, n, timeout=2.0):
    """Collect up to n mirror messages received on srv."""
    srv.settimeout(timeout)
    got = []
    for _ in range(n):
        try:
            data, _addr = srv.recvfrom(65536)
        except socket.timeout:
            break
        m = json.loads(data)
        if m.get("kind") == "mirror":
            got.append(m)
    return got


def test_reassert_sends_mirror_on_repeatedly():
    mod = _load_real_to_sim_with_stubbed_ros()
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    node = mod.RealToSim("127.0.0.1", port, hz=30.0, mirror=True, mirror_interval=0.01)
    # one start-send plus several timer re-asserts (what the rclpy timer does while running)
    node.send_mirror(True)
    for _ in range(4):
        node.reassert_mirror()

    msgs = _recv_mirror_msgs(srv, 5)
    srv.close()
    assert len(msgs) >= 3, f"expected repeated mirror-on datagrams, got {len(msgs)}"
    assert all(m["on"] is True for m in msgs)       # every re-assert is 'on'
    assert node.mirror_interval == 0.01


def test_no_mirror_reassert_is_silent():
    """With mirror disabled (--no-mirror) the re-assert must send nothing."""
    mod = _load_real_to_sim_with_stubbed_ros()
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    node = mod.RealToSim("127.0.0.1", port, hz=30.0, mirror=False)
    for _ in range(4):
        node.reassert_mirror()

    msgs = _recv_mirror_msgs(srv, 1, timeout=0.3)
    srv.close()
    assert msgs == []


def test_send_mirror_off_payload():
    mod = _load_real_to_sim_with_stubbed_ros()
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    node = mod.RealToSim("127.0.0.1", port, hz=30.0, mirror=True)
    node.send_mirror(False)
    msgs = _recv_mirror_msgs(srv, 1)
    srv.close()
    assert len(msgs) == 1 and msgs[0]["on"] is False and msgs[0]["kind"] == "mirror"
