"""Real gripper (EE, firmware axis 7) limits + mapping (rover_intent.control.ee_limits), the follower's grip
plumbing, and the real->sim mirror grip display. Pure logic + stubbed-ROS + a fake app; never touches a real arm."""
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rover_intent.control import ee_limits as eel  # noqa: E402


# ---- mapping + soft limits -----------------------------------------------------------------------
def test_grip_to_fw_endpoints_and_midpoint():
    lim = eel.EELimits(open_fw=100.0, closed_fw=40.0, open_sign=-1, margin_fw=5.0)
    # grip 0 -> open end, grip 1 -> closed end, but each clamped margin short of the stop
    assert lim.grip_to_fw(0.0) == pytest.approx(95.0)    # open_fw 100, soft hi = 100 - 5
    assert lim.grip_to_fw(1.0) == pytest.approx(45.0)    # closed_fw 40, soft lo = 40 + 5
    assert lim.grip_to_fw(0.5) == pytest.approx(70.0)    # midpoint, inside soft range -> unclamped


def test_soft_limits_ordered_for_either_sign():
    a = eel.EELimits(open_fw=40.0, closed_fw=100.0, margin_fw=5.0)   # open < closed
    b = eel.EELimits(open_fw=100.0, closed_fw=40.0, margin_fw=5.0)   # open > closed
    assert a.soft_limits() == (45.0, 95.0)
    assert b.soft_limits() == (45.0, 95.0)                           # same interval, always lo <= hi


def test_clamp_keeps_targets_inside_soft_limits_both_signs():
    for lim in (eel.EELimits(open_fw=40.0, closed_fw=100.0, margin_fw=5.0),
                eel.EELimits(open_fw=100.0, closed_fw=40.0, margin_fw=5.0)):
        lo, hi = lim.soft_limits()
        for g in (-0.5, 0.0, 0.3, 0.7, 1.0, 1.5):
            assert lo - 1e-9 <= lim.grip_to_fw(g) <= hi + 1e-9


def test_margin_too_big_collapses_to_midpoint():
    lim = eel.EELimits(open_fw=50.0, closed_fw=60.0, margin_fw=20.0)   # 2*margin > span
    lo, hi = lim.soft_limits()
    assert lo == hi == pytest.approx(55.0)


def test_fw_to_grip_is_the_inverse_mirror():
    lim = eel.EELimits(open_fw=100.0, closed_fw=40.0, open_sign=-1)
    assert lim.fw_to_grip(100.0) == pytest.approx(0.0)    # at open -> grip 0
    assert lim.fw_to_grip(40.0) == pytest.approx(1.0)     # at closed -> grip 1
    assert lim.fw_to_grip(70.0) == pytest.approx(0.5)
    assert lim.fw_to_grip(130.0) == pytest.approx(0.0)    # past open -> clamped to 0
    assert lim.fw_to_grip(10.0) == pytest.approx(1.0)     # past closed -> clamped to 1


# ---- P-loop velocity + no-drive-past-soft-limits -------------------------------------------------
def test_ee_velocity_deadband_cap_and_direction():
    lim = eel.EELimits(open_fw=40.0, closed_fw=140.0, margin_fw=5.0)   # soft [45, 135]
    # grip 1 target = 135; from 90 the error is +45 -> kp*e huge -> capped at cap
    v = eel.ee_velocity(1.0, 90.0, lim, cap_deg_s=10.0, kp=3.0, deadband=0.5)
    assert v == pytest.approx(10.0)
    # inside the deadband -> zero
    assert eel.ee_velocity(0.5, 90.0, lim, cap_deg_s=10.0, kp=3.0, deadband=0.5) == 0.0   # target 90, cur 90
    # grip 0 target = 45; from 90 error is -45 -> negative velocity, capped
    assert eel.ee_velocity(0.0, 90.0, lim, cap_deg_s=10.0) == pytest.approx(-10.0)


def test_ee_velocity_cap_is_hard():
    lim = eel.EELimits(open_fw=0.0, closed_fw=200.0, margin_fw=5.0)
    v = eel.ee_velocity(1.0, 0.0, lim, cap_deg_s=1e6)          # asking for a huge cap
    assert abs(v) <= eel.EE_SPEED_CAP_DEG_S + 1e-9            # still clamped to the hard cap


def test_ee_velocity_always_drives_inward_never_past_the_stop():
    lim = eel.EELimits(open_fw=40.0, closed_fw=140.0, margin_fw=5.0)   # soft [45, 135]
    # the target is always clamped into [45, 135], so from OUTSIDE the range the velocity points back INWARD, and
    # it is never driven further out than the soft stop.
    assert eel.ee_velocity(1.0, 140.0, lim, cap_deg_s=10.0) < 0        # above hi, closing-grip -> inward (down to 135)
    assert eel.ee_velocity(0.0, 40.0, lim, cap_deg_s=10.0) > 0         # below lo, opening-grip -> inward (up to 45)
    # right at a soft stop with the matching extreme grip -> inside the deadband -> no further push
    assert eel.ee_velocity(1.0, 135.0, lim, cap_deg_s=10.0, deadband=0.5) == 0.0
    assert eel.ee_velocity(0.0, 45.0, lim, cap_deg_s=10.0, deadband=0.5) == 0.0


# ---- gating (end_effector stays 0 unless everything is present) ----------------------------------
def test_ee_command_gated_off_returns_zero():
    lim = eel.EELimits(open_fw=40.0, closed_fw=140.0)
    # gated OFF -> always 0, whatever the grip / feedback
    assert eel.ee_command(False, 1.0, 90.0, lim, 10.0) == 0.0
    # no limits -> 0
    assert eel.ee_command(True, 1.0, 90.0, None, 10.0) == 0.0
    # no EE feedback yet -> 0
    assert eel.ee_command(True, 1.0, None, lim, 10.0) == 0.0
    # everything present -> non-zero
    assert eel.ee_command(True, 1.0, 90.0, lim, 10.0) != 0.0


def test_ee_enabled_requires_both_file_and_flag(tmp_path):
    p = tmp_path / "ee.yaml"
    assert eel.ee_enabled(True, p) is False            # flag on but no file
    eel.save(eel.EELimits(open_fw=10.0, closed_fw=50.0), p)
    assert eel.ee_enabled(False, p) is False           # file but no flag and no config
    assert eel.ee_enabled(True, p) is True             # file + flag
    assert eel.ee_enabled(False, p, config_ee=True) is True   # file + config gate (real_arm.ee)


# ---- stale-reference refusal ---------------------------------------------------------------------
def test_reference_ok_window():
    lim = eel.EELimits(open_fw=100.0, closed_fw=40.0, margin_fw=5.0)   # range [40,100], window +-2*5
    assert lim.reference_ok(70.0) is True
    assert lim.reference_ok(30.5) is True              # within 10 below 40
    assert lim.reference_ok(109.5) is True             # within 10 above 100
    assert lim.reference_ok(29.0) is False             # more than 10 below -> stale reference
    assert lim.reference_ok(111.0) is False


# ---- re-reference shift (span kept) --------------------------------------------------------------
def test_shifted_to_open_keeps_span_and_sign():
    lim = eel.EELimits(open_fw=100.0, closed_fw=40.0, open_sign=-1, margin_fw=4.0, notes="n")
    shifted = lim.shifted_to_open(120.0, marked_at="later")
    assert shifted.open_fw == 120.0
    assert shifted.closed_fw == pytest.approx(60.0)    # span -60 kept (40-100)
    assert shifted.span == pytest.approx(lim.span)
    assert shifted.open_sign == -1 and shifted.margin_fw == 4.0 and shifted.notes == "n"
    assert shifted.marked_at == "later"


# ---- limits file round trip ----------------------------------------------------------------------
def test_limits_file_round_trip(tmp_path):
    p = tmp_path / "dev_arm_real_ee.yaml"
    lim = eel.EELimits(open_fw=123.456, closed_fw=45.6, open_sign=-1, margin_fw=5.0,
                       marked_at="2026-10-23T14:00:00-0700", notes="relative to this power-on; re-reference")
    eel.save(lim, p)
    back = eel.load(p)
    assert back.open_fw == pytest.approx(123.456)
    assert back.closed_fw == pytest.approx(45.6)
    assert back.open_sign == -1
    assert back.margin_fw == pytest.approx(5.0)
    assert back.marked_at == "2026-10-23T14:00:00-0700"
    assert "power-on" in back.notes
    # the saved file is also valid YAML (flat key: value), so load_config / yaml can read it
    import yaml
    d = yaml.safe_load(open(p))
    assert d["open_fw"] == pytest.approx(123.456) and d["open_sign"] == -1


# ---- follower (scripts/sim_to_real.py) reads the app grip, sends end_effector 0 when gated -------
def _load_follower_with_stubbed_ros():
    import types
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
    spec = importlib.util.spec_from_file_location("sim_to_real_ee_mod", ROOT / "scripts/sim_to_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_follower_simquery_reads_gripper():
    mod = _load_follower_with_stubbed_ros()
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0)); srv.settimeout(2)
    port = srv.getsockname()[1]
    state = {"grip": 0.8}

    def serve():
        for _ in range(20):
            try:
                _, addr = srv.recvfrom(65536)
            except socket.timeout:
                return
            srv.sendto(json.dumps({"kind": "state", "q_meas": [0.0] * 6, "hw_busy": 0.0, "mirror": False,
                                   "gripper": state["grip"]}).encode(), addr)

    threading.Thread(target=serve, daemon=True).start()
    sim = mod.SimQuery(port=port)
    sim.joints()
    assert sim.grip == pytest.approx(0.8)
    state["grip"] = 0.1
    sim.joints()
    assert sim.grip == pytest.approx(0.1)


# ---- real -> sim mirror shows the real EE as the sim gripper (app side) --------------------------
def test_mirror_shows_real_ee_as_sim_gripper():
    import test_mirror
    a = test_mirror._App(47198, seconds=8.0)
    a.send({"kind": "mirror", "t": time.time(), "on": True})
    time.sleep(0.2)
    assert a.query()["mirror"] is True
    from rover_intent.control.stepper_map import q_to_fw
    target = a.home.tolist()
    fw = q_to_fw(target) + [0.0]
    # stream real_joints carrying a mapped grip -> the app shows it as the sim gripper
    t_end = time.time() + 1.2
    while time.time() < t_end:
        a.send({"kind": "real_joints", "t": time.time(), "fw": fw, "q": list(target), "grip": 0.8})
        time.sleep(1 / 30.0)
    assert a.query()["gripper"] == pytest.approx(0.8, abs=1e-6)
    t_end = time.time() + 1.0
    while time.time() < t_end:
        a.send({"kind": "real_joints", "t": time.time(), "fw": fw, "q": list(target), "grip": 0.15})
        time.sleep(1 / 30.0)
    assert a.query()["gripper"] == pytest.approx(0.15, abs=1e-6)
    a.send({"kind": "utterance", "t": time.time(), "text": "stop"})
    a.join()
