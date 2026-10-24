"""Frozen-wrist teleop IK (issue 4): orient="frozen" is position-only DLS on joints 1-3, wrist (4-6) held.

Numbers reproduced from an offline analysis of the dev_arm_real profile (app-like 50 Hz
loop, horizontal circle around the home TCP), 'frozen' variant:

    r=5cm  frozen  wrist joints 4-6 span [0 0 0] deg   max tool tilt 7 deg   TCP err (with safety caps) 1.7/4.2 cm
    r=10cm frozen  wrist joints 4-6 span [0 0 0] deg   max tool tilt 12 deg  TCP err (with safety caps) 2.4/5.3 cm

So on a 5 cm circle the wrist does not move at all, and tool tilt stays ~7 deg; at 10 cm tilt reaches ~12 deg. full
and axis modes still move the wrist to track the pose (unchanged). The home wrist is near-singular (joint 5 ~ -28 deg).
"""
import numpy as np
from pathlib import Path

from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.control.safety import SafetyLayer

ROOT = Path(__file__).resolve().parents[1]


def _arm():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a["base_link"], tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a["tool_axis"])
    arm.lower = np.array(a["joint_limits_lower"], float)
    arm.upper = np.array(a["joint_limits_upper"], float)
    return cfg, a, arm


def _circle_run(orient, radius, limited=True):
    """Mirror wrist_dig.py: T_des position follows a horizontal circle around the home TCP, home orientation held."""
    cfg, a, arm = _arm()
    home = np.array(a["home"], float)
    ws_lo, ws_hi = np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"])
    s = SafetyLayer(arm.lower, arm.upper, a["max_vel"], a["max_acc"], ws_lo, ws_hi)
    dt, T_total, period = 0.02, 12.0, 4.0
    q = home.copy()
    T0 = arm.fk(home)
    c = T0[:3, 3].copy()
    z0 = T0[:3, :3] @ arm.tool_axis
    qs, tilt = [q.copy()], []
    for k in range(int(T_total / dt)):
        t = k * dt
        T_des = T0.copy()
        p = c + radius * np.array([np.cos(2 * np.pi * t / period) - 1, np.sin(2 * np.pi * t / period), 0])
        T_des[:3, 3] = np.clip(p, ws_lo, ws_hi)
        q_des = arm.ik_step(q, T_des, orient=orient)
        q = s.step(q, q_des, dt, 0.0) if limited else q_des
        qs.append(q.copy())
        Tq = arm.fk(q)
        tilt.append(np.degrees(np.arccos(np.clip(np.dot(Tq[:3, :3] @ arm.tool_axis, z0), -1, 1))))
    qs = np.array(qs)
    wrist_span = np.degrees(qs[:, 3:].max(0) - qs[:, 3:].min(0))
    return wrist_span, max(tilt)


def test_frozen_holds_the_wrist_on_a_5cm_circle():
    wrist_span, tilt = _circle_run("frozen", 0.05)
    assert np.allclose(wrist_span, 0.0, atol=1e-9)        # joints 4,5,6 do not move at all
    assert tilt < 8.0                                      # ~7 deg (wrist_dig)


def test_frozen_tool_tilt_under_12_deg_at_10cm():
    wrist_span, tilt = _circle_run("frozen", 0.10)
    assert np.allclose(wrist_span, 0.0, atol=1e-9)
    assert tilt <= 12.5                                    # ~12 deg (wrist_dig)


def test_full_and_axis_still_move_the_wrist_unchanged():
    # the new "frozen" branch must not change full/axis: both still spin the wrist to track the circle.
    for orient in ("full", "axis"):
        wrist_span, _ = _circle_run(orient, 0.05)
        assert wrist_span.max() > 5.0                      # wrist clearly moves in full/axis (wrist_dig: tens of deg)


def test_frozen_reaches_a_translated_target_with_position_only():
    cfg, a, arm = _arm()
    home = np.array(a["home"], float)
    T = arm.fk(home).copy()
    T[:3, 3] = T[:3, 3] + np.array([0.0, 0.03, -0.03])
    q = home.copy()
    for _ in range(400):
        q = arm.ik_step(q, T, orient="frozen")
    assert np.allclose(q[3:], home[3:], atol=1e-9)         # wrist never moved
    assert np.linalg.norm(arm.fk(q)[:3, 3] - T[:3, 3]) < 5e-3   # but the position was reached with joints 1-3
