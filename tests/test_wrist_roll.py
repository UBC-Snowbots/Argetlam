"""Optional WRIST-ROLL follow for PLANE body teleop (body/wrist_roll.py + plane.py + app + fast_path + nemotron).

Turning the CONTROL hand (pronation/supination) rotates the gripper about its own axis = joint 6 (the last joint).
Default OFF. Covered here:
  - palm_roll_deg: pure angle from synthetic 21-point hands (+-30/90 deg, wrap-around, missing / short / degenerate).
  - WristRoll: anchor (no jump), deadband, gain, clamp to joint-6 limits, sign flip, wrap-around, hold on a missing
    hand, disabled = hold.
  - PlaneMapper: enabled vs disabled, roll_target anchored at engage, the roll NEVER moves the TCP position, clamp.
  - the app's joint-6 override math: setting joint 6 after the frozen IK keeps the TCP the IK solved.
  - fast-path voice phrases on/off, and that a normal jog ("move left 5 cm" / "turn the base left 30 degrees" /
    "turn the wrist up 10 degrees") is NOT stolen.
  - the planner: the toggle is in the Intent schema + the Nemotron system prompt, and a mocked LLM maps to it.
  - config default OFF.
"""
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent import app
from rover_intent.body.plane import PlaneMapper
from rover_intent.body.wrist_roll import WristRoll, palm_roll_deg
from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel
from rover_intent.planner.fast_path import parse_fast
from rover_intent.planner.nemotron import SYSTEM, IntentParser
from rover_intent.types import Action, BodyPose, Intent, PlaneTune

from test_plane import _App, _arm, _pose, _skel

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/dev_arm/dev_arm_gripper.urdf")


# --- synthetic hands -------------------------------------------------------------------------------------------
def palm_hand(angle_deg, scale=0.12, cx=0.5, cy=0.5):
    """21-point hand whose palm across-vector (index MCP 5 -> pinky MCP 17) sits at `angle_deg` in the image plane.
    Wrist (0) and middle MCP (9) set the palm scale; the other points are filler (palm_roll_deg reads 0/5/9/17)."""
    P = [[cx, cy] for _ in range(21)]
    P[0] = [cx, cy]
    P[9] = [cx, cy - scale]
    a = math.radians(angle_deg)
    half = 0.25 * scale
    P[5] = [cx - half * math.cos(a), cy - half * math.sin(a)]
    P[17] = [cx + half * math.cos(a), cy + half * math.sin(a)]
    return P


# --- palm_roll_deg (pure) --------------------------------------------------------------------------------------
@pytest.mark.parametrize("ang", [0, 30, -30, 90, -90, 120, -150])
def test_palm_roll_matches_angle(ang):
    assert palm_roll_deg(palm_hand(ang)) == pytest.approx(ang, abs=0.5)


def test_palm_roll_missing_and_short_and_degenerate():
    assert palm_roll_deg(None) is None
    assert palm_roll_deg([[0.0, 0.0]] * 10) is None          # fewer than 21 points
    flat = [[0.5, 0.5] for _ in range(21)]                   # zero palm scale + zero across-vector
    assert palm_roll_deg(flat) is None
    edge = palm_hand(0); edge[5] = edge[17] = [0.5, 0.5]     # across-vector collapses (fingers edge-on)
    assert palm_roll_deg(edge) is None


def test_palm_roll_is_distance_invariant():
    # a larger hand (farther finger spread, same orientation) reads the same angle
    assert palm_roll_deg(palm_hand(45, scale=0.08)) == pytest.approx(palm_roll_deg(palm_hand(45, scale=0.30)), abs=0.5)


# --- WristRoll integrator --------------------------------------------------------------------------------------
LO, HI = -9.9127, -0.0873   # dev_arm_real joint-6 travel (rad); home / base = -3.142


def _feed(w, angle, n=80, t0=0.0, dt=0.033):
    p = w.target
    for i in range(n):
        p = w.update(palm_hand(angle), t0 + i * dt)
    return p, t0 + (n - 1) * dt


def _roll_to(w, angle, n=200, dt=0.033):
    """Anchor the hand at 0 (natural pose = the engage value), then roll to `angle` and hold until it settles."""
    t = 0.0
    for _ in range(6):
        w.update(palm_hand(0.0), t); t += dt
    p = w.target
    for _ in range(n):
        p = w.update(palm_hand(angle), t); t += dt
    return p, t


def test_engage_no_jump_and_hold_when_still():
    w = WristRoll(enabled=True, lo=LO, hi=HI)
    w.engage(-3.142)
    assert w.update(palm_hand(0), 0.0) == pytest.approx(-3.142)   # first sample = zero, target = the engage value
    p, _ = _feed(w, 0.0)                                          # no hand rotation -> stays at the engage value
    assert p == pytest.approx(-3.142, abs=1e-3)


def test_disabled_holds_engage_value():
    w = WristRoll(enabled=False, lo=LO, hi=HI)
    w.engage(-3.142)
    p, _ = _feed(w, 60.0)    # even a big hand rotation does nothing while disabled
    assert p == pytest.approx(-3.142)


def test_gain_and_deadband():
    # gain 1.0, deadband 10 deg: a +40 deg hand roll past the dead zone -> +(40-10)=+30 joint deg from the anchor
    w = WristRoll(enabled=True, gain=1.0, deadband_deg=10.0, lo=LO, hi=HI)
    w.engage(-3.142)
    p, _ = _roll_to(w, 40.0)
    assert math.degrees(p - (-3.142)) == pytest.approx(30.0, abs=1.5)
    # a small roll inside the dead zone -> no change
    w.engage(-3.142)
    p2, _ = _roll_to(w, 6.0)
    assert p2 == pytest.approx(-3.142, abs=1e-2)


def test_sign_flip_reverses_direction():
    wp = WristRoll(enabled=True, gain=1.0, deadband_deg=10.0, lo=LO, hi=HI, sign=1.0)
    wn = WristRoll(enabled=True, gain=1.0, deadband_deg=10.0, lo=LO, hi=HI, sign=-1.0)
    wp.engage(-3.142); wn.engage(-3.142)
    pp, _ = _roll_to(wp, 40.0)
    pn, _ = _roll_to(wn, 40.0)
    assert (pp - (-3.142)) == pytest.approx(-(pn - (-3.142)), abs=1e-3)


def test_clamp_to_joint_limits_with_margin():
    w = WristRoll(enabled=True, gain=5.0, deadband_deg=10.0, lo=LO, hi=HI, margin_deg=5.0)
    w.engage(-0.2)                       # near the UPPER limit
    p, _ = _roll_to(w, 90.0)             # a big positive roll would overshoot hi; must clamp inside hi-margin
    assert p <= HI - math.radians(5.0) + 1e-6
    assert p == pytest.approx(HI - math.radians(5.0), abs=1e-3)


def test_missing_hand_holds():
    w = WristRoll(enabled=True, gain=1.0, deadband_deg=10.0, lo=LO, hi=HI)
    w.engage(-3.142)
    _roll_to(w, 40.0)
    held = w.target
    assert math.degrees(held - (-3.142)) == pytest.approx(30.0, abs=1.5)   # it really rolled before the hand was lost
    for i in range(30):                  # hand lost (None) for a while -> target does not move
        assert w.update(None, 10.0 + i * 0.033) == pytest.approx(held)


def test_wraparound_is_continuous():
    # sweep the hand angle past the +-180 boundary; the unwrapped angle must track smoothly (no spike)
    w = WristRoll(enabled=True, gain=1.0, deadband_deg=0.0, lo=-20.0, hi=20.0, margin_deg=0.0)
    w.engage(0.0)
    t = 0.0
    prev = 0.0
    for ang in list(range(150, 210, 2)) + list(range(210, 150, -2)):   # 150 -> 209(=-151) -> back
        a = ((ang + 180) % 360) - 180
        p = w.update(palm_hand(a), t); t += 0.033
        assert abs(p - prev) < 0.2      # radians: no discontinuous jump across the wrap
        prev = p


def test_set_enabled_reanchors_without_jump():
    w = WristRoll(enabled=False, gain=1.0, deadband_deg=10.0, lo=LO, hi=HI)
    w.engage(-3.142)
    # turn it ON mid-pose at a rotated hand: re-anchor at the current joint 6 so the first target == that value
    w.set_enabled(True, -2.5)
    assert w.update(palm_hand(45.0), 0.0) == pytest.approx(-2.5)   # natural (current) pose -> no jump


# --- PlaneMapper integration ---------------------------------------------------------------------------------
def _wr_mapper(enabled=True, **wr):
    arm, home, cfg = _arm()
    pcfg = dict(cfg.get("plane") or {})
    wrcfg = {"enabled": enabled, "gain": 1.0, "deadband_deg": 10.0, "margin_deg": 5.0, "sign": 1.0}
    wrcfg.update(wr)
    m = PlaneMapper(arm, home, pcfg, cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"],
                    wrist_roll_cfg=wrcfg)
    return m, arm, home


def _pose_wr(t, angle=None, **kw):
    """A plane pose carrying the CONTROL hand's landmarks at `angle` (None = no control hand -> wrist roll holds)."""
    p = _pose(t, **kw)
    p.ctrl_hand2d = None if angle is None else palm_hand(angle)
    return p


def test_mapper_disabled_roll_target_none():
    m, arm, home = _wr_mapper(enabled=False)
    m.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    for i in range(40):
        m(_pose_wr(i * 0.033, angle=40.0))
    assert m.roll_target is None


def test_mapper_enabled_anchors_at_engage():
    m, arm, home = _wr_mapper(enabled=True)
    m.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    m(_pose_wr(0.0, angle=0.0))
    assert m.roll_target == pytest.approx(home[5], abs=1e-6)   # natural pose maps to the current joint 6


def test_roll_never_moves_the_tcp():
    """The mapper's TCP target `p` must be identical with the wrist roll on and off for the same body pose: joint 6
    is applied separately (by the app) and its axis runs through the TCP."""
    m_on, arm, home = _wr_mapper(enabled=True)
    m_off, _, _ = _wr_mapper(enabled=False)
    m_on.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    m_off.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    for i in range(60):
        pa = m_on(_pose_wr(i * 0.033, angle=min(90.0, i * 2.0), rwx=0.30, rwy=0.42))
        pb = m_off(_pose_wr(i * 0.033, angle=min(90.0, i * 2.0), rwx=0.30, rwy=0.42))
        assert np.allclose(pa, pb, atol=1e-9)
    assert m_on.roll_target is not None and m_off.roll_target is None


def _mapper_roll_to(m, angle, n=200):
    """engage anchor at 0, then roll the control hand to `angle`; return the settled roll_target."""
    t = 0.0
    for _ in range(6):
        m(_pose_wr(t, angle=0.0)); t += 0.033
    for _ in range(n):
        m(_pose_wr(t, angle=angle)); t += 0.033
    return m.roll_target


def test_mapper_clamps_roll_at_limits():
    m, arm, home = _wr_mapper(enabled=True, gain=6.0)
    m.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    rt = _mapper_roll_to(m, 90.0)                 # gain 6 x ~80 deg = way past the limit -> clamp
    assert m.wrist_roll.lo <= rt <= m.wrist_roll.hi
    assert rt == pytest.approx(m.wrist_roll.hi, abs=1e-3) or rt == pytest.approx(m.wrist_roll.lo, abs=1e-3)


def test_mapper_missing_control_hand_holds():
    m, arm, home = _wr_mapper(enabled=True)
    m.engage(arm.fk(home)[:3, 3], None, joint6=home[5])
    held = _mapper_roll_to(m, 40.0)
    assert math.degrees(held - home[5]) == pytest.approx(30.0, abs=2.0)   # it really rolled first
    for i in range(30):
        m(_pose_wr(8.0 + i * 0.033, angle=None))   # control hand lost -> hold
    assert m.roll_target == pytest.approx(held)


# --- the app's joint-6 override math (keeps the frozen-IK TCP) -------------------------------------------------
def test_joint6_override_after_frozen_ik_keeps_tcp():
    arm, home, cfg = _arm()
    q = home.copy()
    T_des = arm.fk(q); T_des[:3, 3] = arm.fk(q)[:3, 3] + np.array([0.03, 0.02, -0.01])
    q_des = arm.ik_step(q, T_des, orient="frozen")
    tcp_before = arm.fk(q_des)[:3, 3]
    q_roll = q_des.copy(); q_roll[5] = home[5] + 0.8      # the app overrides joint 6 with the roll target
    tcp_after = arm.fk(q_roll)[:3, 3]
    assert np.allclose(tcp_before, tcp_after, atol=1e-9)  # rolling joint 6 does not move the TCP the IK solved


# --- fast-path voice phrases ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text,op", [
    ("follow my wrist", "wrist_roll_on"),
    ("wrist roll on", "wrist_roll_on"),
    ("turn on wrist roll", "wrist_roll_on"),
    ("rotate the gripper with my wrist", "wrist_roll_on"),
    ("lock the wrist", "wrist_roll_off"),
    ("wrist roll off", "wrist_roll_off"),
    ("turn off wrist roll", "wrist_roll_off"),
    ("stop wrist roll", "wrist_roll_off"),
])
def test_fast_path_wrist_roll_phrases(text, op):
    fp = parse_fast(text, ["cup"], ["ready", "base"])
    assert fp is not None and len(fp["steps"]) == 1
    assert fp["steps"][0]["kind"] == "plane_tune" and fp["steps"][0]["op"] == op


@pytest.mark.parametrize("text,kind", [
    ("move left 5 cm", "cartjog"),
    ("turn the base left 30 degrees", "jointjog"),
    ("turn the wrist up 10 degrees", "jointjog"),
    ("rotate the tool clockwise 15 degrees", "jointjog"),
])
def test_fast_path_jogs_not_stolen(text, kind):
    fp = parse_fast(text, ["cup"], ["ready", "base"])
    assert fp is not None and fp["steps"][0]["kind"] == kind   # a real jog is never read as a wrist-roll toggle


# --- planner: schema + prompt + mocked LLM -------------------------------------------------------------------
def test_intent_schema_and_prompt_have_wrist_roll():
    schema = Intent.model_json_schema()
    assert "wrist_roll" in schema["properties"]
    assert "wrist_roll" in [a.value for a in Action]
    assert "wrist_roll" in SYSTEM   # the model is told about the toggle


class _WRComp:
    """A fake OpenAI chat.completions that returns a wrist_roll intent."""
    def create(self, **kw):
        content = '{"action":"wrist_roll","wrist_roll":"on","say":"Wrist roll on."}'
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]})()


def test_mocked_llm_maps_to_wrist_roll():
    os.environ["FAKE_WR_KEY"] = "x"
    p = IntentParser("m", "http://unused", api_key_env="FAKE_WR_KEY")
    p.client = type("Client", (), {"chat": type("Chat", (), {"completions": _WRComp()})()})()
    intent = p.parse("let me rotate the gripper with my wrist", ["cup"], None, ["ready", "base"])
    assert intent.action == Action.wrist_roll and intent.wrist_roll == "on"


# --- config default OFF --------------------------------------------------------------------------------------
def test_config_default_off():
    assert load_config()["body"].get("wrist_roll", False) is False
    assert load_config(arm="dev_arm_real")["body"].get("wrist_roll", False) is False
    cfg = load_config(arm="dev_arm_real")
    _arm_, _safety, _robot, _scene, _arb, retarget, _parser = app.build(cfg, "mock")
    assert retarget.wrist_roll.enabled is False


# --- app over UDP: the toggle flows (plane_tune key + voice) -------------------------------------------------
def test_app_wrist_roll_toggle_over_udp():
    a = _App(47280, seconds=9.0)
    assert a.query()["mapping"] == "PlaneMapper"
    assert a.query()["wrist_roll"]["enabled"] is False
    a.send({"kind": "plane_tune", "t": time.time(), "op": "wrist_roll"}); time.sleep(0.2); a.drain()
    assert a.query()["wrist_roll"]["enabled"] is True          # 'w' key / plane_tune toggle turned it on
    a.send({"kind": "utterance", "t": time.time(), "text": "lock the wrist"}); time.sleep(0.2); a.drain()
    assert a.query()["wrist_roll"]["enabled"] is False         # voice phrase turned it off
    a.send({"kind": "utterance", "t": time.time(), "text": "follow my wrist"}); time.sleep(0.2); a.drain()
    assert a.query()["wrist_roll"]["enabled"] is True          # voice phrase turned it on
    a.join()
