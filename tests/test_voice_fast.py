"""Fast local motion grammar (branch voice-fast): Cartesian jogs, joint jogs, "then" sequences and speed, all
parsed in the control loop with no cloud round trip and (on dev_arm_real) no spoken confirm.

Covers: parse_fast positives/negatives (numbers, word-numbers, units, "a bit"; complex sentences defer to Nemotron),
the distance/angle clamps and the speed-scaling bounds, the Cartesian-jog IK on the dev_arm_real model (TCP moves
~the commanded delta with the tool orientation held), and the app over UDP on the mock backend: poses and jogs move
with no confirm on dev_arm_real, joint jogs clamp, an unreachable jog is refused, a sequence runs step by step and
STOP cancels the rest, speed replies, STOP stays instant, and confirm.skills still gates pick/place.
"""
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from rover_intent.config import load_config
from rover_intent.control.kinematics import ArmModel, rot_err
from rover_intent.control.presets import RETURN, check_joint_goal
from rover_intent.planner.nemotron import quick_pose
from rover_intent.planner.fast_path import (MAX_ANGLE, MAX_DIST_M, make_cart_step, make_joint_step, parse_fast,
                                            scale_speed)

ROOT = Path(__file__).resolve().parents[1]
OBJ, PRE = ["cup", "bottle", "block"], ["ready", "base"]


# --- parse_fast: positives ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind,checks", [
    ("move left 5 cm", "cartjog", {"direction": "left", "dist_m": 0.05}),
    ("nudge up 3 centimetres", "cartjog", {"direction": "up", "dist_m": 0.03}),
    ("move forward ten cm", "cartjog", {"direction": "forward", "dist_m": 0.10}),
    ("go down 200 mm", "cartjog", {"direction": "down", "dist_m": 0.15}),      # clamped to 15 cm
    ("move left", "cartjog", {"direction": "left", "dist_m": 0.05}),           # unspecified = 5 cm
    ("left a bit", "cartjog", {"direction": "left", "dist_m": 0.03}),          # "a bit" = 3 cm
    ("turn the base left 30 degrees", "jointjog", {"joint": 1}),
    ("shoulder up", "jointjog", {"joint": 2}),
    ("elbow up 20 degrees", "jointjog", {"joint": 3}),
    ("wrist down 15 degrees", "jointjog", {"joint": 5}),
    ("twist the tool clockwise 45 degrees", "jointjog", {"joint": 6}),
    ("slower", "speed", {"mode": "slower"}),
    ("normal speed", "speed", {"mode": "normal"}),
])
def test_parse_fast_positives(text, kind, checks):
    r = parse_fast(text, OBJ, PRE)
    assert r is not None and len(r["steps"]) == 1
    step = r["steps"][0]
    assert step["kind"] == kind
    for k, v in checks.items():
        assert step[k] == pytest.approx(v) if isinstance(v, float) else step[k] == v


# --- fix 2: default joint when none is named (base for rotation/left-right, shoulder for up/down) ----------------
@pytest.mark.parametrize("text,joint,sign_pos", [
    ("Go right by 10 degrees", 1, False),     # right = -q1
    ("go left by 10 degrees", 1, True),       # left  = +q1
    ("rotate by 10 degrees", 1, True),        # no direction -> base, default positive (left/ccw)
    ("turn by 30 degrees", 1, True),
    ("spin 45 degrees", 1, True),
    ("up 10 degrees", 2, False),              # shoulder up = -q2
    ("down 15 degrees", 2, True),             # shoulder down = +q2
    ("rotate the elbow by 10 degrees", 3, True),   # joint named, no direction -> joint's positive (elbow up)
])
def test_default_joint_when_none_named(text, joint, sign_pos):
    r = parse_fast(text, OBJ, PRE)
    assert r is not None and len(r["steps"]) == 1
    step = r["steps"][0]
    assert step["kind"] == "jointjog" and step["joint"] == joint
    assert (step["angle_rad"] > 0) == sign_pos


# --- fix 4: US/UK spellings, plurals, number words up to ninety, halves, "rotate the <joint> ..." ----------------
@pytest.mark.parametrize("text,kind,checks", [
    ("Move left 5 centimeters.", "cartjog", {"direction": "left", "dist_m": 0.05}),
    ("Move down 10 centimeters.", "cartjog", {"direction": "down", "dist_m": 0.10}),
    ("Move up 15 centimeters.", "cartjog", {"direction": "up", "dist_m": 0.15}),
    ("go forward 1 meter", "cartjog", {"direction": "forward", "dist_m": 0.15}),      # 1 m clamps to 15 cm
    ("nudge up half a centimeter", "cartjog", {"direction": "up", "dist_m": 0.005}),
    ("move left one and a half cm", "cartjog", {"direction": "left", "dist_m": 0.015}),
    ("turn the base left forty five degrees", "jointjog", {"joint": 1}),
    ("Rotate the elbow by 10 degrees.", "jointjog", {"joint": 3}),
    ("could you please move left 5 cm now", "cartjog", {"direction": "left", "dist_m": 0.05}),
    ("shoulder up by thirty degrees", "jointjog", {"joint": 2}),
])
def test_grammar_spellings_and_fillers(text, kind, checks):
    r = parse_fast(text, OBJ, PRE)
    assert r is not None and len(r["steps"]) == 1
    step = r["steps"][0]
    assert step["kind"] == kind
    for k, v in checks.items():
        assert step[k] == pytest.approx(v) if isinstance(v, float) else step[k] == v


# --- tonight's real-mic utterances (2026-10-23), verbatim incl. punctuation/caps, must parse on the fast path ----
@pytest.mark.parametrize("text,kind", [
    ("Go right by 10 degrees", "jointjog"),
    ("rotate by 10 degrees", "jointjog"),
    ("Move left 5 centimeters.", "cartjog"),
    ("Move down 10 centimeters.", "cartjog"),
    ("Move up 15 centimeters.", "cartjog"),
    ("Rotate the elbow by 10 degrees.", "jointjog"),
])
def test_tonight_utterances_parse_on_fast_path(text, kind):
    r = parse_fast(text, OBJ, PRE)
    assert r is not None and len(r["steps"]) == 1 and r["steps"][0]["kind"] == kind


def test_parse_fast_pose_and_sequence():
    assert parse_fast("stow", OBJ, PRE)["steps"][0] == {"kind": "preset", "name": "base", "say": "Moving to base."}
    assert parse_fast("go back", OBJ, PRE)["steps"][0]["name"] == RETURN
    seq = parse_fast("move left 5 cm then up 5 cm then go to ready", OBJ, PRE)
    assert [s["kind"] for s in seq["steps"]] == ["cartjog", "cartjog", "preset"]
    assert seq["steps"][-1]["name"] == "ready"


@pytest.mark.parametrize("text", [
    "put the cup left of the bottle",     # a skill, not a jog (names an object)
    "pick up the block",
    "move the cup to the left",
    "left", "up",                          # a bare direction with no cue -> defer
    "elbow 20 degrees",                    # a joint jog with no direction and no rotate verb -> defer
    "move up 5 inches",                    # an unknown unit -> defer (cm/mm/m/deg only)
    "spin around",                         # no magnitude: left to Nemotron (base +360, clamped) -> defer
    "open the gripper", "follow me",
    "move left 5 cm then pick up the cup",  # a mixed sequence: the whole thing defers (never half-execute)
    "go to the ready pose when you are ready and then stow it and then wait",  # > 6 clause words etc.
])
def test_parse_fast_defers_complex(text):
    assert parse_fast(text, OBJ, PRE) is None


# --- whisper mishearings -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,want", [("store it", "base"), ("stove", "base"), ("stowe", "base"), ("stoe", "base"),
                                       ("reddy", "ready")])
def test_pose_mishearings_fold_to_base(text, want):
    q = quick_pose(text, OBJ, PRE)
    assert q is not None and q.preset == want
    assert parse_fast(text, OBJ, PRE)["steps"][0]["name"] == want


def test_store_with_object_still_defers():
    # "store the cup" names a scene object -> must go to Nemotron, NOT fold onto the base pose
    assert quick_pose("store the cup", OBJ, PRE) is None
    assert parse_fast("store the cup", OBJ, PRE) is None


@pytest.mark.parametrize("text,direction", [("move write 5 cm", "right"), ("nudge write 3 cm", "right"),
                                            ("move for word 10 cm", "forward"), ("go foward 5 cm", "forward")])
def test_motion_word_mishearings(text, direction):
    r = parse_fast(text, OBJ, PRE)
    assert r is not None and r["steps"][0]["kind"] == "cartjog" and r["steps"][0]["direction"] == direction


# --- clamps + speed bounds ---------------------------------------------------------------------------------------
def test_distance_and_angle_clamps():
    assert make_cart_step("up", 0.50)["dist_m"] == MAX_DIST_M           # 50 cm -> 15 cm
    assert make_cart_step("up", 0.03)["dist_m"] == pytest.approx(0.03)
    assert make_joint_step(3, +1, np.radians(90))["angle_rad"] == pytest.approx(MAX_ANGLE)   # 90 deg -> 45
    assert make_joint_step(3, -1, np.radians(20))["angle_rad"] == pytest.approx(-np.radians(20))


def test_speed_scale_bounds():
    f = 1.0
    f = scale_speed(f, "slower"); assert f == 0.5
    f = scale_speed(f, "slower"); assert f == 0.25
    f = scale_speed(f, "slower"); assert f == 0.25            # never below 0.25x
    f = scale_speed(f, "faster"); assert f == 0.5
    f = scale_speed(f, "faster"); assert f == 1.0
    f = scale_speed(f, "faster"); assert f == 1.0            # never above the profile (1.0x)
    assert scale_speed(0.3, "normal") == 1.0


# --- Cartesian-jog IK on the real-arm model ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_arm():
    cfg = load_config(arm="dev_arm_real")
    a = cfg["arm"]
    arm = ArmModel(str(ROOT / a["urdf"]), base=a["base_link"], tip=a["tip_link"],
                   tcp_offset=a["tcp_offset"], tool_axis=a.get("tool_axis"))
    arm.lower = np.array(a["joint_limits_lower"]); arm.upper = np.array(a["joint_limits_upper"])
    return cfg, arm, np.array(a["home"]), cfg["safety"]["workspace_min"], cfg["safety"]["workspace_max"]


@pytest.mark.parametrize("direction,axis,sign", [("left", 1, +1), ("up", 2, +1), ("forward", 0, +1), ("down", 2, -1)])
def test_cart_jog_ik_moves_delta_and_holds_orientation(real_arm, direction, axis, sign):
    cfg, arm, home, lo, hi = real_arm
    T0 = arm.fk(home); R0 = T0[:3, :3].copy()
    Td = T0.copy(); Td[axis, 3] += sign * 0.05
    q = arm.ik(home.copy(), Td, iters=200, tol=1e-4, orient="full")
    T = arm.fk(q)
    assert np.linalg.norm(T[:3, 3] - Td[:3, 3]) < 0.005                      # TCP reaches ~the 5 cm delta
    assert np.degrees(np.linalg.norm(rot_err(T[:3, :3], R0))) < 3.0          # orientation held within a few degrees
    assert np.all(q >= arm.lower - 1e-9) and np.all(q <= arm.upper + 1e-9)   # inside the joint limits


def test_check_joint_goal_refusals(real_arm):
    cfg, arm, home, lo, hi = real_arm
    bad = home.copy(); bad[0] = arm.upper[0] + 0.5            # past joint 1's upper limit
    goal, msg = check_joint_goal(bad, home, "idle", arm, lo, hi, label="bad")
    assert goal is None and "joint limits" in msg
    assert check_joint_goal(home, home, "following", arm, lo, hi)[0] is None   # clutch engaged refuses any goal


def test_back_jog_15cm_is_unreachable(real_arm):
    # "move back 15 cm" from home leaves the reachable shell: the IK can't get within 1 cm, so the app refuses it
    cfg, arm, home, lo, hi = real_arm
    T0 = arm.fk(home); Td = T0.copy(); Td[0, 3] -= 0.15
    q = arm.ik(home.copy(), Td, iters=200, tol=1e-4, orient="full")
    assert np.linalg.norm(arm.fk(q)[:3, 3] - Td[:3, 3]) > 0.01


# --- app over UDP (mock backend, dev_arm_real = no confirm) -------------------------------------------------------
def _app(cfg, port, seconds):
    cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"   # force the offline parser; the fast path is local
    from rover_intent import app
    th = threading.Thread(target=app.run, args=(cfg, "mock", seconds, False), daemon=True)
    th.start()
    time.sleep(0.4)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.3)
    return app, th, s


def _drain(s):
    s.settimeout(0.05)
    try:
        while True:
            s.recvfrom(65536)
    except (socket.timeout, BlockingIOError):
        pass
    s.settimeout(1.5)


def _recv(s, kind, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            m = json.loads(s.recvfrom(65536)[0])
        except (socket.timeout, BlockingIOError):
            continue
        if m.get("kind") == kind:
            return m
    return None


def _say(s, port, text, offline=True):
    _drain(s)
    s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": text, "offline": offline}).encode(),
             ("127.0.0.1", port))
    return _recv(s, "reply")


def _query(s, port):
    _drain(s)
    s.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), ("127.0.0.1", port))
    return _recv(s, "state")


def _wait_tcp(s, port, pred, tries=60):
    for _ in range(tries):
        st = _query(s, port)
        if st and pred(st):
            return st
        time.sleep(0.1)
    return None


def test_real_pose_and_cart_jog_no_confirm():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47231, 14.0)
    try:
        tcp0 = _query(s, 47231)["tcp"]
        # a named pose moves at once (no confirm) on dev_arm_real
        r = _say(s, 47231, "stow")
        assert r.get("confirm") is not True and r["ok"] is True
        # a Cartesian jog: instant reply (< 50 ms), no confirm, and the TCP moves left (+y)
        _say(s, 47231, "go to ready")
        _wait_tcp(s, 47231, lambda st: st["preset"] is None)   # let the pose settle
        tcp_r = _query(s, 47231)["tcp"]
        r = _say(s, 47231, "move left 5 cm")
        assert r.get("confirm") is not True and r["ok"] is True and "left" in r["say"].lower()
        assert r["latency_s"] < 0.05, r["latency_s"]
        st = _wait_tcp(s, 47231, lambda st: st["preset"] is None and st["tcp"][1] > tcp_r[1] + 0.03)
        assert st is not None, "the left jog never moved the TCP"
        assert st["tcp"][1] == pytest.approx(tcp_r[1] + 0.05, abs=0.012)
    finally:
        s.close()
        th.join(timeout=16)
    assert not th.is_alive()


def test_real_joint_jog_and_unreachable_refused():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47232, 18.0)
    try:
        q0 = _query(s, 47232)["q_meas"]
        # from home, an unreachable jog (back 15 cm leaves the reachable shell) is refused; nothing moves
        r = _say(s, 47232, "move back 15 cm")
        assert r["ok"] is False and ("reach" in r["say"].lower() or "box" in r["say"].lower())
        assert _query(s, 47232)["preset"] is None
        assert _query(s, 47232)["q_meas"] == pytest.approx(q0, abs=1e-3)   # still at home
        # a joint jog moves exactly joint 3 (the elbow)
        r = _say(s, 47232, "elbow up 20 degrees")
        assert r["ok"] is True and "elbow" in r["say"].lower()
        st = _wait_tcp(s, 47232, lambda st: st["preset"] is None and abs(st["q_meas"][2] - q0[2]) > 0.2)
        assert st is not None, "the elbow joint jog did not move joint 3"
        # collision guard: "turn the base left ..." is a J1 jog, NOT the 'base' pose (say says "left", not "to base")
        q1 = _query(s, 47232)["q_meas"]
        r = _say(s, 47232, "turn the base left 20 degrees")
        assert r["ok"] is True and "left" in r["say"].lower() and "to base" not in r["say"].lower()
        st = _wait_tcp(s, 47232, lambda st: st["preset"] is None and abs(st["q_meas"][0] - q1[0]) > 0.1)
        assert st is not None, "the base joint jog did not move joint 1"
    finally:
        s.close()
        th.join(timeout=20)
    assert not th.is_alive()


def test_real_sequence_runs_and_stop_cancels():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47233, 16.0)
    ready = [0.144, -1.131, -1.007, -0.003, 0.679, -2.957]   # dev_arm_real "ready" preset
    try:
        # a 2-step pose sequence runs step by step and ends at the LAST pose (ready)
        r = _say(s, 47233, "stow then go to ready")
        assert r["ok"] is True
        st = _wait_tcp(s, 47233, lambda st: st["preset"] is None
                       and max(abs(a - b) for a, b in zip(st["q_meas"], ready)) < 0.03)
        assert st is not None, "the pose sequence did not finish at the ready pose"
        # STOP cancels the rest of a running sequence: nothing is moving afterwards
        _say(s, 47233, "move left 5 cm then move left 5 cm then move left 5 cm")
        time.sleep(0.2)
        r = _say(s, 47233, "stop")
        assert r["mode"] == "hold"
        time.sleep(0.5)
        q_after = _query(s, 47233)
        assert q_after["preset"] is None                     # no further step started after stop
        time.sleep(0.3)
        assert _query(s, 47233)["q_meas"] == pytest.approx(q_after["q_meas"], abs=1e-3)   # and the arm is still
    finally:
        s.close()
        th.join(timeout=18)
    assert not th.is_alive()


def test_real_speed_and_motion_vs_skills_confirm():
    app, th, s = _app(load_config(arm="dev_arm_real"), 47234, 10.0)
    try:
        assert _say(s, 47234, "slower")["say"].lower().startswith("slow")
        assert _say(s, 47234, "faster")["say"].lower().startswith("fast")
        # motion moves with no confirm, but a skill (aims at sim object positions) still asks first on the real arm
        assert _say(s, 47234, "move left 3 cm").get("confirm") is not True
        r = _say(s, 47234, "pick up the cup")
        assert r["confirm"] is True and r["mode"] != "auto"
        # STOP is always instant
        assert _say(s, 47234, "stop")["mode"] == "hold"
    finally:
        s.close()
        th.join(timeout=12)
    assert not th.is_alive()
