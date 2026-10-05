from pathlib import Path

import numpy as np
import pytest

from rover_intent.control.arbiter import Arbiter, Mode
from rover_intent.control.kinematics import ArmModel, rot_err
from rover_intent.control.safety import SafetyLayer
from rover_intent.planner.nemotron import IntentParser, fallback_parse, urgent_stop
from rover_intent.planner.scene import Scene
from rover_intent.planner.skills import SkillError, plan
from rover_intent.types import Action, Intent

URDF = str(Path(__file__).resolve().parents[1] / "assets/rover2026/rover2026.urdf")
TOP_DOWN = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
SCENE = {"cup": {"pos": [0.55, 0.15, 0.10]}, "bottle": {"pos": [0.55, -0.15, 0.12], "height": 0.22}}


@pytest.fixture(scope="module")
def arm():
    return ArmModel(URDF, tcp_offset=(0.15, 0, 0))


def test_fk_matches_spec_sheet():
    # docs/rover2026_arm_specs.md: all-zero pose puts the EE holder at (-0.03, 0.15, 0.50)
    p = ArmModel(URDF).fk(np.zeros(6))[:3, 3]
    assert np.allclose(p, [-0.03, 0.15, 0.50], atol=0.005)


def test_jacobian_matches_finite_difference(arm):
    q = np.array([1.0, -1.2, 0.9, 0.3, 1.1, -0.4])
    J = arm.jacobian(q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        dp = (arm.fk(q + dq)[:3, 3] - arm.fk(q - dq)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-5)


@pytest.mark.parametrize("p", [(0.55, 0.15, 0.10), (0.7, 0.0, 0.06), (0.45, 0.0, 0.45)])
def test_ik_top_down_reaches_scene_points(arm, p):
    home = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    T = np.eye(4); T[:3, :3] = TOP_DOWN; T[:3, 3] = p
    q = arm.ik(home, T, iters=400)
    Tq = arm.fk(q)
    assert np.linalg.norm(Tq[:3, 3] - p) < 2e-3
    assert np.degrees(np.linalg.norm(rot_err(Tq[:3, :3], TOP_DOWN))) < 2
    assert np.all(q >= arm.lower) and np.all(q <= arm.upper)


def _safety(arm, **kw):
    return SafetyLayer(arm.lower, arm.upper, [0.8] * 3 + [1.5] * 3, [2.0] * 3 + [4.0] * 3,
                       [-0.2, -0.8, 0.05], [1.0, 0.8, 1.0], **kw)


def test_safety_respects_vel_and_acc(arm):
    s, dt = _safety(arm), 0.02
    q = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    prev_v = np.zeros(6)
    for _ in range(200):
        nq = s.step(q, q + 1.0, dt, input_age_s=0.0)  # far target: saturate
        v = (nq - q) / dt
        assert np.all(np.abs(v) <= np.array(s.max_vel) + 1e-9)
        assert np.all(np.abs(v - prev_v) <= np.array(s.max_acc) * dt + 1e-9)
        q, prev_v = nq, v


def test_safety_estop_and_watchdog(arm):
    s, dt = _safety(arm), 0.02
    q = np.array([3.98, -1.31, 0.86, 0.0, 2.12, 2.31])
    for _ in range(20):
        q = s.step(q, q + 0.5, dt, 0.0)
    # stale input -> decelerates to a stop instead of continuing
    for _ in range(100):
        q = s.step(q, q + 0.5, dt, input_age_s=1.0)
    assert np.allclose(s._vel, 0, atol=1e-6)
    s.estop("test")
    assert np.array_equal(s.step(q, q + 1, dt, 0.0), q)
    s.reset()
    assert not s.estopped


def test_workspace_clip(arm):
    assert np.allclose(_safety(arm).clip_workspace(np.array([2.0, 0, -1])), [1.0, 0, 0.05])


@pytest.mark.parametrize("text,action,obj,rel,ref", [
    ("move the cup beside the bottle", Action.move, "cup", "beside", "bottle"),
    ("put the cup next to the bottle", Action.move, "cup", "beside", "bottle"),
    ("pick up the cup", Action.pick, "cup", None, None),
    ("let go", Action.release, None, None, None),
    ("follow me", Action.follow, None, None, None),
])
def test_fallback_parser(text, action, obj, rel, ref):
    i = fallback_parse(text, ["cup", "bottle"])
    assert (i.action, i.object, i.relation, i.reference) == (action, obj, rel, ref)


@pytest.mark.parametrize("text", ["stop!", "whoa whoa hold on", "WAIT", "emergency", "freeze"])
def test_stop_words_never_reach_the_llm(text, monkeypatch):
    assert urgent_stop(text).action == Action.stop
    p = IntentParser("any", "http://unused")
    p.client = object()  # an 'online' parser whose client would crash if called
    assert p.parse(text, ["cup"]).action == Action.stop


@pytest.mark.parametrize("text", ["reset", "Restart the table please", "let's start over"])
def test_reset_is_local(text):
    p = IntentParser("any", "http://unused")
    p.client = object()
    assert p.parse(text, ["cup"]).action == Action.reset


def test_reset_clears_arbiter_and_pick_while_holding_is_refused():
    a = Arbiter(Scene(SCENE))
    a.held, a.gripper = "block", 1.0
    a.on_intent(Intent(action=Action.pick, object="cup"))
    assert a.mode == Mode.HOLD and "already holding the block" in a.last_say  # no mid-air drop
    a.on_intent(Intent(action=Action.reset))
    assert (a.mode, a.held, a.gripper) == (Mode.HOLD, None, 0.0)


def test_object_not_in_scene_never_moves():
    # Nemotron may name anything; the skill layer only moves for objects it can see
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.pick, object="sender", say="Picking up the sender."))
    assert a.mode == Mode.HOLD and "unknown object 'sender'" in a.last_say


def test_clarify_does_not_move():
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.clarify, say="Which one, the cup or the block?"))
    assert a.mode == Mode.HOLD and a.last_say == "Which one, the cup or the block?"


def test_fallback_word_boundaries():
    # 'block' must not be found inside 'blocked', nor 'cup' inside 'cupboard'
    assert fallback_parse("the cupboard is blocked", ["cup", "block"]).action == Action.unknown


def test_skills_move_plan():
    wps = plan(Intent(action=Action.move, object="cup", relation="beside", reference="bottle"),
               Scene(SCENE), held=None)
    assert [w.label for w in wps][:5] == ["above-object", "pre-grasp", "grasp-pos", "close", "lift"]
    # every move between objects happens at the transit height, above the tallest object
    for w in wps:
        if w.label in ("above-object", "lift", "above-goal", "retreat"):
            assert w.pos[2] >= 0.30
    assert wps[-2].label == "open" and wps[-2].gripper == 0
    # released beside the bottle at the cup's own grasp height, not the bottle's
    assert np.isclose(wps[-2].pos[2], SCENE["cup"]["pos"][2])
    assert np.allclose(wps[-2].pos[:2], np.array(SCENE["bottle"]["pos"][:2]) + [0, 0.12])


def test_arbiter_dwells_and_tracks_held_object():
    """Ideal tracking (the gripper is wherever the carrot is): checks dwell, held bookkeeping, and that the final
    descent onto the object is a straight vertical line at the slow approach speed."""
    from rover_intent.planner.skills import APPROACH_SPEED
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.pick, object="cup"))
    cup = np.array(SCENE["cup"]["pos"])
    tcp, t, dt, log = np.array([0.45, 0.0, 0.38]), 0.0, 0.02, []
    while a.held is None and t < 20:
        p, g = a.target(tcp, None, now=t)
        log.append((t, p.copy(), g, a._plan[0].label if a._plan else None))
        tcp, t = p, t + dt
    assert a.held == "cup", "never picked"
    descent = [(t, p) for t, p, g, lab in log if lab == "grasp-pos"]
    xy_dev = max(np.linalg.norm(p[:2] - cup[:2]) for _, p in descent)
    assert xy_dev < 1e-6                                          # straight down: no sideways swing
    speeds = [np.linalg.norm(p2 - p1) / (t2 - t1) for (t1, p1), (t2, p2) in zip(descent, descent[1:])]
    assert max(speeds) <= APPROACH_SPEED + 1e-6                   # slow final approach
    closes = [t for t, p, g, lab in log if lab == "close" and g == 1.0]
    assert closes and log[-1][0] - closes[0] >= 0.8 - 2 * dt      # waited for the fingers before counting as held


def test_arbiter_eeg_hysteresis_and_dwell():
    a = Arbiter(Scene(SCENE), eeg_hold_s=0.5)
    a.on_intent(Intent(action=Action.follow))
    a.on_eeg(0.9, now=0.0); a.on_eeg(0.9, now=0.3)
    assert a.gripper == 0.0          # not held long enough
    a.on_eeg(0.5, now=0.4)           # in the dead band: keeps the pending change alive
    a.on_eeg(0.9, now=0.6)
    assert a.gripper == 1.0
    a.on_eeg(0.1, now=1.0); a.on_eeg(0.9, now=1.2); a.on_eeg(0.1, now=1.3)
    assert a.gripper == 1.0          # flicker does not release


def test_arbiter_waypoint_timeout_holds():
    a = Arbiter(Scene(SCENE), waypoint_timeout=8.0)
    a.on_intent(Intent(action=Action.pick, object="cup"))
    far = np.array([0.0, 0.0, 1.0])  # e.g. blocked by an obstacle, never gets there
    assert a.target(far, None, now=0.0)[0] is not None
    assert a.target(far, None, now=7.9)[0] is not None
    p, _ = a.target(far, None, now=8.1)
    assert p is None and a.mode == Mode.HOLD and "Couldn't reach" in a.last_say


def test_arbiter_stop_beats_auto():
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.pick, object="cup"))
    assert a.mode == Mode.AUTO
    a.on_intent(Intent(action=Action.stop))
    assert a.mode == Mode.HOLD and a.target(np.zeros(3), np.ones(3))[0] is None


def test_arbiter_shared_autonomy_pulls_sideways_only_above_objects():
    a = Arbiter(Scene(SCENE), assist_radius=0.08, assist_gain=0.6)
    a.on_intent(Intent(action=Action.follow))
    cup = np.array(SCENE["cup"]["pos"])
    hand = cup + [0.04, 0.0, 0.06]                 # above the cup, 4 cm off sideways
    p, _ = a.target(np.zeros(3), hand)
    assert np.linalg.norm(p[:2] - cup[:2]) < 0.04 and p[2] == hand[2]   # lined up over it, height untouched
    low = cup + [0.04, 0.0, -0.05]                 # beside it, below the grasp point: never pulled into it
    assert np.allclose(a.target(np.zeros(3), low)[0], low)


def test_retarget_is_relative_after_engage_and_directions_are_right():
    import yaml
    from rover_intent.body.retarget import Retargeter
    from rover_intent.types import BodyPose
    from rover_intent.config import load_config
    cfg = load_config()["body"]
    r = Retargeter(cfg["axes"], scale=1.0, origin=[0.45, 0, 0.45], min_cutoff=1e6)  # huge cutoff = no smoothing
    r.engage([0.5, 0.1, 0.3])
    pose = lambda w, t: BodyPose(t=t, wrist=w, shoulder=(0, 0, 0))
    assert np.allclose(r(pose((0.0, 0.5, 0.0), 0.0)), [0.5, 0.1, 0.3])     # first sample = neutral: no jump
    assert np.allclose(r(pose((0.0, 0.4, -0.1), 0.1)), [0.6, 0.1, 0.4])    # hand 10 cm up + toward the camera
    # user's RIGHT = image LEFT (x decreases) in a raw webcam frame -> gripper to the right in the over-shoulder view (-y)
    assert r(pose((-0.1, 0.5, 0.0), 0.2))[1] < 0.1


def test_one_euro_smooths_jitter_but_follows_fast_moves():
    from rover_intent.body.retarget import OneEuro
    f, rng = OneEuro(1.0, 0.7), np.random.default_rng(0)
    still = [f(np.array([0.0]) + rng.normal(0, 0.01, 1), i / 30)[0] for i in range(60)]
    assert np.std(still[30:]) < 0.004                      # 1 cm jitter -> < 4 mm
    f.reset()
    ramp = [f(np.array([i / 30 * 1.0]), i / 30)[0] for i in range(30)]  # 1 m/s move
    assert 1.0 * 29 / 30 - ramp[-1] < 0.06                  # lag under 6 cm at 1 m/s


def test_hand_not_found_is_not_open():
    from rover_intent.types import BodyPose
    assert BodyPose(t=0, wrist=(0, 0, 0), shoulder=(0, 0, 0)).hand_open is None


def test_hand_grip_only_in_follow_with_hysteresis_and_dwell():
    from rover_intent.control.arbiter import HAND_HOLD_S
    a = Arbiter(Scene(SCENE))
    far = np.array([0.40, 0.0, 0.30])           # nothing within grab range: plain close/open
    a.on_hand(0.1, 0.0, far); a.on_hand(0.1, 1.0, far)
    assert a.gripper == 0.0                     # not following: ignored
    a.on_intent(Intent(action=Action.follow))
    a.on_hand(0.1, 0.0, far); a.on_hand(0.1, HAND_HOLD_S - 0.05, far)
    assert a.gripper == 0.0                     # fist not held long enough
    a.on_hand(0.1, HAND_HOLD_S + 0.01, far)
    assert a.gripper == 1.0 and a.mode == Mode.FOLLOW and "nothing in reach" in a.last_say
    for i, h in enumerate([0.8, 0.1, 0.8, 0.5, 0.8]):   # flicker shorter than the dwell never opens it
        a.on_hand(h, 1.0 + 0.1 * i, far)
    assert a.gripper == 1.0
    a.on_hand(0.8, 2.0, far); a.on_hand(0.5, 2.2, far); a.on_hand(0.8, 2.0 + HAND_HOLD_S + 0.01, far)
    assert a.gripper == 0.0                     # dead band keeps the pending open alive; nothing held -> just opens


def _run_assist(a, tcp, t, dt=0.02, limit=30.0):
    """Ideal tracking: the gripper is wherever the carrot is. Runs until the arbiter is back in FOLLOW."""
    log = []
    while a.mode == Mode.AUTO and t < limit:
        p, g = a.target(tcp, None, now=t)
        if p is not None:
            log.append((t, p.copy(), g, a._plan[0].label if a._plan else None))
            tcp = p
        t += dt
    a.target(tcp, None, now=t)
    return tcp, t, log


def test_grab_assist_snaps_to_the_object_and_hands_back():
    from rover_intent.control.arbiter import ASSIST_LIFT, HAND_HOLD_S
    from rover_intent.planner.skills import APPROACH_SPEED
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    cup = np.array(SCENE["cup"]["pos"])
    tcp = cup + [0.04, -0.03, 0.06]             # roughly above the cup: 5 cm off sideways, 6 cm high
    assert a.grab_candidate(tcp).name == "cup"
    assert a.grab_candidate(cup + [0.09, 0, 0.06]) is None   # too far sideways
    assert a.grab_candidate(cup + [0, 0, 0.25]) is None      # too high above it
    a.on_hand(0.1, 0.0, tcp); a.on_hand(0.1, HAND_HOLD_S + 0.01, tcp)
    assert a.mode == Mode.AUTO and a.assist and "cup" in a.last_say
    tcp, t, log = _run_assist(a, tcp, 1.0)
    assert a.mode == Mode.FOLLOW and not a.assist and a.held == "cup" and a.gripper == 1.0
    assert a.events == ["assist_done"]
    assert np.allclose(tcp, cup + [0, 0, ASSIST_LIFT], atol=1e-6)
    pre = [p for _, p, _, lab in log if lab == "assist-rise"]
    assert all(np.allclose(p[:2], [cup[0] + 0.04, cup[1] - 0.03]) for p in pre)   # rises straight up first
    descent = [(t, p) for t, p, _, lab in log if lab == "grasp-pos"]
    assert max(np.linalg.norm(p[:2] - cup[:2]) for _, p in descent) < 1e-6        # straight down onto it
    speeds = [np.linalg.norm(p2 - p1) / (t2 - t1) for (t1, p1), (t2, p2) in zip(descent, descent[1:])]
    assert max(speeds) <= APPROACH_SPEED + 1e-6
    # body targets drive again after hand-back; holding = no shared-autonomy pull
    hand = cup + [0.0, 0.05, 0.10]
    assert np.allclose(a.target(tcp, hand, now=t + 1)[0], hand)


def test_grab_assist_release_opens_and_backs_off():
    from rover_intent.control.arbiter import HAND_HOLD_S, RELEASE_RETREAT
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    a.gripper, a.held = 1.0, "cup"
    tcp = np.array([0.50, 0.0, 0.20])
    a.on_hand(0.9, 0.0, tcp); a.on_hand(0.9, HAND_HOLD_S + 0.01, tcp)
    assert a.mode == Mode.AUTO and a.assist
    tcp, t, log = _run_assist(a, tcp, 1.0)
    assert a.mode == Mode.FOLLOW and a.held is None and a.gripper == 0.0
    assert np.allclose(tcp, [0.50, 0.0, 0.20 + RELEASE_RETREAT], atol=0.01)   # waypoint tolerance
    opens = [t for t, _, g, lab in log if lab == "open" and g == 0.0]
    retreat = [t for t, _, _, lab in log if lab == "retreat"]
    assert retreat[0] - opens[0] >= 0.8 - 0.05   # fingers open before backing off
    a.z_max = 0.21                               # retreat never targets above the workspace box
    a.gripper, a.held = 1.0, "cup"
    a.on_hand(0.9, 10.0, tcp); a.on_hand(0.9, 10.0 + HAND_HOLD_S + 0.01, tcp)
    assert a._plan[-1].pos[2] <= 0.21


def test_grab_assist_aborts_on_clutch_release():
    from rover_intent.control.arbiter import HAND_HOLD_S
    a = Arbiter(Scene(SCENE))
    a.on_intent(Intent(action=Action.follow))
    tcp = np.array(SCENE["cup"]["pos"]) + [0, 0, 0.08]
    a.on_hand(0.1, 0.0, tcp); a.on_hand(0.1, HAND_HOLD_S + 0.01, tcp)
    assert a.mode == Mode.AUTO
    a.abort()
    assert a.mode == Mode.HOLD and not a.assist and a.target(tcp, None, now=1.0)[0] is None


def test_app_and_laptop_scripts_compile():
    import py_compile
    root = Path(__file__).resolve().parents[1]
    import rover_intent.app  # noqa: F401
    for f in ["laptop/voice_client.py", "laptop/body_client.py", "sim/isaac_bridge.py", "src/rover_intent/eeg/live.py"]:
        py_compile.compile(str(root / f), doraise=True)


def _synthetic_arm(lateral=0.0, up=0.0, flex_deg=60.0, t=0.0):
    """Right arm of a person facing the camera, in MediaPipe world coords (x image-right, y down, z toward camera
    smaller). Their fwd = -z, left = +x, up = -y. Wrist placed via shoulder->elbow->wrist geometry."""
    from rover_intent.types import BodyPose
    fwd, left, upv = np.array([0, 0, -1.]), np.array([1., 0, 0]), np.array([0, -1., 0])
    S, O, H = -0.18 * left, 0.18 * left, -0.5 * upv
    u = _n(-0.5 * upv + 0.8 * fwd + lateral * left + up * upv)
    E = S + 0.30 * u
    side = _n(np.cross(u, upv)); perp = np.cross(side, u)
    f = np.cos(np.radians(flex_deg)) * u + np.sin(np.radians(flex_deg)) * perp
    return BodyPose(t=t, wrist=tuple(E + 0.27 * f), shoulder=tuple(S), elbow=tuple(E), shoulder_other=tuple(O),
                    hips=tuple(H))


def _n(v):
    return v / np.linalg.norm(v)


def test_gesture_mirror_directions():
    from rover_intent.body.mirror import GestureMirror
    arm = ArmModel(URDF, tcp_offset=(0.145, 0, 0))
    home_tcp = np.array([0.45, 0.0, 0.45])
    def run(start=home_tcp, **kw):
        m = GestureMirror(arm, min_cutoff=1e6)
        m.engage(start)
        first = m(_synthetic_arm(t=0.0))
        return first, m(_synthetic_arm(t=0.1, **kw)), m
    first, _, _ = run()
    assert np.allclose(first, home_tcp, atol=1e-6)                     # engage: no jump
    _, p, _ = run(lateral=0.3)                                          # swing hand to your LEFT
    assert p[1] > home_tcp[1] + 0.03                                    # -> gripper moves to robot left (+y)
    _, p, _ = run(up=0.4)                                               # raise hand
    assert p[2] > home_tcp[2] + 0.03                                    # -> gripper up
    _, p_ext, _ = run(flex_deg=20)                                      # straighten elbow (reach out)
    axis = GestureMirror(arm).axis_xy
    assert np.hypot(*(p_ext[:2] - axis)) > np.hypot(*(home_tcp[:2] - axis)) + 0.02   # -> reaches farther
    _, p, m = run(start=np.array([0.45, 0.0, 0.15]), up=-3.0)           # low gripper, hand dropped ~0.34 m
    assert m.clipped and abs(p[2] - 0.03) < 1e-9                        # clamped at the floor, flagged


@pytest.mark.parametrize("mode", ["gesture", "cartesian"])
def test_app_loop_end_to_end_mock(mode):
    """Runs the real control loop (mock arm) and drives it over UDP: reset, follow, gestures, stop."""
    import json, socket, threading, time, yaml
    from rover_intent import app
    from rover_intent.config import load_config
    cfg = load_config()
    port = 47190 + (mode == "cartesian")
    cfg["net"]["listen_port"], cfg["body"]["mode"] = port, mode
    cfg["net"]["hud_port"] = port + 50
    cfg["body"]["clutch"] = {"calib_s": 0.5, "still_m": 0.03, "palm_s": 1.0, "raise_m": 0.05}
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"   # offline parser, no network
    th = threading.Thread(target=app.run, args=(cfg, "mock", 4.0), daemon=True)
    th.start(); time.sleep(0.5)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3)
    def say(t):
        s.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": t}).encode(), ("127.0.0.1", port))
        return json.loads(s.recvfrom(9999)[0])
    assert say("reset")["mode"] == "hold"
    assert say("follow me")["mode"] == "hold"          # clutch starts calibrating: the arm holds
    statuses = []
    s.settimeout(0.001)  # non-blocking-ish drain while streaming poses
    for i in range(75):  # 0.8 s holding still (calibration), then 1.7 s swinging the hand to the left
        k = max(0, i - 24)
        p = _synthetic_arm(lateral=min(0.25, k * 0.01), t=time.time()).model_dump()
        p.update(kind="body", visible=True)
        if mode == "cartesian":
            p["wrist"] = [p["wrist"][0] + min(0.25, k * 0.01), *p["wrist"][1:]]
        s.sendto(json.dumps(p).encode(), ("127.0.0.1", port)); time.sleep(1 / 30)
        try:
            while True:
                m = json.loads(s.recvfrom(9999)[0])
                if m.get("kind") == "status":
                    statuses.append(m)
        except (socket.timeout, BlockingIOError):
            pass
    s.settimeout(3)
    assert any(m.get("clutch") == "calibrating" for m in statuses)
    assert statuses[-1]["clutch"] == "following"
    tcps = [m["tcp"] for m in statuses if m["mode"] == "follow"]
    assert tcps[-1][1] > tcps[0][1] + 0.02   # hand to your left -> gripper moved to robot left (+y)
    assert statuses and statuses[-1]["mode"] == "follow" and statuses[-1]["target"] is not None
    assert say("reset")["mode"] == "hold"
    th.join(timeout=6)
    assert not th.is_alive()


@pytest.mark.parametrize("profile", ["rover2026", "dev_arm"])
def test_scene_is_reachable(profile):
    """Every waypoint of every pick and every 'move X beside Y' in the arm profile's scene is reachable with the home
    orientation (top-down), IK from home like the live loop; objects stand on the table; the palm clears their tops."""
    import itertools
    from rover_intent.config import load_config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(arm=profile)
    a = cfg["arm"]
    arm = ArmModel(str(root / a["urdf"]), base=a["base_link"], tip=a["tip_link"], tcp_offset=a["tcp_offset"])
    home = np.array(a["home"])
    R_home = arm.fk(home)[:3, :3]
    assert np.allclose(R_home @ np.asarray(a["tool_axis"], float) / np.linalg.norm(a["tool_axis"]), [0, 0, -1],
                       atol=0.02), "home must point the tool straight down"
    table = cfg["table"]["top"]
    scene = Scene(cfg["scene"], support_z=table)
    lo, hi = np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"])
    assert np.all(arm.fk(home)[:3, 3] >= lo) and np.all(arm.fk(home)[:3, 3] <= hi), "home inside the box"
    for o in scene.objects.values():
        assert abs(o.pos[2] - o.grasp_z - table) < 1e-9                      # stands on the table
        assert scene.transit_z > o.pos[2] - o.grasp_z + o.height + 0.05      # travel clears the top
        assert o.grasp_z + a["palm_above_tcp"] >= o.height + 0.005, (o.name, o.grasp_z, o.height)
        assert o.grasp_z >= 0.02, o.name
    def reachable(p):
        T = np.eye(4); T[:3, :3] = R_home; T[:3, 3] = p
        q = arm.ik(home, T, iters=300); Tq = arm.fk(q)
        return np.linalg.norm(Tq[:3, 3] - p) < 5e-3 and np.degrees(np.linalg.norm(rot_err(Tq[:3, :3], R_home))) < 3
    intents = [Intent(action=Action.pick, object=n) for n in scene.names()]
    intents += [Intent(action=Action.move, object=x, relation="beside", reference=y)
                for x, y in itertools.permutations(scene.names(), 2)]
    bad = []
    for it in intents:
        for w in plan(it, scene, held=None):
            if not reachable(w.pos) or np.any(w.pos < lo - 1e-6) or np.any(w.pos > hi + 1e-6):
                bad.append((it.action.value, it.object, it.reference, w.label, np.round(w.pos, 3).tolist()))
    assert not bad, bad

def test_clutch_toggle_calibrates_then_follows_and_restarts_if_moved():
    from rover_intent.body.clutch import Clutch
    c = Clutch(calib_s=2.0)
    rel = np.array([0.1, 0.4, -0.1])
    assert c.toggle(0.0).kind == "calibrating"
    for t in np.arange(0.1, 1.0, 0.1):
        assert c.update(rel, False, False, t) is None
    c.update(rel + [0.1, 0, 0], False, False, 1.0)            # moved 10 cm: countdown restarts at t=1.0
    ev = None
    for t in np.arange(1.1, 3.05, 0.1):
        ev = c.update(rel + [0.1, 0, 0], False, False, t) or ev
    assert ev.kind == "engaged" and c.state == "following" and np.allclose(ev.neutral, rel + [0.1, 0, 0])
    assert c.toggle(3.1).kind == "released" and c.state == "idle"


def test_clutch_palm_needs_one_second_and_left_hand_is_dead_man():
    from rover_intent.body.clutch import Clutch
    c, rel = Clutch(calib_s=0.5), np.zeros(3)
    assert c.update(rel, True, False, 0.0) is None and c.update(rel, True, False, 0.5) is None
    assert c.update(rel, False, False, 0.6) is None               # palm dropped: timer resets
    assert c.update(rel, True, False, 0.7) is None
    assert c.update(rel, True, False, 1.75).kind == "calibrating"  # 1 s of open palm held up
    c = Clutch(calib_s=0.5)
    assert c.update(rel, False, True, 0.0) is None                # left hand up: debounced
    assert c.update(rel, False, True, 0.35).kind == "calibrating"
    for t, up in [(0.5, False), (0.53, True), (0.7, False), (0.75, True)]:   # flicker must NOT release
        assert c.update(rel, False, up, t) is None or c.state != "idle"
    assert c.update(rel, False, True, 0.9).kind == "engaged"
    assert c.update(rel, False, False, 1.0) is None               # lowered, not yet 0.5 s
    assert c.update(rel, False, False, 1.55).kind == "released"   # left hand down 0.5 s = freeze


def test_clutch_auto_releases_on_link_or_tracking_loss():
    from rover_intent.body.clutch import Clutch
    c, rel = Clutch(calib_s=0.2), np.zeros(3)
    c.toggle(0.0); c.update(rel, False, False, 0.1); c.update(rel, False, False, 0.35)
    assert c.state == "following"
    assert c.watchdog(0.8, last_packet_t=0.4, last_visible_t=0.4) is None        # 0.4 s gap: fine
    ev = c.watchdog(1.0, last_packet_t=0.4, last_visible_t=0.4)                     # 0.6 s without packets
    assert ev.kind == "released" and c.release_reason == "link lost" and c.state == "idle"
    c.toggle(2.0); c.update(rel, False, False, 2.1); c.update(rel, False, False, 2.35)
    ev = c.watchdog(3.5, last_packet_t=3.45, last_visible_t=2.4)                    # packets, but arm not visible
    assert ev.kind == "released" and c.release_reason == "tracking lost"
    c.toggle(10.0)                                                                   # voice "follow me", no pose yet
    assert c.watchdog(10.3, last_packet_t=0.0, last_visible_t=0.0) is None           # grace from clutch start


def _person(hand_body, yaw_deg=0.0, mirrored=False, noise=0.0, rng=None):
    """A person facing the camera, rotated `yaw_deg` about vertical, hand at `hand_body` (body frame: right, fwd, up
    from the RIGHT shoulder). Returns a BodyPose in MediaPipe world coords (x image-right, y down, z away from the
    camera = larger). mirrored = the webcam flips the image (MediaPipe then swaps left/right labels)."""
    from rover_intent.types import BodyPose
    rng = rng or np.random.default_rng(0)
    # person's axes in camera coords: facing the camera -> their forward = -z (toward camera), their right = image -x
    yaw = np.radians(yaw_deg)
    right = np.array([-np.cos(yaw), 0, np.sin(yaw)]); fwd = np.array([-np.sin(yaw), 0, -np.cos(yaw)])
    up = np.array([0.0, -1.0, 0.0])
    rs, ls, hips = 0.18 * right, -0.18 * right, -0.5 * up
    wr = rs + hand_body[0] * right + hand_body[1] * fwd + hand_body[2] * up + rng.normal(0, noise, 3)
    if mirrored:  # image flipped: x negated, and the labels swap (their right arm is reported as "left")
        f = lambda v: np.array([-v[0], v[1], v[2]])
        rs_m, ls_m, wr_m = f(ls), f(rs), f(wr)
        # the tracked "right" arm is their LEFT arm: put that wrist where their left hand mirrors the motion
        return BodyPose(t=0, wrist=tuple(f(rs + (wr - rs))), shoulder=tuple(f(rs)), shoulder_other=tuple(f(ls)),
                        hips=tuple(f(hips)), elbow=tuple(f(rs)))
    return BodyPose(t=0, wrist=tuple(wr), shoulder=tuple(rs), shoulder_other=tuple(ls), hips=tuple(hips),
                    elbow=tuple(rs))


def test_torso_frame_is_camera_independent():
    from rover_intent.body.calibration import hand_in_body
    h = np.array([0.10, 0.25, -0.05])
    for yaw in (0, 25, -40):
        assert np.allclose(hand_in_body(_person(h, yaw)), h, atol=1e-9)


def test_calibration_fit_and_mapping_follow_the_camera():
    from rover_intent.body.calibration import Capture, command_frame, fit, hand_in_body
    lo, hi = np.array([0.40, -0.40, 0.03]), np.array([0.62, 0.40, 0.32])
    rng = np.random.default_rng(1)
    neutral = np.array([0.05, 0.30, -0.10])
    reach = {"right": [0.25, 0, 0], "up": [0, 0, 0.22], "forward": [0, 0.20, 0], "left": [-0.18, 0, 0],
             "down": [0, 0, -0.20]}
    cap, t = Capture(), 0.0
    for step in cap.steps:
        target = neutral + (np.array(reach[step]) if step != "neutral" else 0)
        for _ in range(int(2.5 * 30)):               # 2.5 s of holding at 30 Hz, 3 mm noise
            ev = cap.update(t, hand_in_body(_person(target, yaw_deg=20, noise=0.003, rng=rng)))
            t += 1 / 30
    assert cap.done and set(cap.holds) == set(cap.steps)
    for cam, want_right in [(((-0.5, 0.0, 0.8), (0.5, 0.0, 0.1)), np.array([0, -1, 0])),      # behind the arm
                            (((0.5, -1.0, 0.8), (0.5, 0.0, 0.1)), np.array([1, 0, 0]))]:     # from the side
        C = command_frame(*cam)
        assert np.allclose(C[:, 0], want_right, atol=1e-9)                                     # screen-right
        prof = fit(cap.holds, C, lo, hi)
        assert prof.residual_deg < 2.0, prof.per_dir_deg
        centre, _ = prof.map(hand_in_body(_person(neutral, 20)), C, lo, hi)
        assert np.allclose(centre, (lo + hi) / 2, atol=0.01)                                  # neutral = box centre
        p, _ = prof.map(hand_in_body(_person(neutral + [0.12, 0, 0], 20)), C, lo, hi)          # hand to YOUR right
        assert np.dot(p - (lo + hi) / 2, want_right) > 0.05                                    # -> screen-right
        p, _ = prof.map(hand_in_body(_person(neutral + [0, 0, 0.1], 20)), C, lo, hi)
        assert p[2] > (lo[2] + hi[2]) / 2 + 0.03                                               # up -> up
        p, clamped = prof.map(hand_in_body(_person(neutral + [0.6, 0, 0], 20)), C, lo, hi)
        assert clamped and np.all(p <= hi + 1e-9) and np.all(p >= lo - 1e-9)                   # far -> clamped


def test_capture_rejects_jitter_and_waits_for_real_motion():
    from rover_intent.body.calibration import Capture
    cap, t = Capture(steps=["neutral", "right"]), 0.0
    n = np.array([0.0, 0.3, 0.0])
    rejected = 0
    for i in range(110):                             # 3.7 s: shaky for the first second (4 cm swings), then still
        wob = 0.04 * np.sin(i) if i < 30 else 0.0
        rejected += cap.update(t, n + [wob, 0, 0]) == "rejected:jitter"; t += 1 / 30
        if i == 60:
            assert "neutral" not in cap.holds        # 2 s of STILLNESS needed: the shaky second doesn't count
    assert rejected > 0 and "neutral" in cap.holds and cap.holds["neutral"][0] == pytest.approx(0.0, abs=0.005)
    for _ in range(30):
        assert cap.update(t, n + [0.03, 0, 0]) is None; t += 1 / 30    # only 3 cm: "move further"
    assert cap.note == "move further" and not cap.done


def test_profile_round_trip(tmp_path):
    from rover_intent.body.calibration import Profile
    p = Profile(R=np.eye(3).tolist(), scale={"right": 1.5}, neutral=[0, 0.3, 0], reach={"right": 0.2},
                residual_deg=1.2, per_dir_deg={"right": 1.2}, arm_profile="dev_arm")
    p.save(tmp_path / "me.json")
    q = Profile.load(tmp_path / "me.json")
    assert q.scale == p.scale and q.arm_profile == "dev_arm" and q.R == p.R


def test_app_calibration_flow_end_to_end_mock(tmp_path):
    """Real app loop (mock arm) over UDP: C -> positioning -> guided capture of a synthetic person -> fit ->
    verifying with a magenta target and the arm following through the calibrated mapper."""
    import json, socket, threading, time
    from rover_intent import app
    from rover_intent.config import load_config
    cfg = load_config()
    port = 47196
    cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
    cfg["body"]["profile"] = str(tmp_path / "{arm}_test.json")
    cfg["body"]["clutch"] = {"calib_s": 0.3, "still_m": 0.03, "palm_s": 1.0, "raise_m": 0.05}
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
    th = threading.Thread(target=app.run, args=(cfg, "mock", 22.0), daemon=True)
    th.start(); time.sleep(0.4)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(0.001)
    A = ("127.0.0.1", port)
    s.sendto(json.dumps({"kind": "clutch", "t": time.time(), "action": "calibrate"}).encode(), A)
    neutral = np.array([0.05, 0.30, -0.10])
    reach = {"neutral": [0, 0, 0], "right": [0.25, 0, 0], "up": [0, 0, 0.22], "forward": [0, 0.20, 0],
             "left": [-0.18, 0, 0], "down": [0, 0, -0.20]}
    last, seq, t_end = {}, 0, time.time() + 20
    while time.time() < t_end:
        cal = last.get("calib") or {}
        step = cal.get("step") if cal.get("state") == "capturing" else "neutral"
        p = _person(neutral + np.array(reach.get(step, [0, 0, 0])), yaw_deg=15).model_dump()
        seq += 1
        p.update(kind="body", t=time.time(), seq=seq, visible=True, in_position=True, hand_open=0.9)
        p["elbow"] = [p["wrist"][0], p["wrist"][1] + 0.1, p["wrist"][2]]   # wrist ABOVE elbow + open hand = palm-up pose
        s.sendto(json.dumps(p).encode(), A); time.sleep(1 / 30)
        try:
            while True:
                m = json.loads(s.recvfrom(65536)[0])
                if m.get("kind") == "status":
                    last = m
        except (socket.timeout, BlockingIOError):
            pass
        if (last.get("calib") or {}).get("state") == "capturing":
            assert last.get("clutch") == "idle", "the arm must stay frozen while capturing (palm-up poses!)"
        if (last.get("calib") or {}).get("state") == "verifying" and last.get("clutch") == "following":
            break
    cal = last.get("calib") or {}
    assert cal.get("state") == "verifying", cal
    assert last["clutch"] == "following" and last["mapping"] == "CalibratedMapper"
    assert cal["of"] == 4
    th.join(timeout=25)
